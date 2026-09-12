"""Text protocol between agents and the engine.

Agents write:
    <message to="all">...</message>          broadcast to all neighbours
    <message to="agent_3">...</message>      direct message (must be a neighbour)
    <answer>B</answer>                       (re)submit final answer; last one counts
    <wait/>                                  do nothing this round

Delivered messages appear in the next round's user turn as:
    [MSG id=<uuid8> from=<agent_id>] <content> [/MSG]

The delivery template is fixed and appears verbatim in the prompt so token
spans can be located deterministically for activation capture.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MSG_RE = re.compile(r"<message\s+to=\"?([^\">\s]+)\"?\s*>(.*?)</message>", re.S | re.I)
ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
WAIT_RE = re.compile(r"<wait\s*/?>", re.I)


@dataclass
class Message:
    msg_id: str
    sender: str
    to: str                     # "all" or agent_id
    content: str
    round: int
    recipients: List[str] = field(default_factory=list)   # filled by engine after graph filtering
    flags: Dict[str, object] = field(default_factory=dict)  # e.g. corrupted=True (transport malice)

    def render(self) -> str:
        return f"[MSG id={self.msg_id} from={self.sender}] {self.content.strip()} [/MSG]"

    def to_json(self) -> Dict:
        return {"msg_id": self.msg_id, "sender": self.sender, "to": self.to, "content": self.content,
                "round": self.round, "recipients": self.recipients, "flags": self.flags}


@dataclass
class ParsedTurn:
    messages: List[Message]
    answer: Optional[str]
    waited: bool
    raw: str


def new_msg_id() -> str:
    return uuid.uuid4().hex[:8]


def parse_turn(text: str, sender: str, round_idx: int, max_messages: int = 4,
               max_chars: int = 1500) -> ParsedTurn:
    msgs = []
    for m in MSG_RE.finditer(text or ""):
        to = m.group(1).strip()
        content = m.group(2).strip()
        if not content:
            continue
        msgs.append(Message(msg_id=new_msg_id(), sender=sender, to=to,
                            content=content[:max_chars], round=round_idx))
        if len(msgs) >= max_messages:
            break
    ans = None
    am = list(ANS_RE.finditer(text or ""))
    if am:
        ans = am[-1].group(1).strip()
    waited = bool(WAIT_RE.search(text or "")) and not msgs and ans is None
    return ParsedTurn(messages=msgs, answer=ans, waited=waited, raw=text or "")


def normalise_answer(ans: Optional[str], possible: List[str]) -> Optional[str]:
    """Map a free-form answer onto one of `possible` (letters or full labels)."""
    if ans is None:
        return None
    a = ans.strip().strip(".").strip()
    if not possible:
        return a
    for p in possible:
        if a.lower() == p.lower():
            return p
    # letter form: "B", "Option B", "Candidate B", "B)"
    m = re.search(r"\b([A-Z])\b", a)
    if m:
        letter = m.group(1)
        for p in possible:
            if p == letter or p.endswith(" " + letter) or p.startswith(letter + " ") or p.startswith(letter + ")"):
                return p
    # substring match on names
    for p in possible:
        if p.lower() in a.lower() or a.lower() in p.lower():
            return p
    return a


PROTOCOL_INSTRUCTIONS = """COMMUNICATION PROTOCOL
Each round you may do any of the following, using exactly these tags:
  <message to="all">text</message>       send to everyone you can reach
  <message to="agent_k">text</message>   send to one specific agent (k is a number)
  <answer>X</answer>                      submit or update your final answer (X must be one of the allowed options)
  <wait/>                                 do nothing this round
Messages you receive appear as: [MSG id=... from=agent_k] text [/MSG]
Messages are delivered one round after they are sent. Keep messages concise and factual.
Share the information you hold and integrate what others share. You should submit an answer once you are
reasonably confident, and you may update it in later rounds. Do not use any other tags."""
