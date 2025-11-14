import json
import re
from typing import Tuple


REASONING_OPEN = "<reasoning>"
REASONING_CLOSE = "</reasoning>"
FINAL_OPEN = "<final>"
FINAL_CLOSE = "</final>"


def extract_trace_and_answer(text: str) -> Tuple[str, str]:
    """Extract reasoning trace and final answer from tagged text.

    Returns empty strings if tags are missing.
    """
    trace = ""
    final = ""
    m_trace = re.search(r"<reasoning>([\s\S]*?)</reasoning>", text)
    if m_trace:
        trace = m_trace.group(1).strip()
    m_final = re.search(r"<final>([\s\S]*?)</final>", text)
    if m_final:
        final = m_final.group(1).strip()
    return trace, final


def normalized_answer(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\s+", " ", s)
    s = s.replace(",", "")
    return s.lower()


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")