import re
from typing import Dict, List, Set

def norm_key(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def build_canonical_map(values: List[str]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for v in values:
        k = norm_key(v)
        if k and k not in m:
            m[k] = v
    return m

def canonicalize_list(tokens: List[str], canonical_map: Dict[str, str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for t in tokens or []:
        k = norm_key(t)
        if not k:
            continue
        v = canonical_map.get(k)
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return out
