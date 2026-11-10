"""Rubric checks. Each check returns (passed, explanation)."""
import re


def _wc(text):
    return len(text.split())


def run_check(check: dict, text: str):
    kind = check["type"]
    low = text.lower()
    if kind == "contains_any":
        hit = [v for v in check["values"] if v.lower() in low]
        return bool(hit), f"expected one of {check['values']}" if not hit else f"found {hit[0]!r}"
    if kind == "contains_all":
        missing = [v for v in check["values"] if v.lower() not in low]
        return not missing, f"missing {missing}" if missing else "all present"
    if kind == "not_contains_any":
        hit = [v for v in check["values"] if v.lower() in low]
        return not hit, f"forbidden content present: {hit}" if hit else "no forbidden content"
    if kind == "regex":
        ok = re.search(check["pattern"], text, flags=re.IGNORECASE | re.DOTALL) is not None
        return ok, f"pattern {check['pattern']!r} " + ("matched" if ok else "did not match")
    if kind == "max_words":
        ok = _wc(text) <= check["value"]
        return ok, f"{_wc(text)} words (limit {check['value']})"
    raise ValueError(f"unknown check type: {kind}")


FAILURE_CLASS = {
    "contains_any": "missing expected content",
    "contains_all": "missing expected content",
    "not_contains_any": "forbidden content / unsafe output",
    "regex": "format or instruction violation",
    "max_words": "format or instruction violation",
}
