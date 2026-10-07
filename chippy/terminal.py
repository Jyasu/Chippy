"""Helpers for printing model-controlled text to the terminal."""


def sanitize(text) -> str:
    """
    Escapes control characters (ANSI escapes, carriage returns, bidi overrides) so
    model output cannot redraw the terminal or spoof the approval prompt.
    """
    out = []
    for ch in str(text):
        if ch in "\n\t" or ch.isprintable():
            out.append(ch)
            continue
        code = ord(ch)
        if code <= 0xFF:
            out.append(f"\\x{code:02x}")
        elif code <= 0xFFFF:
            out.append(f"\\u{code:04x}")
        else:
            out.append(f"\\U{code:08x}")
    return "".join(out)


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."
