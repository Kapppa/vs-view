import sys


def is_preview() -> bool:
    """Check if the current script is running in a preview environment (VSView only)."""
    return bool(sys.modules.get("__vsview__"))
