"""Viewer compatibility exports."""

__all__ = ["InteractiveViewer"]


def __getattr__(name: str):
    if name == "InteractiveViewer":
        from .engine import InteractiveViewer

        return InteractiveViewer
    raise AttributeError(name)
