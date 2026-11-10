"""HTTP inference service. ``create_app`` is the only public entry point."""

from .app import ApiError, create_app

__all__ = ["ApiError", "create_app"]
