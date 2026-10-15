"""Local web frontend driving the same CLI (§16, §18)."""

from video_intel.web.server import WebApp, WebError, serve

__all__ = ["WebApp", "WebError", "serve"]
