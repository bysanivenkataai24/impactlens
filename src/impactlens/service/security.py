"""Authentication, rate limiting and response hardening for the service.

Scope and honesty about it:

* One shared ``API_KEY`` is a *developer-pilot* control. A public product needs
  per-user authentication, short-lived tokens and quotas at a gateway.
* The rate limiter lives in process memory. It is per replica, resets on
  restart and is not a substitute for a distributed limiter.
* CORS is configured elsewhere and only tells browsers which pages may read
  responses. It is not authentication and does not stop non-browser clients.
"""

from __future__ import annotations

import hmac
import math
import time
import uuid
from collections import deque
from collections.abc import Callable

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def presented_api_key(request: Request) -> str | None:
    """Read the key from ``Authorization: Bearer <key>`` or ``X-API-Key``."""
    authorization = request.headers.get("authorization")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    header = request.headers.get("x-api-key")
    return header.strip() if header and header.strip() else None


def api_key_matches(presented: str | None, expected: str | None) -> bool:
    """Constant-time comparison. False whenever either side is missing."""
    if not presented or not expected:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def client_address(request: Request, trusted_proxy_hops: int) -> str:
    """Best-effort client address for rate limiting.

    With ``trusted_proxy_hops == 0`` the socket peer is used and forwarding
    headers are ignored (they are client-controlled). With ``n`` trusted
    proxies in front, the n-th address from the right of ``X-Forwarded-For`` is
    the one appended by the outermost trusted proxy.
    """
    peer = request.client.host if request.client else "unknown"
    if trusted_proxy_hops <= 0:
        return peer
    forwarded = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",")]
    forwarded = [p for p in forwarded if p]
    if len(forwarded) >= trusted_proxy_hops:
        return forwarded[-trusted_proxy_hops][:64]
    return peer


class SlidingWindowRateLimiter:
    """At most ``limit`` hits per ``window_s`` seconds per key, in process memory."""

    def __init__(
        self,
        limit: int,
        window_s: float = 60.0,
        *,
        max_keys: int = 20_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self._max_keys = max_keys
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def hit(self, key: str) -> int | None:
        """Record one attempt. Returns ``None`` if allowed, else seconds until retry."""
        now = self._clock()
        cutoff = now - self.window_s
        hits = self._hits.get(key)
        if hits is None:
            if len(self._hits) >= self._max_keys:
                self._evict(cutoff)
            hits = self._hits[key] = deque()
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self.limit:
            return max(1, math.ceil(hits[0] + self.window_s - now))
        hits.append(now)
        return None

    def _evict(self, cutoff: float) -> None:
        stale = [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]
        for key in stale:
            del self._hits[key]
        # Still full of active keys: drop the oldest entries to stay bounded.
        while len(self._hits) >= self._max_keys:
            del self._hits[next(iter(self._hits))]


_NO_STORE_PREFIXES = ("/health", "/ready", "/predict", "/auth", "/openapi.json")


def content_security_policy(frame_ancestors: str) -> str:
    return "; ".join(
        [
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self'",
            "img-src 'self' blob: data:",
            "font-src 'self'",
            # The connection page lets the operator point the UI at their own service URL.
            "connect-src 'self' https: http://localhost:* http://127.0.0.1:*",
            "worker-src 'self' blob:",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            f"frame-ancestors {frame_ancestors}",
        ]
    )


class SecurityHeadersMiddleware:
    """Assigns a request id and adds security / caching headers to every response."""

    def __init__(self, app: ASGIApp, *, frame_ancestors: str = "'none'") -> None:
        self.app = app
        self._static_headers: list[tuple[bytes, bytes]] = [
            (b"x-content-type-options", b"nosniff"),
            (b"referrer-policy", b"no-referrer"),
            (b"content-security-policy", content_security_policy(frame_ancestors).encode()),
            (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=()"),
            (b"cross-origin-opener-policy", b"same-origin"),
            # Ignored by browsers over plain HTTP; effective behind an HTTPS proxy.
            (b"strict-transport-security", b"max-age=31536000"),
        ]
        if frame_ancestors.strip() == "'none'":
            self._static_headers.append((b"x-frame-options", b"DENY"))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        path: str = scope.get("path", "")

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers: list[tuple[bytes, bytes]] = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                for name, value in self._static_headers:
                    if name not in present:
                        headers.append((name, value))
                headers.append((b"x-request-id", request_id.encode()))
                if b"cache-control" not in present:
                    if path.startswith("/assets/"):
                        cache = b"public, max-age=31536000, immutable"
                    elif path.startswith(_NO_STORE_PREFIXES):
                        cache = b"no-store"
                    else:
                        cache = b"no-cache"
                    headers.append((b"cache-control", cache))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)
