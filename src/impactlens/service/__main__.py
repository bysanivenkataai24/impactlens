"""``python -m impactlens.service`` - run the inference service with uvicorn.

One worker process is the supported shape: the model is loaded once and the
concurrency and rate limits are per process. Scale by adding replicas behind a
gateway, not by raising a worker count here.
"""

from __future__ import annotations

import logging
import os

import uvicorn


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(
        "impactlens.service.app:create_app",
        factory=True,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "7860")),
        workers=1,
        # Access logs contain client addresses; request metadata is logged by the app instead.
        access_log=False,
        server_header=False,
        proxy_headers=False,
        limit_concurrency=int(os.environ.get("UVICORN_LIMIT_CONCURRENCY", "100")),
        timeout_keep_alive=15,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
