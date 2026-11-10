"""Busy-service and timeout behaviour, driven with real concurrent requests.

SYNTHETIC adapter (see conftest.py): the "model" is a gate the test opens and
closes. These tests check scheduling guarantees, not model behaviour.
"""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest

from impactlens.service import create_app

from .conftest import (
    API_KEY,
    SyntheticAdapter,
    canvas_with_block,
    make_settings,
    synthetic_loader,
)

pytestmark = pytest.mark.contract

AUTH = {"Authorization": f"Bearer {API_KEY}"}
PHOTO = canvas_with_block(64, 64, (8, 8, 32, 32))


async def _post(client: httpx.AsyncClient) -> httpx.Response:
    return await client.post(
        "/predict", headers=AUTH, files={"image": ("p.png", PHOTO, "image/png")}
    )


async def _until(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


def _run(scenario, **settings) -> None:
    adapter = SyntheticAdapter()
    app = create_app(make_settings(**settings), model_loader=synthetic_loader(adapter))

    async def main() -> None:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                await scenario(app, adapter, client)

    asyncio.run(main())


def test_requests_beyond_capacity_get_429_and_only_one_inference_runs_at_a_time():
    async def scenario(app, adapter, client):
        service = app.state.inference
        adapter.gate = threading.Event()  # hold the "model" busy

        running = asyncio.create_task(_post(client))
        await _until(adapter.entered.is_set)
        queued = asyncio.create_task(_post(client))
        await _until(lambda: service._admitted == 2)

        # Capacity is 1 running + 1 waiting: the third request is refused at once.
        refused = await asyncio.wait_for(_post(client), timeout=2)
        assert refused.status_code == 429
        assert refused.json()["error"]["code"] == "server_busy"
        assert int(refused.headers["retry-after"]) >= 1
        assert "detections" not in refused.json()
        assert adapter.calls == 1  # the queued request has not reached the model

        adapter.gate.set()
        first, second = await asyncio.gather(running, queued)
        assert first.status_code == 200 and second.status_code == 200
        assert second.json()["latency_ms"]["queue"] > 0
        assert adapter.calls == 2
        assert service._admitted == 0

    _run(scenario, max_pending_requests=1)


def test_zero_queue_means_a_second_request_is_refused_immediately():
    async def scenario(app, adapter, client):
        adapter.gate = threading.Event()
        running = asyncio.create_task(_post(client))
        await _until(adapter.entered.is_set)
        refused = await _post(client)
        assert refused.status_code == 429 and refused.json()["error"]["code"] == "server_busy"
        adapter.gate.set()
        assert (await running).status_code == 200

    _run(scenario, max_pending_requests=0)


def test_timeout_returns_504_and_the_slot_stays_held_until_the_model_call_returns():
    async def scenario(app, adapter, client):
        service = app.state.inference
        adapter.gate = threading.Event()  # the model call will outlive the request

        timed_out = await _post(client)
        assert timed_out.status_code == 504
        assert timed_out.json()["error"]["code"] == "inference_timeout"
        assert "detections" not in timed_out.json()

        # The abandoned call is still running, so its slot must not be reusable yet.
        assert service._slot.locked()
        assert service._admitted == 0
        waiting = await _post(client)
        assert waiting.status_code == 504  # timed out in the queue, never reached the model
        assert adapter.calls == 1

        adapter.gate.set()
        await _until(lambda: not service._slot.locked())
        adapter.gate = None
        recovered = await _post(client)
        assert recovered.status_code == 200
        assert adapter.calls == 2

    _run(scenario, inference_timeout_s=0.3, max_pending_requests=2)


def test_slow_upload_is_cut_off_with_408():
    async def scenario(app, adapter, client):
        boundary = "slowboundary"
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="p.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()

        async def trickle():
            yield head
            await asyncio.sleep(1.0)  # longer than the body timeout
            yield PHOTO + f"\r\n--{boundary}--\r\n".encode()

        response = await client.post(
            "/predict",
            headers={**AUTH, "Content-Type": f"multipart/form-data; boundary={boundary}"},
            content=trickle(),
        )
        assert response.status_code == 408
        assert response.json()["error"]["code"] == "request_timeout"
        assert adapter.calls == 0
        assert app.state.inference._admitted == 0  # the place in the queue was given back

    _run(scenario, request_body_timeout_s=0.2)


def test_model_is_loaded_once_and_closed_on_shutdown():
    loads = []
    adapter = SyntheticAdapter()

    def loader(settings):
        loads.append(1)
        return synthetic_loader(adapter)(settings)

    app = create_app(make_settings(), model_loader=loader)

    async def main() -> None:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                for _ in range(3):
                    assert (await _post(client)).status_code == 200
        assert adapter.closed

    asyncio.run(main())
    assert len(loads) == 1 and adapter.calls == 3
