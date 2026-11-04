"""WS-16: GET /pool/status on the fork.

The shape is pinned: {pool, router_id, connected, last_contact_ms, classes,
serving, originating} - counts, tags and ids only (anti-memory). The serving
counters move when this router serves an offered job, the originating counters
move when it intakes one and holds the relay open, and the excluded mark
(D-F1) does not change the local status (it is a /v1/models concern).
"""

import json
import threading

import pytest
import requests

from test_pool_dispatch import (
    LOCAL_MODEL_ID,
    POOL_TEXT_MD,
    _Job,
    _load_model,
    _occupy_slot,
    _start_router,
    _wait_for_entry,
    _wait_intake,
    _wait_model_status,
    _wait_until,
    _write_preset,
    FakeCoordinator,
)
from utils import *


@pytest.fixture
def coordinator():
    coord = FakeCoordinator()
    yield coord
    coord.stop()


def _status(server: ServerProcess) -> dict:
    """GET /pool/status: the pinned shape, or None when the endpoint is absent."""
    res = server.make_request("GET", "/pool/status")
    if res.status_code != 200:
        return None
    return res.body


def _member_count(status: dict, tag: str = POOL_TEXT_MD) -> int | None:
    entry = next((c for c in status["classes"] if c["tag"] == tag), None)
    return entry["member_count"] if entry else None


def _stage_job(coordinator: FakeCoordinator, job_id: str, body: dict) -> None:
    """Stage a job with its body on the fake coordinator, for the router to pull."""
    with coordinator.lock:
        job = _Job(job_id)
        job.tag = POOL_TEXT_MD
        job.body = json.dumps(body).encode()
        coordinator.jobs[job_id] = job


def test_status_pool_off(tmp_path):
    """no --pool-server: the endpoint still answers (router mode), and reports
    pool:false with zeroes, so the UI needs no 404 handling"""
    os.environ.pop("LLAMA_POOL_TOKEN", None)
    os.environ.pop("LLAMA_POOL_LINK_MODE", None)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    server = ServerPreset.router()
    server.models_dir = str(models_dir)
    server.log_path = str(tmp_path / "server.log")
    server.start()
    try:
        status = _status(server)
        assert status is not None, "GET /pool/status must answer in router mode"
        assert status["pool"] is False
        assert status["router_id"] == ""
        assert status["connected"] is False
        assert status["last_contact_ms"] == 0
        assert status["classes"] == []
        assert status["serving"] == {"pending": 0, "served_total": 0, "tokens_served_total": 0}
        assert status["originating"] == {"pending": 0, "total": 0}
    finally:
        server.stop()


def test_status_shape_when_configured(tmp_path, coordinator):
    """the pinned keys, the router id, and the merged class advertisement
    (tag + member_count only, no names or content)"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, POOL_TEXT_MD)
        router_id = coordinator.wait_for("handshake", lambda e: True)["payload"]["router_id"]

        status = _status(server)
        assert status is not None
        assert set(status.keys()) == {
            "pool",
            "router_id",
            "connected",
            "last_contact_ms",
            "classes",
            "serving",
            "originating"
        }
        assert status["pool"] is True
        assert status["router_id"] == router_id
        assert isinstance(status["connected"], bool)
        assert isinstance(status["last_contact_ms"], int)

        assert set(status["serving"].keys()) == {"pending", "served_total", "tokens_served_total"}
        assert set(status["originating"].keys()) == {"pending", "total"}

        # the local member's class is advertised, with its own numbers (D-C8)
        count = _member_count(status)
        assert count is not None and count >= 1
        for c in status["classes"]:
            assert set(c.keys()) == {"tag", "member_count"}, c
            assert isinstance(c["member_count"], int) and c["member_count"] >= 1

        # counts are non-negative integers
        for section in ("serving", "originating"):
            for value in status[section].values():
                assert isinstance(value, int) and value >= 0
    finally:
        server.stop()


def test_status_link_state(tmp_path, coordinator):
    """connected tracks the open subscribe link; last_contact_ms stays fresh
    while the link is up (the coordinator heartbeats)"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, POOL_TEXT_MD)
        coordinator.wait_for("handshake", lambda e: True)
        coordinator.wait_for_sse()

        # the SSE round is in flight: poll until the link reports open
        _wait_until(
            lambda: (_status(server) or {}).get("connected") is True,
            timeout=60,
            what="the subscribe link to report open"
        )
        status = _status(server)
        # the fake heartbeats every ~5 s, so the contact age stays small while up
        assert 0 <= status["last_contact_ms"] < 20_000, status
    finally:
        server.stop()


def test_status_serving_counters_move(tmp_path, coordinator):
    """an offered job this router serves: served_total and tokens_served_total
    move, pending is back to 0, nothing was originated"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, POOL_TEXT_MD)
        coordinator.wait_for("handshake", lambda e: True)
        coordinator.wait_for_sse()

        _stage_job(coordinator, "job-serve-1", {
            "model": POOL_TEXT_MD,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "status serving test"}],
        })
        coordinator.send_work_offer("job-serve-1", tag=POOL_TEXT_MD, est_ctx=256)

        comp = coordinator.wait_for("completion", lambda e: e["payload"]["job_id"] == "job-serve-1")
        assert comp["payload"]["ok"] is True

        status = _status(server)
        assert status["serving"]["pending"] == 0
        assert status["serving"]["served_total"] == 1
        assert status["serving"]["tokens_served_total"] >= 1
        assert status["originating"] == {"pending": 0, "total": 0}
    finally:
        server.stop()


def test_status_origin_counters_move(tmp_path, coordinator):
    """a request this router originates: pending is 1 while the relay is held
    open, total counts the intake, and both settle when the relay closes (the
    job comes back to this router and is served)"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, POOL_TEXT_MD)
        coordinator.wait_for("handshake", lambda e: True)
        coordinator.wait_for_sse()

        # occupy the (single) slot so the pool request intakes on the coordinator
        t_long, _ = _occupy_slot(server, LOCAL_MODEL_ID)
        try:
            client: dict = {}

            def _client() -> None:
                r = requests.post(
                    server.make_url("/v1/chat/completions"),
                    json={
                        "model": POOL_TEXT_MD,
                        "stream": True,
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "status origin test"}],
                    },
                    stream=True,
                    timeout=180,
                )
                client["status"] = r.status_code
                if r.status_code == 200:
                    for line in r.iter_lines():
                        pass

            t_client = threading.Thread(target=_client, daemon=True)
            t_client.start()

            intake = _wait_intake(coordinator)
            job_id = intake["job_id"]

            # the relay is held open: originating counts the intake as pending
            _wait_until(
                lambda: (_status(server) or {}).get("originating", {}).get("pending") == 1,
                timeout=30,
                what="the held-open relay to count as originating pending"
            )
            status = _status(server)
            assert status["originating"]["total"] == 1

            # free the member; the job is offered back and this router serves it
            t_long.join(timeout=180)
            coordinator.send_work_offer(job_id, tag=POOL_TEXT_MD, est_ctx=256,
                                        conv_key=intake["metadata"].get("conv_key"))
            t_client.join(timeout=180)
            assert client["status"] == 200, client.get("status")

            # the relay closed: pending is back to 0, the total stands,
            # and the serving side counts the job it served
            _wait_until(
                lambda: (_status(server) or {}).get("originating", {}).get("pending") == 0,
                timeout=30,
                what="the originating pending to settle after the relay closed"
            )
            status = _status(server)
            assert status["originating"]["total"] == 1
            assert status["serving"]["served_total"] == 1
            assert status["serving"]["tokens_served_total"] >= 1
        finally:
            t_long.join(timeout=180)
    finally:
        server.stop()


def test_status_excluded_interplay(tmp_path, coordinator):
    """D-F1: the excluded mark is a /v1/models concern. The local status keeps
    counting the member (it stays loaded) and the local dispatch keeps
    serving; the mark neither shows up nor changes the counters"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, LOCAL_MODEL_ID)
        coordinator.wait_for("handshake", lambda e: True)
        coordinator.wait_for_sse()

        assert _member_count(_status(server)) is not None

        # withhold: /models/unload on the pool member is the mark, not an unload
        res = server.make_request("POST", "/models/unload", data={"model": LOCAL_MODEL_ID})
        assert res.status_code == 200, res.body
        assert res.body.get("success") is True
        coordinator.wait_for("exclusion_update", lambda e:
            e["payload"]["model_id"] == LOCAL_MODEL_ID and e["payload"]["excluded"] is True)
        _wait_model_status(server, LOCAL_MODEL_ID, "excluded")

        # the local status is unchanged: the member still counts, and nothing
        # about the mark is exposed here
        assert _member_count(_status(server)) is not None

        # the local dispatch keeps serving offers for the member
        _stage_job(coordinator, "job-excl-1", {
            "model": POOL_TEXT_MD,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "excluded serve test"}],
        })
        coordinator.send_work_offer("job-excl-1", tag=POOL_TEXT_MD, est_ctx=256)
        coordinator.wait_for("completion", lambda e: e["payload"]["job_id"] == "job-excl-1")
        status = _status(server)
        assert status["serving"]["served_total"] == 1
    finally:
        server.stop()
