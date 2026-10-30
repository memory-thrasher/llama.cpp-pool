import json
import os
import queue
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from utils import *

MODEL_ID = "ggml-org/tinygemma3-GGUF:Q8_0"
POOL_TOKEN = "test-member-token"
ROUTER_NAME = "test-router"
# WS-04 will consume this; the fake serves it now so the endpoint is exercised
CLASS_ADVERTISEMENT_PATH = "/home/sid/projects/llama-pool/pool-coordinator/protocol/examples/class_advertisement.json"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeCoordinator:
    """A minimal pool coordinator for the link test.

    - POST /pool/events records the envelope
    - GET /pool/subscribe serves work offers: SSE when the Accept header asks
      for text/event-stream, long-poll (JSON array) otherwise
    """

    def __init__(self):
        self.port = _free_port()
        self.lock = threading.Lock()
        self.received: list[dict] = []
        self.last_auth = ""
        self.pending: list[dict] = []
        self.sse_subs: list[queue.Queue] = []
        self.subscribe_count = 0
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), self._make_handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def received_of(self, etype: str) -> list[dict]:
        with self.lock:
            return [e for e in self.received if e.get("type") == etype]

    def send_work_offer(self, job_id: str, tag: str = "test") -> None:
        env = {"type": "work_offer", "payload": {"job_id": job_id, "tag": tag}}
        with self.lock:
            subs = list(self.sse_subs)
            if not subs:
                self.pending.append(env)
        for q in subs:
            q.put(env)

    def wait_for(self, etype: str, pred, timeout: float = 60.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            for env in self.received_of(etype):
                if pred(env):
                    return env
            time.sleep(0.05)
        raise AssertionError(
            f"timed out waiting for {etype!r}; received: {json.dumps(self.received)[:4000]}"
        )

    def wait_for_sse(self, timeout: float = 60.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if self.sse_subs:
                    return
            time.sleep(0.05)
        raise AssertionError("timed out waiting for the router SSE subscription")

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _send_json(self, obj, status: int = 200) -> None:
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path != "/pool/events":
                    self._send_json({"error": "not found"}, 404)
                    return
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                try:
                    env = json.loads(raw)
                except json.JSONDecodeError:
                    self._send_json({"error": "bad json"}, 400)
                    return
                with outer.lock:
                    outer.received.append(env)
                    outer.last_auth = self.headers.get("Authorization", "")
                self._send_json({"ok": True})

            def do_GET(self):
                if self.path == "/pool/classes":
                    try:
                        with open(CLASS_ADVERTISEMENT_PATH, "r") as f:
                            self._send_json(json.loads(f.read()))
                    except (OSError, json.JSONDecodeError):
                        self._send_json({"protocol": 1, "classes": []})
                    return
                if self.path != "/pool/subscribe":
                    self._send_json({"error": "not found"}, 404)
                    return
                with outer.lock:
                    outer.subscribe_count += 1
                if "text/event-stream" in self.headers.get("Accept", ""):
                    self._sse()
                else:
                    self._long_poll()

            def _sse(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                q: "queue.Queue" = queue.Queue()
                with outer.lock:
                    # deliver offers that accumulated before this subscription
                    for env in list(outer.pending):
                        q.put(env)
                    outer.pending = []
                    outer.sse_subs.append(q)
                try:
                    while True:
                        try:
                            env = q.get(timeout=5)
                        except queue.Empty:
                            self.wfile.write(f": hb {int(time.time() * 1000)}\n\n".encode())
                            self.wfile.flush()
                            continue
                        self.wfile.write(f"data: {json.dumps(env)}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with outer.lock:
                        if q in outer.sse_subs:
                            outer.sse_subs.remove(q)

            def _long_poll(self):
                deadline = time.time() + 5.0
                got: list[dict] = []
                while time.time() < deadline and not got:
                    with outer.lock:
                        if outer.pending:
                            got = list(outer.pending)
                            outer.pending = []
                    if not got:
                        time.sleep(0.05)
                self._send_json(got)

        return Handler


def _set_pool_env(link_mode: str | None = None) -> None:
    os.environ["LLAMA_POOL_TOKEN"] = POOL_TOKEN
    if link_mode is not None:
        os.environ["LLAMA_POOL_LINK_MODE"] = link_mode
    else:
        os.environ.pop("LLAMA_POOL_LINK_MODE", None)


@pytest.fixture(autouse=True)
def _restore_env():
    saved = {k: os.environ.get(k) for k in ("LLAMA_POOL_TOKEN", "LLAMA_POOL_LINK_MODE")}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def coordinator():
    coord = FakeCoordinator()
    yield coord
    coord.stop()


def _start_router(tmp_path, coordinator, link_mode: str | None = None) -> ServerProcess:
    _set_pool_env(link_mode)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    server = ServerPreset.router()
    server.models_dir = str(models_dir)
    server.pool_server = coordinator.url()
    server.pool_name = ROUTER_NAME
    server.log_path = str(tmp_path / "server.log")
    server.start()
    return server


def _wait_log(server: ServerProcess, needle: str, timeout: float = 60.0) -> str:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            with open(server.log_path, "r", errors="replace") as f:
                last = f.read()
        except FileNotFoundError:
            last = ""
        if needle in last:
            return last
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {needle!r} in the server log; log tail:\n{last[-4000:]}")


def _model_status(server: ServerProcess, model_id: str) -> str | None:
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    for item in res.body.get("data", []):
        if item.get("id") == model_id or item.get("model") == model_id:
            return item["status"]["value"]
    return None


def _wait_model_status(server: ServerProcess, model_id: str, desired: str, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = _model_status(server, model_id)
        if last == desired:
            return
        time.sleep(0.5)
    raise AssertionError(f"timed out waiting for {model_id} to reach {desired!r}, last status: {last}")


def _load_model(server: ServerProcess, model_id: str) -> None:
    res = server.make_request("POST", "/models/load", data={"model": model_id})
    assert res.status_code == 200
    assert res.body.get("success") is True
    _wait_model_status(server, model_id, "loaded")


def _model_in_capacity(env: dict, model_id: str) -> dict | None:
    for m in env["payload"]["models"]:
        if m["model_id"] == model_id:
            return m
    return None


def test_pool_off_by_default(tmp_path, coordinator):
    """no --pool-server: the router behaves exactly as before, the coordinator sees nothing"""
    os.environ.pop("LLAMA_POOL_TOKEN", None)
    os.environ.pop("LLAMA_POOL_LINK_MODE", None)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    server = ServerPreset.router()
    server.models_dir = str(models_dir)
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["role"] == "router"
    time.sleep(1.0)
    assert coordinator.received == []
    server.stop()


def test_pool_handshake(tmp_path, coordinator):
    server = _start_router(tmp_path, coordinator)
    try:
        env = coordinator.wait_for("handshake", lambda e: True)
        p = env["payload"]
        assert p["proto"] == 1
        assert UUID_RE.match(p["router_id"])
        assert p["router_name"] == ROUTER_NAME
        assert isinstance(p["software"], str) and p["software"]
        assert isinstance(p["models"], list)
        assert coordinator.last_auth == f"Bearer {POOL_TOKEN}"

        # the router id is persisted in <models_dir>/.pool/router-id
        id_file = tmp_path / "models" / ".pool" / "router-id"
        assert id_file.read_text().strip() == p["router_id"]
    finally:
        server.stop()


def test_pool_router_id_stable_across_restart(tmp_path, coordinator):
    first_id = None
    server = _start_router(tmp_path, coordinator)
    try:
        first_id = coordinator.wait_for("handshake", lambda e: True)["payload"]["router_id"]
    finally:
        server.stop()

    server = _start_router(tmp_path, coordinator)
    try:
        env = coordinator.wait_for("handshake", lambda e: e["payload"]["router_id"] == first_id)
        assert env["payload"]["router_id"] == first_id
    finally:
        server.stop()


def test_pool_capacity_on_model_load(tmp_path, coordinator):
    server = _start_router(tmp_path, coordinator)
    try:
        coordinator.wait_for("handshake", lambda e: True)
        _load_model(server, MODEL_ID)

        env = coordinator.wait_for("capacity", lambda e: _model_in_capacity(e, MODEL_ID) is not None)
        m = _model_in_capacity(env, MODEL_ID)
        assert m["status"] == "loaded"
        assert m["n_ctx"] >= 1
        assert m["n_embd"] >= 1
        assert m["n_slots"] >= 1
        assert m["busy_slots"] == 0
        assert "tags" in m

        # the OpenAI surface is unchanged by the pool: the local model is listed as usual
        res = server.make_request("GET", "/v1/models")
        assert res.status_code == 200
        assert MODEL_ID in [m["id"] for m in res.body.get("data", [])]
    finally:
        server.stop()


def test_pool_capacity_on_model_unload(tmp_path, coordinator):
    server = _start_router(tmp_path, coordinator)
    try:
        coordinator.wait_for("handshake", lambda e: True)
        _load_model(server, MODEL_ID)
        coordinator.wait_for("capacity", lambda e: _model_in_capacity(e, MODEL_ID) is not None)

        unload_res = server.make_request("POST", "/models/unload", data={"model": MODEL_ID})
        assert unload_res.status_code == 200
        assert unload_res.body.get("success") is True
        _wait_model_status(server, MODEL_ID, "unloaded")

        def model_gone(env):
            return all(m["model_id"] != MODEL_ID for m in env["payload"]["models"])

        coordinator.wait_for("capacity", model_gone, timeout=60)
    finally:
        server.stop()


def test_pool_work_offer_via_sse(tmp_path, coordinator):
    server = _start_router(tmp_path, coordinator)
    try:
        coordinator.wait_for("handshake", lambda e: True)
        coordinator.wait_for_sse()
        coordinator.send_work_offer("job-sse-1", tag="link-test")
        _wait_log(server, "pool: offer job_id=job-sse-1 tag=link-test")
    finally:
        server.stop()


def test_pool_degraded_when_coordinator_unreachable(tmp_path):
    """--pool-server at a closed port: the router still starts and serves its model"""
    _set_pool_env(None)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    closed_port = _free_port() # nothing is listening on it
    server = ServerPreset.router()
    server.models_dir = str(models_dir)
    server.pool_server = f"http://127.0.0.1:{closed_port}"
    server.pool_name = ROUTER_NAME
    server.log_path = str(tmp_path / "server.log")
    server.start()
    try:
        _load_model(server, MODEL_ID)
        res = server.make_request("GET", "/v1/models")
        assert res.status_code == 200
        assert MODEL_ID in [m["id"] for m in res.body.get("data", [])]
        # the pool thread keeps retrying in the background
        _wait_log(server, "pool: subscribe failed: cannot reach", timeout=30)
    finally:
        server.stop()


def test_pool_work_offer_via_long_poll(tmp_path, coordinator):
    server = _start_router(tmp_path, coordinator, link_mode="poll")
    try:
        env = coordinator.wait_for("handshake", lambda e: True)
        assert env["payload"]["router_name"] == ROUTER_NAME
        coordinator.send_work_offer("job-poll-1", tag="link-test")
        _wait_log(server, "pool: offer job_id=job-poll-1 tag=link-test")
    finally:
        server.stop()
