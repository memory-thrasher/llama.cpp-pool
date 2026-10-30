"""WS-04: pool class resolution + advertisement merge.

The fake coordinator serves GET /pool/classes with the protocol contract
example (the same values as pool-coordinator/protocol/examples/
class_advertisement.json). The router loads a local model tagged pool-text-md.
"""

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from utils import *

MODEL_HF_REPO = "ggml-org/tinygemma3-GGUF"
MODEL_HF_FILE = "tinygemma3-Q8_0.gguf"
# the preset section name of the local tagged model
LOCAL_MODEL_ID = "pool-md-member"
POOL_TEXT_MD = "pool-text-md"
POOL_TEXT_LG = "pool-text-lg"
POOL_TEXT_XS = "pool-text-xs"
LOCAL_N_CTX = 1024
POOL_TOKEN = "test-member-token"
ROUTER_NAME = "test-router"

# the contract example of protocol/schemas/class_advertisement.json
CLASS_ADVERTISEMENT = {
    "protocol": 1,
    "classes": [
        {
            "tag": POOL_TEXT_MD,
            "member_count": 2,
            "advertised_ctx": 131072,
            "min_ctx": 32768,
            "max_ctx": 193280,
        },
        {
            "tag": POOL_TEXT_LG,
            "member_count": 1,
            "advertised_ctx": 208128,
            "min_ctx": 208128,
            "max_ctx": 208128,
        },
    ],
}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeCoordinator:
    """A minimal pool coordinator: POST /pool/events is recorded,
    GET /pool/classes serves the advertisement state, GET /pool/subscribe
    serves a long-poll JSON array (no offers needed for this test)."""

    def __init__(self, classes: dict | None = None):
        self.port = _free_port()
        self.classes = classes if classes is not None else CLASS_ADVERTISEMENT
        self.lock = threading.Lock()
        self.received: list[dict] = []
        self.last_auth = ""
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), self._make_handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def set_classes(self, obj: dict) -> None:
        with self.lock:
            self.classes = obj

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
                    with outer.lock:
                        self._send_json(outer.classes)
                    return
                if self.path != "/pool/subscribe":
                    self._send_json({"error": "not found"}, 404)
                    return
                # long-poll: hold the response briefly and close with an empty list
                time.sleep(1.0)
                self._send_json([])

        return Handler


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


def _write_preset(path: str, model_id: str, tag: str | None) -> None:
    lines = [
        f"[{model_id}]",
        f"hf       = {MODEL_HF_REPO}",
        f"hf-file  = {MODEL_HF_FILE}",
        f"ctx-size = {LOCAL_N_CTX}",
    ]
    if tag is not None:
        lines.append(f"tags     = {tag}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _start_router(tmp_path, pool_server: str | None, preset: str | None = None) -> ServerProcess:
    if pool_server is not None:
        os.environ["LLAMA_POOL_TOKEN"] = POOL_TOKEN
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    server = ServerPreset.router()
    server.models_dir = str(models_dir)
    server.models_preset = preset
    if pool_server is not None:
        server.pool_server = pool_server
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
        if item.get("id") == model_id:
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


def _model_entries(server: ServerProcess) -> dict:
    res = server.make_request("GET", "/v1/models")
    assert res.status_code == 200
    return {item["id"]: item for item in res.body.get("data", [])}


def _wait_for_entry(server: ServerProcess, model_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        entries = _model_entries(server)
        last = entries.get(model_id)
        if last is not None:
            return last
        time.sleep(0.5)
    raise AssertionError(f"timed out waiting for {model_id!r} in /v1/models, entries: {list(last or {})}")


def test_pool_classes_advertised(tmp_path, coordinator):
    """both the coordinator's classes and the local one are listed, the
    coordinator numbers win, status is loaded (D-C1, D-C2, D-C8)"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)

        md = _wait_for_entry(server, POOL_TEXT_MD)
        assert md["status"]["value"] == "loaded"
        # the coordinator value wins over the local one, no double count:
        # 2 members total, not local 1 + coordinator 2
        assert md["context_length"] == 131072
        assert md["pool"]["member_count"] == 2
        assert md["pool"]["min_ctx"] == 32768
        assert md["pool"]["max_ctx"] == 193280
        assert POOL_TEXT_MD in md["tags"]

        # advertised even though this router has no local model under it (D-C1)
        lg = _wait_for_entry(server, POOL_TEXT_LG)
        assert lg["status"]["value"] == "loaded"
        assert lg["context_length"] == 208128
        assert lg["pool"]["member_count"] == 1

        # the local model is still listed as a real model, unchanged shape
        local = _model_entries(server)[LOCAL_MODEL_ID]
        assert local["status"]["value"] == "loaded"
        assert POOL_TEXT_MD in local["tags"]
        assert "pool" not in local

        # /props?model=<tag>: synthetic props with the advertised context, no 404
        props_res = server.make_request("GET", f"/props?model={POOL_TEXT_MD}")
        assert props_res.status_code == 200
        assert props_res.body["role"] == "router"
        assert props_res.body["model_alias"] == POOL_TEXT_MD
        assert props_res.body["n_ctx"] == 131072
        assert props_res.body["default_generation_settings"]["n_ctx"] == 131072
    finally:
        server.stop()


def test_pool_class_served_locally(tmp_path, coordinator):
    """a request naming the class is served by the local member with a free
    slot, local-first (D-D1)"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, POOL_TEXT_MD)

        res = server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": POOL_TEXT_MD,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert res.status_code == 200
        assert res.body.get("choices")
        content = "".join(
            c.get("message", {}).get("content") or "" for c in res.body["choices"]
        )
        assert len(content) > 0
    finally:
        server.stop()


def test_pool_class_no_membership(tmp_path, coordinator):
    """a class with no member anywhere gets the canonical no-membership error (D-C7)"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, POOL_TEXT_MD)

        # the fake coordinator omits pool-text-xs, no local model carries it
        assert POOL_TEXT_XS not in _model_entries(server)
        res = server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": POOL_TEXT_XS,
                "max_tokens": 4,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert res.status_code == 400
        assert "error" in res.body
        assert res.body["error"]["message"] == "model is not loaded"
    finally:
        server.stop()


def test_pool_class_local_only_fallback(tmp_path):
    """coordinator down: the class of the local tagged model is still
    advertised, with the local numbers (D-C8 degradation)"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    closed_port = _free_port()  # nothing is listening on it
    server = _start_router(tmp_path, f"http://127.0.0.1:{closed_port}", preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)

        md = _wait_for_entry(server, POOL_TEXT_MD)
        assert md["status"]["value"] == "loaded"
        assert md["context_length"] == LOCAL_N_CTX
        assert md["pool"]["member_count"] == 1
        assert md["pool"]["min_ctx"] == LOCAL_N_CTX
        assert md["pool"]["max_ctx"] == LOCAL_N_CTX

        # no cross-member classes without the coordinator
        assert POOL_TEXT_LG not in _model_entries(server)
    finally:
        server.stop()


def test_pool_off_no_tags_stock_listing(tmp_path):
    """no --pool-server and no pool tags: the model list is unchanged from stock"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, "plain-member", None)
    os.environ.pop("LLAMA_POOL_TOKEN", None)
    server = _start_router(tmp_path, None, preset)
    try:
        _load_model(server, "plain-member")

        entries = _model_entries(server)
        assert "plain-member" in entries
        entry = entries["plain-member"]
        assert entry["status"]["value"] == "loaded"
        assert "pool" not in entry
        assert "context_length" not in entry or isinstance(entry["context_length"], int)

        # no synthetic pool class entries at all
        assert not any(i.startswith("pool-") for i in entries)
        for item in entries.values():
            assert "pool" not in item
    finally:
        server.stop()
