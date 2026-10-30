"""WS-05: the fork dispatch core.

The fake coordinator implements the full protocol surface the router needs:
job intake (POST /pool/jobs), body staging (PUT/GET .../body), the held-open
response relay (GET .../response fed by PUT .../response), the subscribe
channel (SSE work offers + inbound events), and the class advertisement.

Scenarios:
  A local-first (no coordinator round trip)
  B originating + serving on the same router (the full round trip)
  C offer reject with reason no_slot
  D offer reject with reason model_gone
  E anti-memory: the prompt never reaches the router log
"""

import itertools
import json
import os
import queue
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from utils import *

MODEL_HF_REPO = "ggml-org/tinygemma3-GGUF"
MODEL_HF_FILE = "tinygemma3-Q8_0.gguf"
LOCAL_MODEL_ID = "pool-md-member"
POOL_TEXT_MD = "pool-text-md"
POOL_TEXT_LG = "pool-text-lg"
LOCAL_N_CTX = 1024
POOL_TOKEN = "test-member-token"
ROUTER_NAME = "test-router"
# a recognizable substring of the B/E prompt that must never reach the router log
PROMPT_MARKER = "ZEBRA-MARKER-XYZ-98765"

CLASS_ADVERTISEMENT = {
    "protocol": 1,
    "classes": [
        {"tag": POOL_TEXT_MD, "member_count": 2, "advertised_ctx": 1024, "min_ctx": 1024, "max_ctx": 1024},
        {"tag": POOL_TEXT_LG, "member_count": 1, "advertised_ctx": 2048, "min_ctx": 2048, "max_ctx": 2048},
    ],
}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _Job:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.tag = ""
        self.metadata = {}
        self.body: bytes = b""
        # the held-open GET reads relayed lines from here; None closes the stream
        self.resp_queue: "queue.Queue" = queue.Queue()
        self.pushed_lines: list[str] = []
        self.origin_router = ""


class FakeCoordinator:
    def __init__(self, classes: dict | None = None):
        self.port = _free_port()
        self.classes = classes if classes is not None else CLASS_ADVERTISEMENT
        self.lock = threading.Lock()
        self.jobs: dict[str, _Job] = {}
        self.intakes: list[dict] = []
        self.received: list[dict] = []
        self.last_auth = ""
        self.sse_subs: list["queue.Queue"] = []
        self.pending_offers: list[dict] = []
        self._next_job = itertools.count(1)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), self._make_handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def received_of(self, etype: str) -> list[dict]:
        with self.lock:
            return [e for e in self.received if e.get("type") == etype]

    def intake_count(self) -> int:
        with self.lock:
            return len(self.intakes)

    def send_work_offer(self, job_id: str, tag: str, est_ctx: int = 0, est_prompt: int = 0,
                        conv_key: str | None = None) -> None:
        env = {
            "type": "work_offer",
            "payload": {
                "job_id": job_id,
                "tag": tag,
                "est_prompt_tokens": est_prompt,
                "est_ctx_needed": est_ctx,
                "priority": "live",
                "source": "queue",
                "conv_key": conv_key,
            },
        }
        with self.lock:
            subs = list(self.sse_subs)
            if not subs:
                self.pending_offers.append(env)
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
                if self.path == "/pool/jobs":
                    return outer._intake(self)
                if self.path != "/pool/events":
                    return self._send_json({"error": "not found"}, 404)
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                try:
                    env = json.loads(raw)
                except json.JSONDecodeError:
                    return self._send_json({"error": "bad json"}, 400)
                with outer.lock:
                    outer.received.append(env)
                    outer.last_auth = self.headers.get("Authorization", "")
                self._send_json({"ok": True})

            def do_PUT(self):
                m = re.match(r"^/pool/jobs/([^/]+)/body$", self.path)
                if m:
                    return outer._put_body(self, m.group(1))
                m = re.match(r"^/pool/jobs/([^/]+)/response$", self.path)
                if m:
                    return outer._put_response(self, m.group(1))
                self._send_json({"error": "not found"}, 404)

            def do_GET(self):
                if self.path == "/pool/classes":
                    return self._send_json(outer.classes)
                if self.path == "/pool/subscribe":
                    return outer._subscribe(self)
                m = re.match(r"^/pool/jobs/([^/]+)/body$", self.path)
                if m:
                    return outer._get_body(self, m.group(1))
                m = re.match(r"^/pool/jobs/([^/]+)/response$", self.path)
                if m:
                    return outer._get_response(self, m.group(1))
                self._send_json({"error": "not found"}, 404)

        return Handler

    # -- coordinator endpoints -------------------------------------------

    def _intake(self, h: "Handler") -> None:
        length = int(h.headers.get("Content-Length", 0))
        meta = json.loads(h.rfile.read(length) or b"{}")
        job_id = f"job-{next(self._next_job)}"
        job = _Job(job_id)
        job.tag = meta.get("tag", "")
        job.metadata = meta
        job.origin_router = h.headers.get("X-Pool-Router-Id", "")
        with self.lock:
            self.jobs[job_id] = job
            self.intakes.append({"job_id": job_id, "metadata": meta, "origin": job.origin_router})
        h.send_response(201)
        h.send_header("Content-Type", "application/json")
        body = json.dumps({"job_id": job_id}).encode()
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)

    def _put_body(self, h: "Handler", job_id: str) -> None:
        length = int(h.headers.get("Content-Length", 0))
        data = h.rfile.read(length)
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                h._send_json({"error": "unknown job"}, 404)
                return
            job.body = data
        h._send_json({"status": "queued"})

    def _get_body(self, h: "Handler", job_id: str) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                h._send_json({"error": "unknown job"}, 404)
                return
            body = job.body
        h.send_response(200)
        h.send_header("Content-Type", "application/octet-stream")
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)

    def _put_response(self, h: "Handler", job_id: str) -> None:
        length = int(h.headers.get("Content-Length", 0))
        data = h.rfile.read(length).decode("utf-8", "replace")
        lines = [l for l in data.split("\n") if l.strip()]
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                h._send_json({"error": "unknown job"}, 404)
                return
            job.pushed_lines = lines
        # relay the lines to the held-open GET, in order, then close it
        for line in lines:
            job.resp_queue.put(line)
        job.resp_queue.put(None)
        h._send_json({"ok": True})

    def _get_response(self, h: "Handler", job_id: str) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            h._send_json({"error": "unknown job"}, 404)
            return
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Connection", "close")
        h.end_headers()
        while True:
            try:
                line = job.resp_queue.get(timeout=5)
            except queue.Empty:
                h.wfile.write(f": hb {int(time.time() * 1000)}\n\n".encode())
                h.wfile.flush()
                continue
            if line is None:
                break
            h.wfile.write(f"data: {line}\n\n".encode())
            h.wfile.flush()
        try:
            h.wfile.close()
        except OSError:
            pass

    def _subscribe(self, h: "Handler") -> None:
        if "text/event-stream" in h.headers.get("Accept", ""):
            h.send_response(200)
            h.send_header("Content-Type", "text/event-stream")
            h.send_header("Cache-Control", "no-cache")
            h.send_header("Connection", "close")
            h.end_headers()
            q: "queue.Queue" = queue.Queue()
            with self.lock:
                for env in list(self.pending_offers):
                    q.put(env)
                self.pending_offers = []
                self.sse_subs.append(q)
            try:
                while True:
                    try:
                        env = q.get(timeout=5)
                    except queue.Empty:
                        h.wfile.write(f": hb {int(time.time() * 1000)}\n\n".encode())
                        h.wfile.flush()
                        continue
                    h.wfile.write(f"data: {json.dumps(env)}\n\n".encode())
                    h.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with self.lock:
                    if q in self.sse_subs:
                        self.sse_subs.remove(q)
        else:
            deadline = time.time() + 5.0
            got: list[dict] = []
            while time.time() < deadline and not got:
                with self.lock:
                    if self.pending_offers:
                        got = list(self.pending_offers)
                        self.pending_offers = []
                if not got:
                    time.sleep(0.05)
            h._send_json(got)


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


def _start_router(tmp_path, pool_server: str, preset: str) -> ServerProcess:
    os.environ["LLAMA_POOL_TOKEN"] = POOL_TOKEN
    os.environ.pop("LLAMA_POOL_LINK_MODE", None)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    server = ServerPreset.router()
    server.models_dir = str(models_dir)
    server.models_preset = preset
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


def _wait_until(cond, timeout: float = 60.0, what: str = "condition") -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def _consume_stream(resp, state: dict) -> None:
    state["lines"] = []
    for line_bytes in resp.iter_lines():
        line = line_bytes.decode("utf-8", "replace")
        if line:
            state["lines"].append(line)
    state["done"] = True


# start a generation on the model that keeps its (single) slot busy
def _occupy_slot(server: ServerProcess, model_id: str):
    req = requests.post(
        server.make_url("/v1/chat/completions"),
        json={
            "model": model_id,
            "stream": True,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper"}],
        },
        stream=True,
        timeout=180,
    )
    state: dict = {}
    t = threading.Thread(target=_consume_stream, args=(req, state), daemon=True)
    t.start()
    _wait_until(lambda: state.get("lines"), what="the slot-occupying generation to be in flight")
    return t, state


def _wait_intake(coordinator: FakeCoordinator, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with coordinator.lock:
            if coordinator.intakes:
                return coordinator.intakes[0]
        time.sleep(0.1)
    raise AssertionError("timed out waiting for the intake on the coordinator")


def test_dispatch_local_first_no_intake(tmp_path, coordinator):
    """A: a request naming the class with a free local member is served locally
    without any coordinator round trip (D-D1)"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, POOL_TEXT_MD)

        res = server.make_request("POST", "/v1/chat/completions", data={
            "model": POOL_TEXT_MD,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert res.status_code == 200
        assert res.body.get("choices")
        content = "".join(c.get("message", {}).get("content") or "" for c in res.body["choices"])
        assert len(content) > 0
        # served locally: nothing reached the coordinator
        assert coordinator.intake_count() == 0
    finally:
        server.stop()


def test_dispatch_offer_reject_no_slot(tmp_path, coordinator):
    """C: an offer for a tag the router has but with no free slot is rejected
    with the precise reason no_slot"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, POOL_TEXT_MD)
        coordinator.wait_for("handshake", lambda e: True)
        coordinator.wait_for_sse()

        t_long, _ = _occupy_slot(server, LOCAL_MODEL_ID)
        try:
            coordinator.send_work_offer("job-c-1", tag=POOL_TEXT_MD, est_ctx=256)
            reject = coordinator.wait_for("offer_reject", lambda e: e["payload"]["job_id"] == "job-c-1")
            assert reject["payload"]["reason"] == "no_slot"
            # no accept, no body pull, no completion for this job
            assert coordinator.received_of("offer_accept") == []
        finally:
            t_long.join(timeout=180)
    finally:
        server.stop()


def test_dispatch_offer_reject_model_gone(tmp_path, coordinator):
    """D: an offer for a tag the router has no local model for is rejected with
    the precise reason model_gone"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, POOL_TEXT_MD)
        coordinator.wait_for("handshake", lambda e: True)
        coordinator.wait_for_sse()

        # pool-text-lg is advertised by the coordinator but has no local member here
        coordinator.send_work_offer("job-d-1", tag=POOL_TEXT_LG, est_ctx=256)
        reject = coordinator.wait_for("offer_reject", lambda e: e["payload"]["job_id"] == "job-d-1")
        assert reject["payload"]["reason"] == "model_gone"
        assert coordinator.received_of("offer_accept") == []
    finally:
        server.stop()


def test_dispatch_origin_serve_roundtrip(tmp_path, coordinator):
    """B + E: the local member is busy, so the request intakes on the coordinator,
    which offers the job back to the same router once the slot frees; the router
    accepts, pulls the body, serves, pushes the response, and the client gets the
    full completion. the prompt never reaches the router log (anti-memory)."""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, POOL_TEXT_MD)
        router_id = coordinator.wait_for("handshake", lambda e: True)["payload"]["router_id"]
        coordinator.wait_for_sse()

        # occupy the (single) slot so the pool request cannot be served locally
        t_long, _ = _occupy_slot(server, LOCAL_MODEL_ID)
        try:
            # the pool class request: it intakes because the local member is busy
            client: dict = {}

            def _client() -> None:
                r = requests.post(
                    server.make_url("/v1/chat/completions"),
                    json={
                        "model": POOL_TEXT_MD,
                        "stream": True,
                        "max_tokens": 32,
                        "messages": [{"role": "user", "content": PROMPT_MARKER + " hello there"}],
                    },
                    stream=True,
                    timeout=180,
                )
                client["status"] = r.status_code
                client["lines"] = []
                if r.status_code == 200:
                    for line_bytes in r.iter_lines():
                        line = line_bytes.decode("utf-8", "replace")
                        if line:
                            client["lines"].append(line)

            t_client = threading.Thread(target=_client, daemon=True)
            t_client.start()

            # the originating side intaked the job
            intake = _wait_intake(coordinator)
            job_id = intake["job_id"]
            assert intake["metadata"]["tag"] == POOL_TEXT_MD
            assert intake["origin"] == router_id
            assert isinstance(intake["metadata"].get("conv_key"), str)
            assert len(intake["metadata"]["conv_key"]) == 64

            # wait for the slot-occupying generation to end so the slot frees
            t_long.join(timeout=180)

            # now offer the job back to the same router (uniform path, D-D1)
            coordinator.send_work_offer(job_id, tag=POOL_TEXT_MD, est_ctx=256, est_prompt=32,
                                        conv_key=intake["metadata"]["conv_key"])

            # the serving side accepts, serves, and pushes; the client gets the stream
            t_client.join(timeout=180)
            assert client["status"] == 200, (
                f"client status {client.get('status')}, lines {client.get('lines')[:5]}"
            )
            # the client received the child's NDJSON stream, verbatim framing
            assert any(l.startswith("data: ") for l in client["lines"]), client["lines"][:5]

            # the coordinator received a completion event with sane counts
            comp = coordinator.wait_for("completion", lambda e: e["payload"]["job_id"] == job_id)
            p = comp["payload"]
            assert p["ok"] is True
            assert p["served_by"] == router_id
            assert p["model_id"] == LOCAL_MODEL_ID
            assert p["response_tokens"] >= 1
            assert p["prompt_tokens"] >= 1
            assert p["conv_key"] == intake["metadata"]["conv_key"]

            # the coordinator actually staged and served the raw body
            with coordinator.lock:
                job = coordinator.jobs[job_id]
            body = json.loads(job.body)
            assert body["model"] == POOL_TEXT_MD
            assert PROMPT_MARKER in json.dumps(body["messages"])

            # E: anti-memory. the prompt content never reaches the router log;
            # ids and counts may, the marker must not
            with open(server.log_path, "r", errors="replace") as f:
                log = f.read()
            assert PROMPT_MARKER not in log, "the prompt marker leaked into the router log"
            # the job id and model id may appear (metadata)
            assert job_id in log or LOCAL_MODEL_ID in log
        finally:
            t_long.join(timeout=180)
    finally:
        server.stop()
