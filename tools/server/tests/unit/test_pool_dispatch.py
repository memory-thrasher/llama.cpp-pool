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

import hashlib
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


# the lines a fake serving router pushes for an auto-served batch job: the
# child's stream lines verbatim (data-framed, ending in data: [DONE])
CANNED_BATCH_PUSH = [
    'data: {"id":"chatcmpl-batch","object":"chat.completion.chunk","created":1,"model":"tiny","choices":[{"index":0,"delta":{"role":"assistant","content":"WS10-"},"finish_reason":null}]}',
    'data: {"id":"chatcmpl-batch","object":"chat.completion.chunk","created":1,"model":"tiny","choices":[{"index":0,"delta":{"content":"BATCH"},"finish_reason":null}]}',
    'data: {"id":"chatcmpl-batch","object":"chat.completion.chunk","created":1,"model":"tiny","choices":[{"index":0,"delta":{"content":" OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":4}}',
    'data: [DONE]',
]


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
        # job_id -> {"message": str, "keep_lines": int}; the response relay is
        # cut mid-stream (the serving member died) and the terminal error is
        # mirrored, exactly as the coordinator's running cap does (D-G2)
        self.mid_stream_death: dict[str, dict] = {}
        # WS-10 batch: intake dedupe (the D-G5 request_id) + auto-serve. when
        # auto_serve is on, the pushed lines of a staged job are HELD until
        # release_batch_jobs(), so a test can observe the pending status first
        self.request_ids: dict[str, str] = {}
        self.auto_serve = False
        self.held_lines: dict[str, list[str]] = {}
        # set once the test releases; a staged job whose PUT lands after the
        # release must not be held (lost-wakeup), so it is served directly
        self.release_event = threading.Event()
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

    def arm_mid_stream_death(self, job_id: str, message: str = "service timeout after 2s (D-G2)",
                             keep_lines: int = 4) -> None:
        """the relay is cut after keep_lines, then the terminal error envelope is
        mirrored (event first, relay close after - the coordinator's order)"""
        with self.lock:
            self.mid_stream_death[job_id] = {"message": message, "keep_lines": keep_lines}

    def send_terminal(self, job_id: str, ok: bool, message: str = "") -> None:
        """mirror a terminal envelope (completion / error) onto the subscribe stream"""
        payload: dict = {"job_id": job_id, "ok": ok}
        if ok:
            payload.update({"model_id": "", "served_by": ""})
        else:
            payload.update({"stage": "mid_stream", "code": "other", "message": message})
        env = {"type": "completion" if ok else "error", "payload": payload}
        with self.lock:
            subs = list(self.sse_subs)
        for q in subs:
            q.put(env)

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

    def _serve_job(self, job_id: str, lines: list[str]) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            return
        for line in lines:
            job.resp_queue.put(line)
        job.resp_queue.put(None)

    def release_batch_jobs(self) -> None:
        """the fake serving side serves the held batch jobs: push the canned
        stream lines (as the serving router would) so the held-open relay
        closes"""
        with self.lock:
            self.release_event.set()
            held = dict(self.held_lines)
            self.held_lines = {}
        for job_id, lines in held.items():
            self._serve_job(job_id, lines)

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
        # the D-G5 idempotency: the same (user, request_id) within the window
        # returns the existing job (200); the window is not modeled here
        request_id = meta.get("request_id", "")
        with self.lock:
            existing = self.request_ids.get(request_id) if request_id else None
        if existing is not None:
            with self.lock:
                job_id = existing
                self.intakes.append({"job_id": job_id, "metadata": meta, "origin": h.headers.get("X-Pool-Router-Id", "")})
            h.send_response(200)
            h.send_header("Content-Type", "application/json")
            body = json.dumps({"job_id": job_id}).encode()
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
            return
        job_id = f"job-{next(self._next_job)}"
        job = _Job(job_id)
        job.tag = meta.get("tag", "")
        job.metadata = meta
        job.origin_router = h.headers.get("X-Pool-Router-Id", "")
        with self.lock:
            self.jobs[job_id] = job
            if request_id:
                self.request_ids[request_id] = job_id
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
        serve_now = False
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                h._send_json({"error": "unknown job"}, 404)
                return
            job.body = data
            if self.auto_serve:
                if self.release_event.is_set():
                    # the test released before this PUT landed: serve directly,
                    # never hold (a held job would wait for a release that
                    # already went by)
                    serve_now = True
                else:
                    # the fake serving side will serve this job: hold the canned
                    # lines until the test releases them
                    self.held_lines[job_id] = list(CANNED_BATCH_PUSH)
        if serve_now:
            self._serve_job(job_id, list(CANNED_BATCH_PUSH))
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
            death = self.mid_stream_death.pop(job_id, None)
        if death is not None:
            # the serving member died mid-push: the coordinator's running cap
            # mirrors the terminal error and cuts the relay (no child [DONE])
            self.send_terminal(job_id, ok=False, message=death["message"])
            kept = [l for l in lines if l.strip() != "data: [DONE]"][: death["keep_lines"]]
            for line in kept:
                job.resp_queue.put(line)
            job.resp_queue.put(None)
            h._send_json({"ok": True})
            return
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
    saved = {
        k: os.environ.get(k)
        for k in ("LLAMA_POOL_TOKEN", "LLAMA_POOL_LINK_MODE", "POOL_BATCH_DIR", "POOL_BATCH_RESULT_TTL_MS")
    }
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


#
# WS-09 Part A: the D-J5 error taxonomy (hold / give-up / retry / interrupted)
#


def test_dispatch_hold_while_member_busy(tmp_path, coordinator):
    """A.1/D-J5 HOLD: a request for a class whose only member is busy holds the
    client - 200 + open socket, ZERO content bytes until the member serves,
    then the stream (pi waits; its own timeout governs)"""
    import http.client
    import select

    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, POOL_TEXT_MD)
        coordinator.wait_for("handshake", lambda e: True)
        coordinator.wait_for_sse()

        # the only member's (single) slot is busy
        t_long, _ = _occupy_slot(server, LOCAL_MODEL_ID)
        try:
            body = json.dumps({
                "model": POOL_TEXT_MD,
                "stream": True,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hold test"}],
            })
            conn = http.client.HTTPConnection(
                server.server_host.split(",")[0].strip(), server.server_port, timeout=60)
            conn.request("POST", "/v1/chat/completions", body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            assert resp.status == 200
            assert resp.getheader("Content-Type", "").startswith("text/event-stream")

            # the job is queued (the member is still busy): zero content bytes
            readable, _, _ = select.select([resp.fp], [], [], 2.0)
            assert not readable, "content bytes arrived while the member was still busy"

            intake = _wait_intake(coordinator)
            job_id = intake["job_id"]

            # free the member and serve: now the stream arrives
            t_long.join(timeout=180)
            coordinator.send_work_offer(job_id, tag=POOL_TEXT_MD, est_ctx=256, est_prompt=16,
                                        conv_key=intake["metadata"].get("conv_key"))
            chunks = []
            while True:
                c = resp.read(4096)
                if not c:
                    break
                chunks.append(c)
            data = b"".join(chunks)
            assert b"data: " in data, data[:200]
            assert b"[DONE]" in data, data[-200:]
        finally:
            t_long.join(timeout=180)
    finally:
        server.stop()


def test_dispatch_give_up_no_members(tmp_path):
    """A.2/D-J5 GIVE-UP: a class with no members at all (nowhere, not local)
    -> neutral 4xx with the canonical wording, and no busy wording that stock
    pi would retry on"""
    # pool-text-lg has no membership anywhere: not listed by the coordinator,
    # no local model carries that tag
    ad = {"protocol": 1, "classes": [
        {"tag": POOL_TEXT_MD, "member_count": 1, "advertised_ctx": 1024,
         "min_ctx": 1024, "max_ctx": 1024},
    ]}
    coordinator = FakeCoordinator(classes=ad)
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, POOL_TEXT_MD)

        # pool-text-lg has no member at all: not local (no model with that
        # tag), not on the coordinator (the advertisement lists no such class
        # with membership)
        res = server.make_request("POST", "/v1/chat/completions", data={
            "model": POOL_TEXT_LG,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "give up test"}],
        })
        assert res.status_code == 400, res.status_code
        err = res.body.get("error")
        assert err is not None, res.body
        assert err.get("message") == "model is not loaded", err
        # neutral wording: nothing stock pi would auto-retry on
        s = json.dumps(res.body).lower()
        for word in ("overloaded", "server_busy", "at capacity", "rate limit",
                     "unavailable", "server_error", "retry"):
            assert word not in s, (word, s)
    finally:
        server.stop()
        coordinator.stop()


def test_dispatch_retry_coordinator_down(tmp_path):
    """A.3/D-J5 RETRY: the coordinator is unreachable (dead port) -> 5xx with
    busy wording in the body, so stock pi's retry pattern matches"""
    coordinator = FakeCoordinator()
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, POOL_TEXT_MD)
        coordinator.wait_for("handshake", lambda e: True)

        # the coordinator goes away; the advertisements are already cached
        coordinator.stop()

        # the only member's slot is busy, so the request must intake - on the
        # dead port
        t_long, _ = _occupy_slot(server, LOCAL_MODEL_ID)
        try:
            res = server.make_request("POST", "/v1/chat/completions", data={
                "model": POOL_TEXT_MD,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "retry test"}],
            })
            assert res.status_code in (502, 503), res.status_code
            s = json.dumps(res.body)
            assert "server_busy" in s or "overloaded" in s, s
        finally:
            t_long.join(timeout=180)
    finally:
        server.stop()


def test_dispatch_mid_stream_death(tmp_path, coordinator):
    """A.4/D-G2: the stream started and then the serving member died (the
    coordinator's running cap cut the relay and mirrored the terminal error).
    The client sees a clean, structured end-of-stream error - not a raw hang,
    not a half-open socket"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, POOL_TEXT_MD)
        coordinator.wait_for("handshake", lambda e: True)
        coordinator.wait_for_sse()

        # occupy the slot so the request intakes (uniform path)
        t_long, _ = _occupy_slot(server, LOCAL_MODEL_ID)
        try:
            client: dict = {}

            def _client() -> None:
                r = requests.post(
                    server.make_url("/v1/chat/completions"),
                    json={
                        "model": POOL_TEXT_MD,
                        "stream": True,
                        "max_tokens": 32,
                        "messages": [{"role": "user", "content": "mid stream death"}],
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

            intake = _wait_intake(coordinator)
            job_id = intake["job_id"]

            # arm the coordinator-side simulation: cut the relay mid-stream and
            # mirror the terminal error (the running cap's behavior, D-G2)
            coordinator.arm_mid_stream_death(job_id, message="service timeout after 2s (D-G2)")

            t_long.join(timeout=180)
            coordinator.send_work_offer(job_id, tag=POOL_TEXT_MD, est_ctx=256, est_prompt=32,
                                        conv_key=intake["metadata"].get("conv_key"))

            # the client stream must END (no hang, no half-open socket)
            t_client.join(timeout=180)
            assert not t_client.is_alive(), "the client stream hung (half-open socket)"
            assert client["status"] == 200, client.get("status")
            lines = client["lines"]
            assert lines, "no lines at all"

            # the stream started: content chunks before the terminal
            body_lines = lines[:-3]
            assert any('"delta"' in l for l in body_lines), lines[:5]

            # the structured terminal: a network_error finish chunk, the
            # coordinator's reason in an error object, and a final [DONE]
            assert lines[-1] == "data: [DONE]", lines[-3:]
            tail = "\n".join(lines[-3:])
            assert "network_error" in tail, tail
            assert "server_error" in tail, tail
            assert "service timeout after 2s (D-G2)" in tail, tail
        finally:
            t_long.join(timeout=180)
    finally:
        server.stop()


#
# WS-12: exclusion marks (D-F1) + intake idempotency (D-G5)
#


def _expected_request_id(raw_body: bytes) -> str:
    """the D-G5 request id: sha-256 of the full body, first 16 bytes,
    8-4-4-4-12 lowercase. the reference the fork must match"""
    h = hashlib.sha256(raw_body).hexdigest()
    return h[0:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:32]


def _intake_request_id(server: ServerProcess, coordinator: "FakeCoordinator", raw_body: bytes) -> str:
    """occupy the slot (so the local member cannot serve the pool request), POST
    the pool request with the exact raw body (it intakes on the coordinator), and
    return the request_id the fork stamped. the client is not awaited: the intake
    is all the test needs, the held-open response simply times out. each call
    occupies the slot fresh so the fake-timing occupation cannot run out mid-test"""
    t_long, _ = _occupy_slot(server, LOCAL_MODEL_ID)
    try:
        n = coordinator.intake_count()

        def _client() -> None:
            requests.post(
                server.make_url("/v1/chat/completions"),
                data=raw_body,
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=15,
            )

        t = threading.Thread(target=_client, daemon=True)
        t.start()
        deadline = time.time() + 60
        while time.time() < deadline:
            with coordinator.lock:
                if len(coordinator.intakes) > n:
                    return coordinator.intakes[-1]["metadata"].get("request_id", "")
            time.sleep(0.1)
        raise AssertionError("timed out waiting for the intake")
    finally:
        t_long.join(timeout=180)


def test_request_id_determinism(tmp_path, coordinator):
    """D-G5: the intake request_id is the sha-256 of the full raw body (first 16
    bytes, uuid-shaped). same body -> same id; different body -> different id"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, POOL_TEXT_MD)
        coordinator.wait_for("handshake", lambda e: True)
        coordinator.wait_for_sse()

        body_a = b'{"model":"pool-text-md","messages":[{"role":"user","content":"REQ-A-WS12"}]}'
        body_b = b'{"model":"pool-text-md","messages":[{"role":"user","content":"REQ-B-WS12"}]}'

        rid_a1 = _intake_request_id(server, coordinator, body_a)
        rid_a2 = _intake_request_id(server, coordinator, body_a)
        rid_b = _intake_request_id(server, coordinator, body_b)

        assert rid_a1, "no request_id stamped on the intake"
        # same body -> same id
        assert rid_a1 == rid_a2, f"same body gave different ids: {rid_a1} vs {rid_a2}"
        # different body -> different id
        assert rid_a1 != rid_b, "different bodies gave the same id"
        # the id is the sha-256 of the body (first 16 bytes, uuid-shaped)
        assert rid_a1 == _expected_request_id(body_a), rid_a1
        assert rid_b == _expected_request_id(body_b), rid_b
    finally:
        server.stop()


def test_exclusion_mark_flow(tmp_path, coordinator):
    """D-F1: /models/unload on a loaded pool member is an exclusion mark, not a
    real unload (the coordinator gets exclusion_update(excluded=true), the model
    stays loaded, /models reports "excluded"). /models/load on the excluded
    member unmarks it (exclusion_update(excluded=false), no reload, status back
    to loaded)"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, LOCAL_MODEL_ID)
        _wait_for_entry(server, LOCAL_MODEL_ID)
        router_id = coordinator.wait_for("handshake", lambda e: True)["payload"]["router_id"]

        # unload the pool member: it is marked excluded, not really unloaded
        res = server.make_request("POST", "/models/unload", data={"model": LOCAL_MODEL_ID})
        assert res.status_code == 200, res.body
        assert res.body.get("success") is True
        excl_on = coordinator.wait_for("exclusion_update", lambda e:
            e["payload"]["model_id"] == LOCAL_MODEL_ID and e["payload"]["excluded"] is True)
        assert excl_on["payload"]["router_id"] == router_id
        # the model stays loaded and now reports "excluded"
        _wait_model_status(server, LOCAL_MODEL_ID, "excluded")

        # load the excluded member: it is unmarked, not re-loaded
        res = server.make_request("POST", "/models/load", data={"model": LOCAL_MODEL_ID})
        assert res.status_code == 200, res.body
        assert res.body.get("success") is True
        excl_off = coordinator.wait_for("exclusion_update", lambda e:
            e["payload"]["model_id"] == LOCAL_MODEL_ID and e["payload"]["excluded"] is False)
        assert excl_off["payload"]["router_id"] == router_id
        # back to loaded (the model was never unloaded, so no real reload ran)
        _wait_model_status(server, LOCAL_MODEL_ID, "loaded")
    finally:
        server.stop()


def test_unload_plain_model_real_unload(tmp_path, coordinator):
    """D-F1: /models/unload on a real model WITHOUT a pool tag is a real unload
    (the existing router behavior, unchanged). the model goes away, no exclusion
    event is sent"""
    preset = str(tmp_path / "preset.ini")
    # a plain model: no pool- tag
    _write_preset(preset, "plain-md", None)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        _load_model(server, "plain-md")
        coordinator.wait_for("handshake", lambda e: True)

        res = server.make_request("POST", "/models/unload", data={"model": "plain-md"})
        assert res.status_code == 200, res.body
        assert res.body.get("success") is True
        # the model is really unloaded, and no exclusion event was sent
        _wait_model_status(server, "plain-md", "unloaded")
        assert coordinator.received_of("exclusion_update") == [], "no exclusion event for a plain model"
    finally:
        server.stop()


#
# WS-10: async batch (D-H1) - submit, detached capture, poll
#


BATCH_MARKER = "YAK-MARKER-ABC-12345"


def _batch_job(content: str, marker: str = "") -> dict:
    return {
        "model": POOL_TEXT_MD,
        "stream": True,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": (marker + content) if marker else content}],
    }


def _blob_batch_job(content: str) -> dict:
    # non-streaming: the child's response is one blob line, no [DONE]
    return {
        "model": POOL_TEXT_MD,
        "stream": False,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": content}],
    }


def _submit_batch(server: ServerProcess, class_name: str, jobs: list[dict]) -> tuple[int, dict]:
    res = server.make_request("POST", "/pool/batches", data={"class": class_name, "jobs": jobs})
    return res.status_code, res.body


def _poll_batch(server: ServerProcess, batch_id: str) -> tuple[int, dict]:
    res = server.make_request("GET", f"/pool/batches/{batch_id}")
    return res.status_code, res.body


def _jobs_of(body: dict) -> dict[str, dict]:
    return {j["job_id"]: j for j in body.get("jobs", [])}


def _wait_batch_done(server: ServerProcess, batch_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        status, body = _poll_batch(server, batch_id)
        assert status == 200, (status, body)
        last = body
        jobs = _jobs_of(body)
        if all(j["status"] != "pending" for j in jobs.values()):
            return body
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for batch {batch_id} to finish; last: {json.dumps(last)[:2000]}")


def test_batch_submit_poll_roundtrip(tmp_path, coordinator):
    """submit -> 202 with ids; the detached capture runs without any client
    socket; the poll goes pending -> done with the captured stream, ok=true.
    the intake is priority=batch on the class; the prompt never reaches the
    router log (anti-memory)"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    batch_dir = tmp_path / "batches"
    os.environ["POOL_BATCH_DIR"] = str(batch_dir)
    coordinator.auto_serve = True
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        coordinator.wait_for("handshake", lambda e: True)
        coordinator.wait_for_sse()

        status, body = _submit_batch(server, POOL_TEXT_MD, [
            _batch_job("first job", BATCH_MARKER),
            _batch_job("second job"),
        ])
        assert status == 202, (status, body)
        batch_id = body["batch_id"]
        assert len(batch_id) == 36
        job_ids = body["job_ids"]
        assert len(job_ids) == 2
        assert all(job_ids) and len(set(job_ids)) == 2  # the ids are the coordinator's (uuids in prod)

        # the batch record on disk (the fork's data dir)
        with open(batch_dir / (batch_id + ".json")) as f:
            record = json.load(f)
        assert record["class"] == POOL_TEXT_MD
        assert record["job_ids"] == job_ids
        assert "created_at" in record

        # the intakes: priority batch, the class tag, the deterministic id
        deadline = time.time() + 60
        while time.time() < deadline and coordinator.intake_count() < 2:
            time.sleep(0.1)
        assert coordinator.intake_count() >= 2, coordinator.intakes
        with coordinator.lock:
            meta_a = coordinator.jobs[job_ids[0]].metadata
            meta_b = coordinator.jobs[job_ids[1]].metadata
        assert meta_a["priority"] == "batch"
        assert meta_b["priority"] == "batch"
        assert meta_a["tag"] == POOL_TEXT_MD
        # the fork re-dumps the job object compactly (insertion order) before
        # the sha-256; mirror the exact bytes here
        expected_rid = _expected_request_id(json.dumps(
            _batch_job("first job", BATCH_MARKER), separators=(",", ":")).encode())
        assert meta_a["request_id"] == expected_rid, (meta_a["request_id"], expected_rid)

        # the capture is not released yet: the poll reports pending
        time.sleep(1.0)
        status, body = _poll_batch(server, batch_id)
        assert status == 200, (status, body)
        assert body["batch_id"] == batch_id
        assert body["class"] == POOL_TEXT_MD
        jobs = _jobs_of(body)
        assert all(j["status"] == "pending" for j in jobs.values()), jobs
        assert all("response" not in j for j in jobs.values()), jobs

        # the fake serving side serves the jobs; the capture closes
        coordinator.release_batch_jobs()
        body = _wait_batch_done(server, batch_id)
        jobs = _jobs_of(body)
        for jid in job_ids:
            j = jobs[jid]
            assert j["status"] == "done", j
            assert j["ok"] is True, j
            # the captured stream: the child's lines verbatim, [DONE] included.
            # the fake splits the marker across chunks; reassembling the deltas
            # must give the full text (the capture is line-for-line, not a re-parse)
            resp = j["response"]
            assert "data: [DONE]" in resp, resp
            content = "".join(
                json.loads(line[len("data: "):])["choices"][0]["delta"].get("content", "")
                for line in resp.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            )
            assert content == "WS10-BATCH OK", content
            # the capture files on disk
            ndjson = (batch_dir / (jid + ".ndjson")).read_text()
            assert ndjson == j["response"], "poll response differs from the capture file"
            meta = json.loads((batch_dir / (jid + ".meta.json")).read_text())
            assert meta["ok"] is True
            assert "done_at" in meta

        # anti-memory: the prompt content never reaches the router log
        with open(server.log_path, "r", errors="replace") as f:
            log = f.read()
        assert BATCH_MARKER not in log, "the batch prompt marker leaked into the router log"
        # job ids and counts may (metadata)
        assert batch_id in log or job_ids[0] in log
    finally:
        server.stop()


def test_batch_duplicate_intake(tmp_path, coordinator):
    """a job whose intake dedupes (D-G5, same body) is recorded as status
    "duplicate" in the batch metadata - no second capture"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    coordinator.auto_serve = True
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        coordinator.wait_for("handshake", lambda e: True)
        coordinator.wait_for_sse()

        job = _batch_job("duplicate me")

        status, body = _submit_batch(server, POOL_TEXT_MD, [job])
        assert status == 202, (status, body)
        first_id = body["job_ids"][0]
        coordinator.release_batch_jobs()
        body = _wait_batch_done(server, body["batch_id"])
        assert _jobs_of(body)[first_id]["status"] == "done"

        # the same body again: the intake dedupes onto the first job
        status, body = _submit_batch(server, POOL_TEXT_MD, [job])
        assert status == 202, (status, body)
        assert body["job_ids"] == [first_id], (body["job_ids"], first_id)
        status, body = _poll_batch(server, body["batch_id"])
        assert status == 200, (status, body)
        j = _jobs_of(body)[first_id]
        assert j["status"] == "duplicate", j
        assert j["ok"] is None
        assert "response" not in j, j
    finally:
        server.stop()


def test_batch_validation(tmp_path, coordinator):
    """class must exist, jobs must be a non-empty array of JSON objects,
    capped at 50"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        coordinator.wait_for("handshake", lambda e: True)

        # an unknown class
        status, body = _submit_batch(server, "pool-no-such-class", [_batch_job("x")])
        assert status == 400, (status, body)
        assert "class not found" in json.dumps(body), body

        # empty jobs
        status, body = _submit_batch(server, POOL_TEXT_MD, [])
        assert status == 400, (status, body)

        # a non-object job
        status, body = _submit_batch(server, POOL_TEXT_MD, ["not an object"])
        assert status == 400, (status, body)

        # over the cap
        status, body = _submit_batch(server, POOL_TEXT_MD, [_batch_job(f"j{i}") for i in range(51)])
        assert status == 400, (status, body)
        assert "max" in json.dumps(body), body
    finally:
        server.stop()


def test_batch_retention_ttl(tmp_path, coordinator):
    """POOL_BATCH_RESULT_TTL_MS: a batch older than the TTL is swept (the
    housekeeping tick, triggered by the next poll); the poll then 404s"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    batch_dir = tmp_path / "batches"
    os.environ["POOL_BATCH_DIR"] = str(batch_dir)
    # the sweep is throttled to once per second after the last sweep, so the
    # TTL must sit well above that; 2s leaves the detached capture (which
    # starts right after the submit) ample room to finish before expiry
    os.environ["POOL_BATCH_RESULT_TTL_MS"] = "2000"
    coordinator.auto_serve = True
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        coordinator.wait_for("handshake", lambda e: True)

        status, body = _submit_batch(server, POOL_TEXT_MD, [_batch_job("ttl job")])
        assert status == 202, (status, body)
        batch_id = body["batch_id"]
        job_id = body["job_ids"][0]
        coordinator.release_batch_jobs()
        body = _wait_batch_done(server, batch_id)
        assert all(j["status"] == "done" for j in _jobs_of(body).values())
        assert (batch_dir / (batch_id + ".json")).exists()

        # past the TTL: poll until the sweep has removed the batch
        deadline = time.time() + 30
        removed = False
        while time.time() < deadline:
            status, _ = _poll_batch(server, batch_id)
            if status == 404:
                removed = True
                break
            time.sleep(0.5)
        assert removed, "the batch was not swept after the TTL"
        assert not (batch_dir / (batch_id + ".json")).exists()
        assert not (batch_dir / (job_id + ".ndjson")).exists()
        assert not (batch_dir / (job_id + ".meta.json")).exists()
    finally:
        server.stop()


def test_batch_failed_job_ok_flag(tmp_path, coordinator):
    """the capture's ok flag (review F1): a served non-streaming job is
    ok=true; a job whose relay is closed with NO lines at all - a
    coordinator-side terminal (expiry, restart) pushes nothing on the relay,
    its error goes to the subscribe stream - is done/ok=false, never
    ok=true with an empty response"""
    preset = str(tmp_path / "preset.ini")
    _write_preset(preset, LOCAL_MODEL_ID, POOL_TEXT_MD)
    batch_dir = tmp_path / "batches"
    os.environ["POOL_BATCH_DIR"] = str(batch_dir)
    server = _start_router(tmp_path, coordinator.url(), preset)
    try:
        coordinator.wait_for("handshake", lambda e: True)
        coordinator.wait_for_sse()

        status, body = _submit_batch(server, POOL_TEXT_MD, [
            _blob_batch_job("served blob job"),
            _blob_batch_job("failed blob job"),
        ])
        assert status == 202, (status, body)
        batch_id = body["batch_id"]
        job_ids = body["job_ids"]
        assert len(job_ids) == 2

        deadline = time.time() + 60
        while time.time() < deadline and coordinator.intake_count() < 2:
            time.sleep(0.1)
        assert coordinator.intake_count() >= 2, coordinator.intakes
        # the capture threads have PUT the bodies and hold their relays open
        time.sleep(1.5)

        blob = json.dumps({"choices": [{"message": {"role": "assistant", "content": "WS10-BLOB-OK"}}]})
        # job 1: served - the child's blob response, one line
        coordinator._serve_job(job_ids[0], [blob])
        # job 2: the relay closes with no lines (F1's case)
        coordinator._serve_job(job_ids[1], [])

        body = _wait_batch_done(server, batch_id)
        jobs = _jobs_of(body)
        served, failed = jobs[job_ids[0]], jobs[job_ids[1]]
        assert served["status"] == "done", served
        assert served["ok"] is True, served
        assert "WS10-BLOB-OK" in served["response"], served
        assert failed["status"] == "done", failed
        assert failed["ok"] is False, failed
        assert failed["response"] == "", failed
        # the ok flag on disk matches the poll
        assert json.loads((batch_dir / (job_ids[1] + ".meta.json")).read_text())["ok"] is False
    finally:
        server.stop()
