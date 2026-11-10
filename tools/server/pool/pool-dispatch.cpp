#include "pool-dispatch.h"

#include "pool-client.h"
#include "pool-classes.h"
#include "pool-reformat.h"
#include "server-models.h"
#include "common.h"
#include "log.h"
#include "json.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace pool {

namespace {

const int PROTO_VERSION = 1;
const int CONNECT_TIMEOUT_S = 10;
const int EVENT_TIMEOUT_S = 5;
const int READ_TIMEOUT_S = 120;   // a held-open GET or a child response
const int WRITE_TIMEOUT_S = 30;
const uint64_t OFFER_LEASE_MS = 5000;

uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string json_str(const json & o, const char * key) {
    if (o.is_object() && o.contains(key) && o[key].is_string()) {
        return o[key].get<std::string>();
    }
    return "";
}

int64_t json_int(const json & o, const char * key, int64_t fallback) {
    if (o.is_object() && o.contains(key) && o[key].is_number()) {
        return o[key].get<int64_t>();
    }
    return fallback;
}

// holds one slot of a pending counter (WS-16) for its scope: a serving job
// is in flight from accept until its terminal, an originating relay is
// pending from thread start until the relay is closed
struct pending_guard {
    std::atomic<uint64_t> & pending;

    explicit pending_guard(std::atomic<uint64_t> & p) : pending(p) {
        pending.fetch_add(1, std::memory_order_relaxed);
    }

    ~pending_guard() {
        pending.fetch_sub(1, std::memory_order_relaxed);
    }
};

// set the top-level "model" to the local model id and clamp max_tokens to the
// member's remaining context (D-C6). a body that does not parse is a failure
std::string rewrite_pool_body(const std::string & body, const std::string & model_id,
                              int64_t est_prompt_tokens, uint64_t n_ctx) {
    json obj;
    try {
        obj = json::parse(body);
    } catch (const std::exception &) {
        return "";
    }
    if (!obj.is_object()) {
        return "";
    }
    obj["model"] = model_id;
    // the context left for the completion after the prompt; the child still
    // rejects a genuine overflow, this only trims the over-optimistic requests
    const int64_t budget = (int64_t) n_ctx - est_prompt_tokens - 16;
    if (budget > 0) {
        for (const char * key : {"max_tokens", "max_completion_tokens"}) {
            if (obj.contains(key) && obj[key].is_number_integer()) {
                const int64_t v = obj[key].get<int64_t>();
                if (v > budget) {
                    obj[key] = budget;
                }
            }
        }
    }
    return obj.dump();
}

// the prompt/completion counts and finish reason of the response. a stream
// carries them on the last content chunk, a blob has them as fields
void parse_usage(const std::string & response, int64_t & prompt_tokens, int64_t & completion_tokens, std::string & stop_reason) {
    prompt_tokens = 0;
    completion_tokens = 0;
    size_t start = 0;
    while (true) {
        const size_t nl = response.find('\n', start);
        const bool last = nl == std::string::npos;
        std::string line = last ? response.substr(start) : response.substr(start, nl - start);
        if (!last) {
            start = nl + 1;
        }
        if (!line.empty()) {
            // a streamed line is "data: <json>"; strip the prefix before parsing
            if (line.rfind("data:", 0) == 0) {
                line = line.substr(5);
                if (!line.empty() && line[0] == ' ') {
                    line.erase(0, 1);
                }
            }
            if (line != "[DONE]") {
                json obj;
                try {
                    obj = json::parse(line);
                } catch (const std::exception &) {
                    obj = json();
                }
                if (obj.is_object()) {
                    // the standard OpenAI usage block
                    if (obj.contains("usage") && obj["usage"].is_object()) {
                        prompt_tokens     = json_int(obj["usage"], "prompt_tokens", prompt_tokens);
                        completion_tokens = json_int(obj["usage"], "completion_tokens", completion_tokens);
                    }
                    // llama.cpp streams a timings block in place of usage
                    if (obj.contains("timings") && obj["timings"].is_object()) {
                        prompt_tokens     = json_int(obj["timings"], "prompt_n", prompt_tokens);
                        completion_tokens = json_int(obj["timings"], "predicted_n", completion_tokens);
                    }
                    if (obj.contains("choices") && obj["choices"].is_array() && !obj["choices"].empty()
                            && obj["choices"][0].is_object()) {
                        const json & c0 = obj["choices"][0];
                        if (c0.contains("finish_reason") && c0["finish_reason"].is_string() && !c0["finish_reason"].empty()) {
                            stop_reason = c0["finish_reason"].get<std::string>();
                        }
                    }
                }
            }
        }
        if (last) {
            break;
        }
    }
}

} // namespace

pool_dispatch::pool_dispatch(const pool_config & cfg, server_models & models, pool_classes & classes)
        : cfg_(cfg), models_(models), classes_(classes) {
    bus_token_ = pool_instance().bus().subscribe([this](const json & env) {
        on_envelope(env);
    });
}

pool_dispatch::~pool_dispatch() {
    pool_instance().bus().unsubscribe(bus_token_);
}

void pool_dispatch::on_envelope(const json & env) {
    if (json_str(env, "type") != "work_offer") {
        return;
    }
    if (!env.contains("payload") || !env["payload"].is_object()) {
        return;
    }
    // the offer is handled on its own thread so the subscribe reader is not blocked
    json payload = env["payload"];
    std::thread([this, p = std::move(payload)]() {
        handle_work_offer(p);
    }).detach();
}

void pool_dispatch::handle_work_offer(const json & payload) {
    const std::string job_id = json_str(payload, "job_id");
    const std::string tag    = json_str(payload, "tag");
    const std::string conv_key = json_str(payload, "conv_key");
    // the origin's template identity (WS-17); "" when the offer predates the field
    const std::string origin_template_hash = json_str(payload, "origin_template_hash");
    const int64_t est_prompt  = json_int(payload, "est_prompt_tokens", 0);
    const int64_t est_ctx     = json_int(payload, "est_ctx_needed", 0);
    if (job_id.empty() || tag.empty()) {
        return;
    }

    // dispatch-time membership check (D-C7, the second one)
    std::string member;
    uint64_t member_ctx = 0;
    std::string reason;
    if (!pick_member(tag, est_ctx, member, member_ctx, reason)) {
        LOG_INF("pool: offer job_id=%s tag=%s rejected reason=%s\n", job_id.c_str(), tag.c_str(), reason.c_str());
        post_event("offer_reject", {
            {"job_id", job_id},
            {"reason", reason},
            {"detail", json(reason + " at dispatch")},
        });
        return;
    }

    LOG_INF("pool: offer job_id=%s tag=%s accepted model=%s n_ctx=%llu\n",
            job_id.c_str(), tag.c_str(), member.c_str(), (unsigned long long) member_ctx);
    post_event("offer_accept", {
        {"job_id", job_id},
        {"lease_ms", OFFER_LEASE_MS},
        {"model_id", member},
        {"n_ctx", (uint64_t) member_ctx},
    });

    serve_job(job_id, tag, member, conv_key, est_prompt, est_ctx, member_ctx, origin_template_hash);
}

// a loaded member with the tag and a free slot, most headroom over the
// estimated context first. the reject reason is precise, never a body snippet
bool pool_dispatch::pick_member(const std::string & tag, int64_t est_ctx, std::string & member,
                                uint64_t & n_ctx, std::string & reason) const {
    std::vector<pool_local_member> members;
    try {
        members = models_.pool_local_members();
    } catch (const std::exception &) {
        // no local state, treat as no member
    }
    bool any_tagged = false;
    bool any_free = false;
    const pool_local_member * best = nullptr;
    for (auto & m : members) {
        if (m.tag != tag) {
            continue;
        }
        any_tagged = true;
        if (m.busy_slots >= m.n_slots) {
            continue; // this member has no free slot
        }
        any_free = true;
        if (est_ctx > 0 && (int64_t) m.n_ctx < est_ctx) {
            continue; // a free slot, but not enough context
        }
        if (best == nullptr || (int64_t) m.n_ctx - est_ctx > (int64_t) best->n_ctx - est_ctx) {
            best = &m;
        }
    }
    if (best != nullptr) {
        member = best->name;
        n_ctx  = best->n_ctx;
        return true;
    }
    if (!any_tagged) {
        reason = "model_gone";
    } else if (!any_free) {
        reason = "no_slot";
    } else {
        reason = "ctx_insufficient";
    }
    return false;
}

void pool_dispatch::post_event(const std::string & type, const json & payload) const {
    const json env = {
        {"type", type},
        {"payload", payload},
    };
    httplib::Client cli(cfg_.server);
    cli.set_connection_timeout(CONNECT_TIMEOUT_S);
    cli.set_read_timeout(EVENT_TIMEOUT_S);
    cli.set_write_timeout(EVENT_TIMEOUT_S);
    const httplib::Headers headers = {
        {"Authorization", "Bearer " + cfg_.token},
        {"X-Pool-Proto", std::to_string(PROTO_VERSION)},
    };
    auto res = cli.Post("/pool/events", headers, env.dump(), "application/json");
    if (!res) {
        LOG_WRN("pool: dispatch event post failed: type=%s (no response)\n", type.c_str());
        return;
    }
    if (res->status != 200) {
        LOG_WRN("pool: dispatch event post failed: type=%s status=%d\n", type.c_str(), res->status);
        return;
    }
}

bool pool_dispatch::is_excluded(const std::string & model_name) const {
    std::lock_guard<std::mutex> lk(excl_mutex_);
    return excl_models_.count(model_name) > 0;
}

void pool_dispatch::mark_excluded(const std::string & model_name) {
    {
        std::lock_guard<std::mutex> lk(excl_mutex_);
        excl_models_.insert(model_name);
    }
    post_event("exclusion_update", {
        {"router_id", cfg_.router_id},
        {"model_id", model_name},
        {"excluded", true},
    });
    LOG_INF("pool: excluded model %s (owner mark, D-F1)\n", model_name.c_str());
}

void pool_dispatch::unmark_excluded(const std::string & model_name) {
    {
        std::lock_guard<std::mutex> lk(excl_mutex_);
        excl_models_.erase(model_name);
    }
    post_event("exclusion_update", {
        {"router_id", cfg_.router_id},
        {"model_id", model_name},
        {"excluded", false},
    });
    LOG_INF("pool: unexcluded model %s (owner mark cleared, D-F1)\n", model_name.c_str());
}

// phase 2 of the dispatch: download the raw client payload (catalog 5)
std::string pool_dispatch::pull_body(const std::string & job_id) const {
    httplib::Client cli(cfg_.server);
    cli.set_connection_timeout(CONNECT_TIMEOUT_S);
    cli.set_read_timeout(READ_TIMEOUT_S);
    const httplib::Headers headers = {
        {"Authorization", "Bearer " + cfg_.token},
        {"X-Pool-Proto", std::to_string(PROTO_VERSION)},
        // e2e fix: the coordinator requires the router identity on the body pull
        {"X-Pool-Router-Id", cfg_.router_id},
    };
    auto res = cli.Get("/pool/jobs/" + job_id + "/body", headers);
    if (!res) {
        LOG_WRN("pool: body pull failed: job_id=%s (no response)\n", job_id.c_str());
        return "";
    }
    if (res->status != 200) {
        LOG_WRN("pool: body pull failed: job_id=%s status=%d\n", job_id.c_str(), res->status);
        return "";
    }
    return res->body;
}

// push the response lines to the coordinator (catalog 6b); the relay is line
// oriented, the bytes are the child's verbatim
bool pool_dispatch::push_response(const std::string & job_id, const std::string & response) const {
    httplib::Client cli(cfg_.server);
    cli.set_connection_timeout(CONNECT_TIMEOUT_S);
    cli.set_read_timeout(EVENT_TIMEOUT_S);
    cli.set_write_timeout(WRITE_TIMEOUT_S);
    std::string body = response;
    if (!body.empty() && body.back() != '\n') {
        body += '\n';
    }
    const httplib::Headers headers = {
        {"Authorization", "Bearer " + cfg_.token},
        {"X-Pool-Proto", std::to_string(PROTO_VERSION)},
        // e2e fix: the coordinator requires the router identity on the response push
        {"X-Pool-Router-Id", cfg_.router_id},
        {"Content-Type", "application/x-ndjson"},
    };
    auto res = cli.Put("/pool/jobs/" + job_id + "/response", headers, body, "application/x-ndjson");
    if (!res) {
        LOG_WRN("pool: response push failed: job_id=%s (no response)\n", job_id.c_str());
        return false;
    }
    if (res->status != 200) {
        LOG_WRN("pool: response push failed: job_id=%s status=%d\n", job_id.c_str(), res->status);
        return false;
    }
    return true;
}

// forward the rewritten body to the local child and read the full response.
// the pool request is served on /v1/chat/completions (the pool class is a text
// class); the slot accounting of the router is the existing req_count
bool pool_dispatch::serve_child(pool_job_state & job, const std::string & body, int & status, std::string & response) {
    auto meta = models_.get_meta(job.model_id);
    if (!meta.has_value() || !meta->is_running()) {
        status = 503;
        return false;
    }
    std::function<bool()> stop_fn = [this]() { return false; };
    server_http_req req{ {}, {}, "/v1/chat/completions", "", body, {}, stop_fn };
    auto res = models_.proxy_request(req, "POST", job.model_id, false);
    status = res->status;
    std::string buf;
    while (true) {
        std::string chunk;
        const bool has_next = res->next(chunk);
        if (!chunk.empty()) {
            buf += chunk;
        }
        if (!has_next) {
            break;
        }
    }
    response = std::move(buf);
    return status >= 200 && status < 300;
}

void pool_dispatch::post_error(pool_job_state & job, const std::string & stage, const std::string & code, const std::string & message) {
    post_event("error", {
        {"job_id", job.job_id},
        {"ok", false},
        {"stage", stage},
        {"code", code},
        {"message", message},
        {"prompt_tokens", json(nullptr)},
        {"response_tokens", json(nullptr)},
        {"conv_key", job.conv_key.empty() ? json(nullptr) : json(job.conv_key)},
        {"t_queued_ms", json(nullptr)},
        {"t_service_ms", (uint64_t)(now_ms() - job.accept_time_ms)},
    });
}

void pool_dispatch::post_completion(pool_job_state & job, const std::string & response) {
    int64_t prompt_tokens = 0;
    int64_t completion_tokens = 0;
    std::string stop_reason;
    parse_usage(response, prompt_tokens, completion_tokens, stop_reason);
    // the serving counters of GET /pool/status: one job served, plus its
    // completion tokens (anti-memory: counts only)
    served_total_.fetch_add(1, std::memory_order_relaxed);
    tokens_served_total_.fetch_add((uint64_t) completion_tokens, std::memory_order_relaxed);
    post_event("completion", {
        {"job_id", job.job_id},
        {"ok", true},
        {"conv_key", job.conv_key.empty() ? json(nullptr) : json(job.conv_key)},
        {"model_id", job.model_id},
        {"prompt_tokens", (uint64_t) prompt_tokens},
        {"response_tokens", (uint64_t) completion_tokens},
        {"cached_tokens", json(nullptr)},
        {"idle_ms_hint", json(nullptr)},
        {"t_queued_ms", (uint64_t) 0},
        {"t_service_ms", (uint64_t)(now_ms() - job.accept_time_ms)},
        {"stop_reason", stop_reason.empty() ? json(nullptr) : json(stop_reason)},
        {"served_by", cfg_.router_id},
    });
}

void pool_dispatch::serve_job(std::string job_id, std::string tag, std::string model_id,
                              std::string conv_key, int64_t est_prompt_tokens, int64_t est_ctx_needed,
                              uint64_t member_ctx, std::string origin_template_hash) {
    const pending_guard pending(serving_pending_);
    pool_job_state job;
    job.job_id   = job_id;
    job.tag      = tag;
    job.model_id = model_id;
    job.conv_key = conv_key;
    job.est_prompt_tokens = est_prompt_tokens;
    job.est_ctx_needed    = est_ctx_needed;
    job.accept_time_ms = now_ms();

    const std::string pulled = pull_body(job_id);
    if (pulled.empty()) {
        post_event("body_abort", {{"job_id", job_id}, {"reason", "other"}});
        post_error(job, "pre_token", "other", "body pull failed");
        return;
    }

    const std::string rewritten = rewrite_pool_body(pulled, job.model_id, est_prompt_tokens, member_ctx);
    if (rewritten.empty()) {
        post_error(job, "pre_token", "other", "body rewrite failed");
        return;
    }

    // WS-17: cross-template reformat (D-M1/D-M2). same template (or the origin
    // had no local member) is a byte-identical skip; a different template
    // re-serializes the messages under the serving template. "" is a terminal
    // pre_token reformat_failed - the body is never served malformed
    const std::string serving_hash = models_.pool_member_template_hash(job.model_id);
    const std::string serving_src  = models_.pool_member_template_source(job.model_id);
    const std::string reformatted  = pool_reformat_body(rewritten, origin_template_hash, serving_hash, serving_src);
    if (reformatted.empty()) {
        post_error(job, "pre_token", "reformat_failed", "cross-template reformat failed");
        return;
    }

    int status = 0;
    std::string response;
    if (!serve_child(job, reformatted, status, response)) {
        // an immediate child failure before any token: the coordinator re-offers (D-G1)
        const std::string code = (status == 413 || status == 400) ? "overflow" : "crash";
        post_error(job, "pre_token", code, status == 200 ? "child stream ended early" : "child request failed");
        return;
    }

    if (!push_response(job_id, response)) {
        // the push closed: the client gave up (D-G4)
        post_error(job, "mid_stream", "client_disconnected", "response relay closed");
        return;
    }

    post_completion(job, response);
    LOG_INF("pool: job_id=%s completed model=%s\n", job_id.c_str(), model_id.c_str());
}

//
// the originating side: a held-open response relay to the client
//

// the held-open response of a pool job: a background thread keeps the
// coordinator GET .../response open and feeds the pipe the client reads from
struct server_http_pool_origin : server_http_res {
    std::shared_ptr<server_pipe<std::string>> pipe;
    std::shared_ptr<httplib::Client> cli;
    std::thread thread;
    std::shared_ptr<std::atomic<bool>> ka_stop; // first real data sent, or relay ended
    std::thread ka_thread;                      // client keep-alive while queued

    ~server_http_pool_origin() {
        if (ka_stop) {
            ka_stop->store(true, std::memory_order_relaxed);
        }
        if (ka_thread.joinable()) {
            ka_thread.join();
        }
        if (cli) {
            cli->stop(); // ends the held-open GET in the thread
        }
        if (pipe) {
            pipe->close_read();
        }
        if (thread.joinable()) {
            thread.join();
        }
    }
};

// feed the held-open response stream; emit the payload of each completed
// "data:" line to the pipe, line terminated. heartbeats and blanks are dropped.
// saw_done records the child's stream terminator ("data: [DONE]") - the
// local signal that the served stream finished, for the client-facing
// terminal (D-G2)
// first_data: set once the first "data:" line is forwarded - stops the
// client keep-alive (see origin_request)
static bool sse_feed_response(const char * data, size_t len, std::string & buf,
                              server_pipe<std::string> & pipe, bool & saw_done,
                              std::atomic<bool> * first_data = nullptr) {
    buf.append(data, len);
    size_t start = 0;
    while (true) {
        const size_t nl = buf.find('\n', start);
        if (nl == std::string::npos) {
            break;
        }
        std::string line = buf.substr(start, nl - start);
        start = nl + 1;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.rfind("data:", 0) != 0) {
            continue; // a heartbeat comment or a blank line
        }
        std::string payload = line.substr(5);
        if (!payload.empty() && payload[0] == ' ') {
            payload.erase(0, 1);
        }
        payload += '\n'; // the client reads line oriented
        if (payload == "data: [DONE]\n") {
            saw_done = true;
        }
        if (first_data) {
            first_data->store(true, std::memory_order_relaxed);
        }
        if (!pipe.write(std::move(payload))) {
            return false;
        }
    }
    buf.erase(0, start);
    return true;
}

// a neutral error body, no request content
static std::string pool_origin_error(const std::string & message) {
    return safe_json_to_str({{"error", {{"type", "server_error"}, {"message", message}}}});
}

// the terminal state of one origin job, filled by the bus handler (the
// coordinator's terminal envelope, D-D6) and read by the relay thread for the
// client-facing terminal (D-G2, WS-09)
struct origin_terminal {
    std::mutex mu;
    bool have = false;
    bool is_error = false;
    std::string message; // the coordinator's reason, error case only (bounded)
};

// the client-facing terminal of an interrupted relay (D-G2, WS-09): the
// response started and then ended without the child's [DONE] (the serving
// member died mid-push and the coordinator's running cap cut the relay). The
// client sees a clean, structured end-of-stream error - never a raw hang or a
// half-open socket:
//   stream: a finish_reason chunk (network_error) + an error chunk + [DONE]
//   blob:   an error object (the whole body, or appended after partial bytes)
// The finish_reason network_error is deliberate: stock pi maps it to an error
// stop with a retryable message, and a retry is the only recovery for output
// lost to an infrastructure failure (D-J5: client-retry only when retry is
// actually desired)
static void render_interrupted(server_pipe<std::string> & pipe, bool streaming, const std::string & reason) {
    const json error_field = {{"type", "server_error"}, {"message", reason}};
    if (streaming) {
        const json finish_chunk = {
            {"id", "chatcmpl-pool"},
            {"object", "chat.completion.chunk"},
            {"created", (uint64_t) 0},
            {"model", "pool"},
            {"choices", json::array({{{"index", 0}, {"delta", json::object()}, {"finish_reason", "network_error"}}})},
            {"error", error_field},
        };
        const json error_chunk = {{"error", error_field}};
        pipe.write("data: " + finish_chunk.dump() + "\n");
        pipe.write("data: " + error_chunk.dump() + "\n");
        pipe.write("data: [DONE]\n");
    } else {
        pipe.write("\n" + safe_json_to_str({{"error", error_field}}) + "\n");
    }
}

// wait briefly for the coordinator's terminal envelope (it is mirrored just
// before the relay closes; the bus delivery may lag the relay EOF by a moment)
// and render the structured terminal unless the job completed
static void render_interrupted_terminal(server_pipe<std::string> & pipe, bool streaming,
                                        const std::shared_ptr<origin_terminal> & term) {
    bool have = false;
    bool is_error = false;
    std::string message;
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
        while (true) {
            std::lock_guard<std::mutex> lk(term->mu);
            if (term->have || std::chrono::steady_clock::now() >= deadline) {
                have = term->have;
                is_error = term->is_error;
                message = term->message;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    // the coordinator says the job failed: render with its reason
    if (is_error) {
        const std::string reason = message.empty()
                                       ? "pool: response ended without completion"
                                       : "pool: " + message;
        LOG_WRN("pool: client-facing terminal (error): %s\n", reason.c_str());
        render_interrupted(pipe, streaming, reason);
        return;
    }
    // the coordinator says the job completed: the stream stands as is. a
    // stream with NO terminal event in time ended as an interruption; a blob
    // stands as is (its body either parsed or it did not)
    if (streaming && !have) {
        const std::string reason =
            "pool: response ended without completion (serving member may have failed)";
        LOG_WRN("pool: client-facing terminal (interrupted): %s\n", reason.c_str());
        render_interrupted(pipe, streaming, reason);
    }
}

server_http_res_ptr pool_dispatch::origin_request(const std::string & tag, const std::string & raw_body,
                                                  int64_t est_prompt_tokens, int64_t est_ctx_needed) {
    auto res = std::make_unique<server_http_pool_origin>();
    res->pipe = std::make_shared<server_pipe<std::string>>();

    // the client's expected framing: a stream is text/event-stream, else a blob
    bool streaming = false;
    json body_json;
    try {
        body_json = json::parse(raw_body);
    } catch (const std::exception &) {
        // keep the default framing
    }
    if (body_json.is_object() && body_json.contains("stream") && body_json["stream"].is_boolean()) {
        streaming = body_json["stream"].get<bool>();
    }

    // 1. intake the metadata (catalog 4). the origin's template identity
    // (WS-17): the SHA-256 of the local member's template source - the member
    // that would have served - or null when this router has no local member for
    // the class. the serving side compares it to its own to skip or reformat
    const std::string origin_template_hash = models_.pool_local_member_template_hash(tag);
    const json intake = {
        {"request_id", pool_request_id(raw_body)}, // D-G5: deterministic prompt hash
        {"tag", tag},
        {"est_prompt_tokens", (uint64_t) est_prompt_tokens},
        {"est_ctx_needed", (uint64_t) est_ctx_needed},
        {"priority", "live"},
        {"conv_key", json(pool_conv_key(raw_body))},
        {"client_hint", json(nullptr)},
        {"origin_template_hash", origin_template_hash.empty() ? json(nullptr) : json(origin_template_hash)},
    };
    httplib::Client cli(cfg_.server);
    cli.set_connection_timeout(CONNECT_TIMEOUT_S);
    cli.set_read_timeout(EVENT_TIMEOUT_S);
    cli.set_write_timeout(EVENT_TIMEOUT_S);
    const httplib::Headers auth_headers = {
        {"Authorization", "Bearer " + cfg_.token},
        {"X-Pool-Proto", std::to_string(PROTO_VERSION)},
        {"X-Pool-Router-Id", cfg_.router_id},
        {"Content-Type", "application/json"},
    };
    auto intake_res = cli.Post("/pool/jobs", auth_headers, intake.dump(), "application/json");
    // 201 is a fresh job, 200 is a D-G5 dedupe (the existing job is returned);
    // anything else is a refusal
    if (!intake_res || (intake_res->status != 201 && intake_res->status != 200)) {
        // D-J5 RETRY: the coordinator is unreachable or refused the submit -
        // transient, a retry plausibly helps. 5xx + busy wording so stock pi's
        // retry pattern matches even from the body alone
        res->status = 502;
        res->data = pool_origin_error("pool intake failed (server_busy)");
        LOG_WRN("pool: intake failed: tag=%s status=%d\n", tag.c_str(), intake_res ? intake_res->status : 502);
        return res;
    }
    std::string job_id;
    try {
        const json intake_body = json::parse(intake_res->body);
        job_id = intake_body["job_id"].get<std::string>();
    } catch (const std::exception &) {
        // fall through to the empty check
    }
    if (job_id.empty()) {
        res->status = 502;
        res->data = pool_origin_error("pool intake failed (server_busy)");
        return res;
    }
    LOG_INF("pool: intake ok: tag=%s job_id=%s est_prompt=%lld est_ctx=%lld\n",
            tag.c_str(), job_id.c_str(), (long long) est_prompt_tokens, (long long) est_ctx_needed);
    // one more intake since process start (WS-16); the pending counter moves
    // when the relay below is held open, and back at its close
    origin_total_.fetch_add(1, std::memory_order_relaxed);

    // 2. put the raw body verbatim (catalog 4)
    auto body_res = cli.Put("/pool/jobs/" + job_id + "/body", auth_headers, raw_body, "application/octet-stream");
    if (!body_res) {
        // D-J5 RETRY, as above
        res->status = 502;
        res->data = pool_origin_error("pool body upload failed (server_busy)");
        LOG_WRN("pool: body put failed: job_id=%s status=%d\n", job_id.c_str(), body_res ? body_res->status : 502);
        return res;
    }
    if (body_res->status == 409) {
        // D-G5: the existing job is already running/completed, so the body is
        // refused. this is a duplicate retry, not a transient failure: a
        // structured 409 to the client, no retry
        res->status = 409;
        res->content_type = "application/json";
        res->data = safe_json_to_str({{"error", {{"type", "duplicate_request"},
            {"message", "duplicate request already accepted"}}}});
        LOG_INF("pool: duplicate intake: tag=%s job_id=%s (already accepted)\n", tag.c_str(), job_id.c_str());
        return res;
    }
    if (body_res->status != 200) {
        // D-J5 RETRY, as above
        res->status = 502;
        res->data = pool_origin_error("pool body upload failed (server_busy)");
        LOG_WRN("pool: body put failed: job_id=%s status=%d\n", job_id.c_str(), body_res->status);
        return res;
    }

    // 3. hold the response open and relay to the client (catalog 6b)
    res->status = 200;
    res->content_type = streaming ? "text/event-stream" : "application/json; charset=utf-8";
    res->headers["X-Accel-Buffering"] = "no";

    // the terminal envelope of this job (completion / error, D-D6): the
    // coordinator mirrors it on the subscribe stream just before closing the
    // relay (D-G2). the bus handler only records it; the relay thread renders
    // the client-facing terminal from it
    auto term = std::make_shared<origin_terminal>();
    const pool_event_bus::token_t term_token = pool_instance().bus().subscribe(
        [term, job_id](const json & env) {
            const std::string type = json_str(env, "type");
            if (type != "completion" && type != "error") {
                return;
            }
            if (!env.contains("payload") || !env["payload"].is_object()) {
                return;
            }
            if (json_str(env["payload"], "job_id") != job_id) {
                return;
            }
            std::lock_guard<std::mutex> lk(term->mu);
            if (term->have) {
                return;
            }
            term->have = true;
            term->is_error = (type == "error");
            if (term->is_error) {
                std::string msg = json_str(env["payload"], "message");
                if (msg.size() > 200) {
                    msg = msg.substr(0, 200); // bound router/coordinator-supplied text
                }
                term->message = msg;
            }
        });

    auto pipe = res->pipe;
    auto cli2 = std::make_shared<httplib::Client>(cfg_.server);
    cli2->set_connection_timeout(CONNECT_TIMEOUT_S);
    cli2->set_read_timeout(READ_TIMEOUT_S); // held open until the relay is terminal
    cli2->set_write_timeout(CONNECT_TIMEOUT_S);
    res->cli = cli2;

    // keep the client's idle timer alive while the job is queued or waiting
    // for the first token (the relay sends nothing until then; pi's default
    // http idle timeout is 30s): one SSE comment per 15s, stream requests
    // only - comment lines are ignored by OpenAI-compatible parsers
    if (streaming) {
        auto ka_stop = std::make_shared<std::atomic<bool>>(false);
        res->ka_stop = ka_stop;
        res->ka_thread = std::thread([pipe, ka_stop]() {
            int secs = 0;
            while (secs < 1800) { // bounded: the job expiry (30min default) ends the relay
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if (ka_stop->load(std::memory_order_relaxed) || ++secs >= 1800) {
                    break;
                }
                if (secs % 15 == 0 && !pipe->write(": keepalive\n\n")) {
                    break; // the reader is gone
                }
            }
        });
    }

    const std::string token = cfg_.token;
    const std::string router_id = cfg_.router_id;
    res->thread = std::thread([this, pipe, job_id, cli2, token, router_id, streaming, term, term_token,
                                ka_stop = res->ka_stop]() {
        const pending_guard pending(origin_pending_);
        const httplib::Headers headers = {
            {"Authorization", "Bearer " + token},
            {"X-Pool-Proto", std::to_string(PROTO_VERSION)},
            {"X-Pool-Router-Id", router_id},
            {"Accept", "text/event-stream"},
        };
        std::string sse_buf;
        bool saw_done = false;
        cli2->Get("/pool/jobs/" + job_id + "/response", headers,
                  [](const httplib::Response &) -> bool { return true; },
                  [&](const char * d, size_t len) -> bool {
                      return sse_feed_response(d, len, sse_buf, *pipe, saw_done, ka_stop ? ka_stop.get() : nullptr);
                  });
        // the relay ended: a stream that never saw the child's [DONE] is an
        // interruption (the serving member died mid-push) - render the clean,
        // structured terminal before closing (D-G2, WS-09). a completed stream
        // needs no terminal wait; a blob renders only on an error event
        if (!streaming || !saw_done) {
            render_interrupted_terminal(*pipe, streaming, term);
        }
        if (ka_stop) {
            ka_stop->store(true, std::memory_order_relaxed);
        }
        pipe->close_write();
        pool_instance().bus().unsubscribe(term_token);
    });

    res->next = [pipe](std::string & out) -> bool {
        std::string line;
        const bool has_next = pipe->read(line, nullptr);
        if (!line.empty()) {
            out = std::move(line);
        }
        return has_next;
    };
    return res;
}

uint64_t pool_dispatch::serving_pending() const {
    return serving_pending_.load(std::memory_order_relaxed);
}

uint64_t pool_dispatch::served_total() const {
    return served_total_.load(std::memory_order_relaxed);
}

uint64_t pool_dispatch::tokens_served_total() const {
    return tokens_served_total_.load(std::memory_order_relaxed);
}

uint64_t pool_dispatch::originating_pending() const {
    return origin_pending_.load(std::memory_order_relaxed);
}

uint64_t pool_dispatch::originating_total() const {
    return origin_total_.load(std::memory_order_relaxed);
}

} // namespace pool
