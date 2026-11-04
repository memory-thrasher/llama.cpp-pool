#include "pool-batches.h"

#include "common.h"
#include "log.h"
#include "json.h"
#include "server-common.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef _WIN32
#  include <ctime>
#  include <sys/stat.h>
#endif

namespace pool {

namespace {

const int PROTO_VERSION = 1;
const int CONNECT_TIMEOUT_S = 10;
const int EVENT_TIMEOUT_S = 5;
const int READ_TIMEOUT_S = 120;  // a held-open GET
const uint64_t DEFAULT_RESULT_TTL_MS = 86400000; // 24h
const uint64_t MAX_BATCH_JOBS = 50;
const uint64_t SWEEP_INTERVAL_S = 30;

// the router's data dir for batch results (POOL_BATCH_DIR override, tests)
std::string batches_dir() {
    const std::string dir = common_get_env("POOL_BATCH_DIR");
    return dir.empty() ? "data/pool-batches" : dir;
}

std::string rfc3339_now() {
    const std::time_t t = std::time(nullptr);
    std::tm tm_utc{};
#ifdef _WIN32
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return buf;
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

std::string file_path(const std::string & dir, const std::string & name) {
    return dir + "/" + name;
}

// a router-generated uuid: 8-4-4-4-12 hex. the ids in the files are all
// router-generated, so the shape check is the path-safety gate
bool is_uuid_shape(const std::string & s) {
    if (s.size() != 36) {
        return false;
    }
    for (int i = 0; i < 36; i++) {
        const bool dash = (i == 8 || i == 13 || i == 18 || i == 23);
        const char c = s[(size_t) i];
        if (dash ? c != '-' : !std::isxdigit((unsigned char) c)) {
            return false;
        }
    }
    return true;
}

bool write_file(const std::string & path, const std::string & content) {
    std::ofstream f(path, std::ios::trunc | std::ios::binary);
    if (!f) {
        return false;
    }
    f << content;
    return (bool) f;
}

// true when the file exists and its content was read
bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    out.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return true;
}

// the file age in ms; -1 when the mtime is unreadable. stat directly -
// std::filesystem::last_write_time is not system_clock based on this
// toolchain and its age math is wrong (it swept fresh batches at once)
int64_t file_age_ms(const std::string & path) {
#ifdef _WIN32
    (void) path;
    return -1; // no sweep on an unreadable age: the TTL just never fires
#else
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) {
        return -1;
    }
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    const int64_t now_ms = (int64_t) ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    const int64_t mtime_ms = (int64_t) st.st_mtim.tv_sec * 1000 + st.st_mtim.tv_nsec / 1000000;
    return now_ms - mtime_ms;
#endif
}

// the neutral error body (no request content)
json err_obj(const std::string & message, const std::string & type) {
    return {{"error", {{"type", type}, {"message", message}}}};
}

// the terminal of one captured job: ok = the relay stream saw the child's
// [DONE] and no structured error chunk
void write_job_meta(const std::string & dir, const std::string & job_id, bool ok) {
    const json meta = {
        {"ok", ok},
        {"done_at", rfc3339_now()},
    };
    const std::string path = file_path(dir, job_id + ".meta.json");
    if (!write_file(path, safe_json_to_str(meta))) {
        LOG_WRN("pool: batch meta write failed: job_id=%s\n", job_id.c_str());
    }
}

} // namespace

pool_batches::pool_batches(const pool_config & cfg, pool_classes & classes)
        : cfg_(cfg), classes_(classes) {
    const std::string ttl_env = common_get_env("POOL_BATCH_RESULT_TTL_MS");
    char * end = nullptr;
    const long long parsed = std::strtoll(ttl_env.c_str(), &end, 10);
    ttl_ms_ = (parsed > 0) ? parsed : (int64_t) DEFAULT_RESULT_TTL_MS;
    last_sweep_ = std::chrono::steady_clock::now();
    if (cfg_.enabled()) {
        sweeper_ = std::thread([this]() {
            while (!stopping_.load()) {
                for (int i = 0; i < (int) SWEEP_INTERVAL_S && !stopping_.load(); i++) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
                if (stopping_.load()) {
                    return;
                }
                maybe_sweep();
            }
        });
    }
}

pool_batches::~pool_batches() {
    stopping_.store(true);
    if (sweeper_.joinable()) {
        sweeper_.join();
    }
}

std::string pool_batches::dir() const {
    return batches_dir();
}

int pool_batches::submit(const json & body, json & out) {
    if (!cfg_.enabled()) {
        out = err_obj("pool is not configured", "server_error");
        return 503;
    }
    const std::string class_name = json_str(body, "class");
    if (class_name.empty()) {
        out = err_obj("the \"class\" field is required", "invalid_request_error");
        return 400;
    }
    if (!body.contains("jobs") || !body["jobs"].is_array()) {
        out = err_obj("the \"jobs\" array is required", "invalid_request_error");
        return 400;
    }
    const json & jobs = body["jobs"];
    if (jobs.empty()) {
        out = err_obj("the \"jobs\" array must not be empty", "invalid_request_error");
        return 400;
    }
    if (jobs.size() > MAX_BATCH_JOBS) {
        out = err_obj("too many jobs (max " + std::to_string(MAX_BATCH_JOBS) + ")", "invalid_request_error");
        return 400;
    }
    for (size_t i = 0; i < jobs.size(); i++) {
        if (!jobs[i].is_object()) {
            out = err_obj("jobs[" + std::to_string(i) + "] is not a JSON object", "invalid_request_error");
            return 400;
        }
    }
    // the class must exist (the merged advertisement), else the jobs would
    // intake onto a tag no member can serve
    const auto ads = classes_.get();
    if (ads.find(class_name) == ads.end()) {
        out = err_obj("class not found: " + class_name, "invalid_request_error");
        return 400;
    }

    maybe_sweep();

    const std::string batch_id = new_uuid();
    const std::string dir = this->dir();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

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

    // phase 1: intake every job (priority batch, the D-G5 deterministic id).
    // a hard refusal of one job fails the submit: the batch is not recorded,
    // the already-intaked jobs stay queued on the coordinator (they expire on
    // its side), and nothing is captured
    json job_ids = json::array();
    json duplicates = json::object();
    bool intake_failed = false;
    for (size_t i = 0; i < jobs.size() && !intake_failed; i++) {
        const json & jb = jobs[i];
        const std::string raw = jb.dump();
        const json intake = {
            {"request_id", pool_request_id(raw)},
            {"tag", class_name},
            {"est_prompt_tokens", (uint64_t) (raw.size() / 4)},
            {"est_ctx_needed", (uint64_t) json_int(jb, "n_ctx", 0)},
            {"priority", "batch"},
            {"conv_key", json(pool_conv_key(raw))},
            {"client_hint", json(nullptr)},
        };
        auto res = cli.Post("/pool/jobs", auth_headers, intake.dump(), "application/json");
        if (!res || (res->status != 201 && res->status != 200)) {
            LOG_WRN("pool: batch intake failed: batch_id=%s job_index=%d status=%d\n",
                    batch_id.c_str(), (int) i, res ? res->status : 0);
            intake_failed = true;
            break;
        }
        std::string job_id;
        try {
            job_id = json_str(json::parse(res->body), "job_id");
        } catch (const std::exception &) {
            // fall through to the empty check
        }
        if (job_id.empty()) {
            LOG_WRN("pool: batch intake bad response: batch_id=%s job_index=%d\n", batch_id.c_str(), (int) i);
            intake_failed = true;
            break;
        }
        job_ids.push_back(job_id);
        if (res->status == 200) {
            duplicates[job_id] = true; // the D-G5 dedupe: the existing job stands
        }
    }
    if (intake_failed) {
        out = err_obj("pool intake failed (server_busy)", "server_error");
        return 502;
    }

    // phase 2: the batch record, then the detached capture flows
    const json batch = {
        {"batch_id", batch_id},
        {"class", class_name},
        {"created_at", rfc3339_now()},
        {"job_ids", job_ids},
        {"duplicates", duplicates},
    };
    if (!write_file(file_path(dir, batch_id + ".json"), safe_json_to_str(batch))) {
        LOG_WRN("pool: batch file write failed: batch_id=%s\n", batch_id.c_str());
        out = err_obj("failed to record the batch", "server_error");
        return 500;
    }
    for (size_t i = 0; i < jobs.size(); i++) {
        const std::string job_id = job_ids[i].get<std::string>();
        if (duplicates.contains(job_id)) {
            continue; // the existing job stands, no second capture
        }
        const json & jb = jobs[i];
        const std::string raw = jb.dump();
        const int64_t est_prompt = (int64_t) (raw.size() / 4);
        const int64_t est_ctx = json_int(jb, "n_ctx", 0);
        std::thread([this, batch_id, job_id, raw, est_prompt, est_ctx]() {
            capture_job(batch_id, job_id, raw, est_prompt, est_ctx);
        }).detach();
    }
    LOG_INF("pool: batch submitted: batch_id=%s class=%s jobs=%d duplicates=%d\n",
            batch_id.c_str(), class_name.c_str(), (int) job_ids.size(), (int) duplicates.size());

    out = {
        {"batch_id", batch_id},
        {"job_ids", job_ids},
    };
    return 202;
}

void pool_batches::mark_duplicate(const std::string & batch_id, const std::string & job_id) {
    const std::string dir = this->dir();
    const std::string path = file_path(dir, batch_id + ".json");
    std::string content;
    json batch;
    {
        std::lock_guard<std::mutex> lk(files_mutex_);
        if (!read_file(path, content)) {
            return; // the sweep took it, or it never was; the poll reports pending
        }
        try {
            batch = json::parse(content);
        } catch (const std::exception &) {
            return;
        }
        if (!batch.is_object() || !batch.contains("duplicates") || !batch["duplicates"].is_object()) {
            return;
        }
        if (batch["duplicates"].contains(job_id)) {
            return;
        }
        batch["duplicates"][job_id] = true;
        if (!write_file(path, safe_json_to_str(batch))) {
            LOG_WRN("pool: batch file update failed: batch_id=%s job_id=%s\n", batch_id.c_str(), job_id.c_str());
        }
    }
    LOG_INF("pool: batch job duplicate: batch_id=%s job_id=%s (deduped, no second capture)\n",
            batch_id.c_str(), job_id.c_str());
}

void pool_batches::capture_job(const std::string & batch_id, const std::string & job_id,
                               const std::string & raw_body, int64_t est_prompt_tokens, int64_t est_ctx_needed) {
    (void) est_prompt_tokens;
    (void) est_ctx_needed;
    httplib::Client cli(cfg_.server);
    cli.set_connection_timeout(CONNECT_TIMEOUT_S);
    cli.set_read_timeout(EVENT_TIMEOUT_S);
    cli.set_write_timeout(EVENT_TIMEOUT_S);
    const httplib::Headers auth_headers = {
        {"Authorization", "Bearer " + cfg_.token},
        {"X-Pool-Proto", std::to_string(PROTO_VERSION)},
        {"X-Pool-Router-Id", cfg_.router_id},
    };

    // the body PUT, retried briefly: a transient refusal must not orphan the job
    bool staged = false;
    bool duplicate = false;
    for (int attempt = 0; attempt < 3 && !staged && !duplicate; attempt++) {
        if (attempt > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        auto res = cli.Put("/pool/jobs/" + job_id + "/body", auth_headers, raw_body, "application/octet-stream");
        if (!res) {
            continue;
        }
        if (res->status == 200) {
            staged = true;
        } else if (res->status == 409) {
            duplicate = true; // the existing job already has its body
        }
    }
    if (duplicate) {
        mark_duplicate(batch_id, job_id);
        return;
    }
    if (!staged) {
        LOG_WRN("pool: batch body put failed: batch_id=%s job_id=%s (the job stays pending on the coordinator)\n",
                batch_id.c_str(), job_id.c_str());
        return;
    }

    // the client's expected framing: a stream is text/event-stream, else a
    // blob (the child's [DONE] only exists on a stream)
    bool streaming = false;
    try {
        const json body_obj = json::parse(raw_body);
        if (body_obj.is_object() && body_obj.contains("stream") && body_obj["stream"].is_boolean()) {
            streaming = body_obj["stream"].get<bool>();
        }
    } catch (const std::exception &) {
        // keep the default framing
    }

    // the held-open relay, captured to <job_id>.ndjson. the relay wraps each
    // pushed line in its own SSE framing ("data: <line>"); the pushed line is
    // the child's line verbatim ("data: <chunk>" on a stream, the blob line
    // otherwise). the capture stores the pushed line, one layer stripped -
    // what a live client of the same job would have received
    const std::string dir = this->dir();
    const std::string ndjson_path = file_path(dir, job_id + ".ndjson");
    std::ofstream ndjson(ndjson_path, std::ios::trunc);
    if (!ndjson) {
        LOG_WRN("pool: batch capture open failed: job_id=%s\n", job_id.c_str());
        return;
    }

    std::string sse_buf;
    bool saw_done = false;
    bool saw_error = false;
    size_t bytes = 0;

    httplib::Client cli2(cfg_.server);
    cli2.set_connection_timeout(CONNECT_TIMEOUT_S);
    cli2.set_read_timeout(READ_TIMEOUT_S); // held open until the relay is terminal
    cli2.set_write_timeout(CONNECT_TIMEOUT_S);
    const httplib::Headers headers = {
        {"Authorization", "Bearer " + cfg_.token},
        {"X-Pool-Proto", std::to_string(PROTO_VERSION)},
        {"X-Pool-Router-Id", cfg_.router_id},
        {"Accept", "text/event-stream"},
    };
    cli2.Get("/pool/jobs/" + job_id + "/response", headers,
             [](const httplib::Response &) -> bool { return true; },
             [&](const char * d, size_t len) -> bool {
                 sse_buf.append(d, len);
                 size_t start = 0;
                 while (true) {
                     const size_t nl = sse_buf.find('\n', start);
                     if (nl == std::string::npos) {
                         break;
                     }
                     std::string line = sse_buf.substr(start, nl - start);
                     start = nl + 1;
                     if (!line.empty() && line.back() == '\r') {
                         line.pop_back();
                     }
                     if (line.empty()) {
                         continue;
                     }
                     if (line.rfind("data:", 0) != 0) {
                         continue; // a relay heartbeat comment
                     }
                     // strip the relay's framing: what remains is the pushed line
                     std::string pushed = line.substr(5);
                     if (!pushed.empty() && pushed[0] == ' ') {
                         pushed.erase(0, 1);
                     }
                     ndjson << pushed << "\n";
                     bytes += pushed.size() + 1;
                     // strip the child's framing for the terminal probe
                     std::string probe = pushed;
                     if (probe.rfind("data:", 0) == 0) {
                         probe = probe.substr(5);
                         if (!probe.empty() && probe[0] == ' ') {
                             probe.erase(0, 1);
                         }
                     }
                     if (probe == "[DONE]") {
                         saw_done = true;
                         continue;
                     }
                     try {
                         const json chunk = json::parse(probe);
                         if (chunk.is_object() && chunk.contains("error") && chunk["error"].is_object()) {
                             saw_error = true;
                         }
                     } catch (const std::exception &) {
                         // not JSON on this line; the capture is verbatim either way
                     }
                 }
                 sse_buf.erase(0, start);
                 return true;
             });

    // the relay ended: the terminal of the job. ok = no structured error
    // chunk, and the child's [DONE] on a stream (a blob has no [DONE]); a
    // mid-stream coordinator terminal is an error chunk, already accounted
    const bool ok = !saw_error && (!streaming || saw_done);
    if (ndjson) {
        ndjson.flush();
    }
    write_job_meta(dir, job_id, ok);
    LOG_INF("pool: batch capture done: batch_id=%s job_id=%s ok=%d bytes=%zu\n",
            batch_id.c_str(), job_id.c_str(), (int) ok, bytes);
}

void pool_batches::maybe_sweep() {
    std::lock_guard<std::mutex> lk(files_mutex_);
    const auto now = std::chrono::steady_clock::now();
    if (now - last_sweep_ < std::chrono::seconds(1)) {
        return; // the request paths call this on every request
    }
    last_sweep_ = now;

    const std::string dir = this->dir();
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
        return;
    }
    int removed = 0;
    for (const auto & entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }
        const std::string name = entry.path().filename().string();
        // a batch file is <uuid>.json (41 chars); the per-job metas are swept with it
        if (name.size() != 41 || name.substr(36) != ".json" || !is_uuid_shape(name.substr(0, 36))) {
            continue;
        }
        const int64_t age = file_age_ms(entry.path().string());
        if (age < 0 || age <= ttl_ms_) {
            continue;
        }
        const std::string batch_id = name.substr(0, 36);
        std::vector<std::string> jobs;
        std::string content;
        if (read_file(entry.path().string(), content)) {
            try {
                const json batch = json::parse(content);
                if (batch.is_object() && batch.contains("job_ids") && batch["job_ids"].is_array()) {
                    for (const auto & j : batch["job_ids"]) {
                        if (j.is_string()) {
                            jobs.push_back(j.get<std::string>());
                        }
                    }
                }
            } catch (const std::exception &) {
                // sweep the batch file alone
            }
        }
        for (const auto & job_id : jobs) {
            std::filesystem::remove(file_path(dir, job_id + ".ndjson"), ec);
            std::filesystem::remove(file_path(dir, job_id + ".meta.json"), ec);
        }
        std::filesystem::remove(entry.path(), ec);
        removed++;
    }
    if (removed > 0) {
        LOG_INF("pool: batch sweep: removed %d expired batch(es) (ttl %lld ms)\n", removed, (long long) ttl_ms_);
    }
}

int pool_batches::poll(const std::string & batch_id, json & out) {
    if (!is_uuid_shape(batch_id)) {
        out = err_obj("batch not found", "not_found_error");
        return 404;
    }
    const std::string dir = this->dir();
    std::string content;
    if (!read_file(file_path(dir, batch_id + ".json"), content)) {
        out = err_obj("batch not found", "not_found_error");
        return 404;
    }
    json batch;
    try {
        batch = json::parse(content);
    } catch (const std::exception &) {
        out = err_obj("batch not found", "not_found_error");
        return 404;
    }
    if (!batch.is_object()) {
        out = err_obj("batch not found", "not_found_error");
        return 404;
    }
    maybe_sweep();
    json job_entries = json::array();
    if (batch.contains("job_ids") && batch["job_ids"].is_array()) {
        const json dupes = batch.value("duplicates", json::object());
        for (const auto & j : batch["job_ids"]) {
            if (!j.is_string()) {
                continue;
            }
            const std::string job_id = j.get<std::string>();
            json entry = {{"job_id", job_id}};
            if (dupes.is_object() && dupes.contains(job_id)) {
                entry["status"] = "duplicate";
                entry["ok"] = json(nullptr);
            } else {
                std::string meta_str;
                bool done = false;
                bool ok = false;
                if (read_file(file_path(dir, job_id + ".meta.json"), meta_str)) {
                    try {
                        const json meta = json::parse(meta_str);
                        if (meta.is_object()) {
                            done = true;
                            ok = meta.value("ok", false);
                        }
                    } catch (const std::exception &) {
                        done = false;
                    }
                }
                if (done) {
                    entry["status"] = "done";
                    entry["ok"] = ok;
                    std::string ndjson;
                    read_file(file_path(dir, job_id + ".ndjson"), ndjson);
                    entry["response"] = ndjson;
                } else {
                    entry["status"] = "pending";
                    entry["ok"] = json(nullptr);
                }
            }
            job_entries.push_back(entry);
        }
    }
    out = {
        {"batch_id", json_str(batch, "batch_id")},
        {"class", json_str(batch, "class")},
        {"jobs", job_entries},
    };
    return 200;
}

} // namespace pool
