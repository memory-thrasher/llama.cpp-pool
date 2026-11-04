#pragma once

#include "pool-config.h"
#include "pool-classes.h"
#include "json.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

using json = common_json;

namespace pool {

// async batch (D-H1): the client POSTs jobs to /pool/batches and polls
// /pool/batches/{id}. the client is gone after submit: no streaming, no held
// client socket. each job becomes one intake (priority batch) + body PUT on
// the coordinator, and the held-open response relay is captured to a file
// here instead of fed to a client pipe (D-I5: the coordinator stores no
// responses). files: <dir>/<batch_id>.json, <dir>/<job_id>.ndjson,
// <dir>/<job_id>.meta.json. a housekeeping tick removes batches older than
// the result TTL (POOL_BATCH_RESULT_TTL_MS, default 24h)
class pool_batches {
public:
    pool_batches(const pool_config & cfg, pool_classes & classes);
    ~pool_batches();

    pool_batches(const pool_batches &) = delete;
    pool_batches & operator=(const pool_batches &) = delete;

    // POST /pool/batches body: {"class": "pool-text-md", "jobs": [chat body, ...]}.
    // out: the full response body ({batch_id, job_ids} on 202, an error
    // object otherwise)
    int submit(const json & body, json & out);

    // GET /pool/batches/{id}. out: the full response body ({batch_id, class,
    // jobs: [...]} on 200, an error object otherwise)
    int poll(const std::string & batch_id, json & out);

private:
    // one detached job: body PUT (retried), then the held-open relay captured
    // to <job_id>.ndjson; <job_id>.meta.json on the terminal
    void capture_job(const std::string & batch_id, const std::string & job_id,
                     const std::string & raw_body, int64_t est_prompt_tokens, int64_t est_ctx_needed);
    // mark one job of an existing batch as duplicate (the intake deduped)
    void mark_duplicate(const std::string & batch_id, const std::string & job_id);
    // the housekeeping tick: remove batches older than the result TTL.
    // throttled to one run per second when called from the request paths
    void maybe_sweep();
    std::string dir() const;

    pool_config cfg_;
    pool_classes & classes_;
    int64_t ttl_ms_;

    mutable std::mutex files_mutex_; // serializes batch-file writes + the sweep
    std::chrono::steady_clock::time_point last_sweep_;
    std::thread sweeper_;
    std::atomic<bool> stopping_ {false};
};

} // namespace pool
