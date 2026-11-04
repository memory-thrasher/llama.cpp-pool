#pragma once

#include "pool-config.h"
#include "server-http.h"
#include "json.h"

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>

using json = common_json;

// forward declared to avoid a cycle with server-models.h
class server_models;
namespace pool { class pool_classes; }

namespace pool {

// one job this router is serving (the serving side). in memory only: no body
// or response content is stored (anti-memory)
struct pool_job_state {
    std::string job_id;
    std::string tag;
    std::string model_id;
    std::string conv_key;
    int64_t est_prompt_tokens = 0;
    int64_t est_ctx_needed    = 0;
    uint64_t accept_time_ms   = 0;
};

// the dispatch core of this router (WS-05). the serving side runs off the event
// bus: work_offer -> membership check -> accept/reject -> body pull -> serve the
// local child -> response push -> completion/error. the originating side turns a
// local-first miss into an intake + held-open response relay to the coordinator.
// state is in memory only
class pool_dispatch {
public:
    pool_dispatch(const pool_config & cfg, server_models & models, pool_classes & classes);
    ~pool_dispatch();

    pool_dispatch(const pool_dispatch &) = delete;
    pool_dispatch & operator=(const pool_dispatch &) = delete;

    // the originating side: no local free member for the class, so intake the job
    // on the coordinator (POST /pool/jobs + PUT body), hold the response open
    // (GET .../response) and stream the relayed bytes to the client as they arrive
    server_http_res_ptr origin_request(const std::string & tag, const std::string & raw_body,
                                       int64_t est_prompt_tokens, int64_t est_ctx_needed);

    // GET /pool/status (WS-16): the local counters, counts only (anti-memory).
    // serving: this router as a serving member; originating: this router as a
    // job submitter with the response relay held open
    uint64_t serving_pending() const;
    uint64_t served_total() const;
    uint64_t tokens_served_total() const;
    uint64_t originating_pending() const;
    uint64_t originating_total() const;

    // exclusion marks (D-F1): the owner's /models/unload on a pool member is a
    // mark in this router's local state, not a real unload. the mark lives in
    // router memory (a router is single-user); the coordinator DB is the
    // dispatch authority
    bool is_excluded(const std::string & model_name) const;
    // record the mark and post exclusion_update(excluded=true); the local mark
    // is applied even when the event send fails (it is logged)
    void mark_excluded(const std::string & model_name);
    // clear the mark and post exclusion_update(excluded=false)
    void unmark_excluded(const std::string & model_name);

private:
    // serving side, runs on the bus thread
    void on_envelope(const json & env);
    void handle_work_offer(const json & payload);
    // one job per thread: pull, serve, push, then completion or error
    void serve_job(std::string job_id, std::string tag, std::string model_id,
                   std::string conv_key, int64_t est_prompt_tokens, int64_t est_ctx_needed,
                   uint64_t member_ctx);

    // membership + selection (D-C7, second check). returns true with the member
    // to serve, or false with the precise reject reason (never a body snippet)
    bool pick_member(const std::string & tag, int64_t est_ctx, std::string & member,
                     uint64_t & n_ctx, std::string & reason) const;
    void post_event(const std::string & type, const json & payload) const;
    std::string pull_body(const std::string & job_id) const;
    bool serve_child(pool_job_state & job, const std::string & body, int & status, std::string & response);
    bool push_response(const std::string & job_id, const std::string & response) const;
    void post_error(pool_job_state & job, const std::string & stage, const std::string & code, const std::string & message);
    void post_completion(pool_job_state & job, const std::string & response);

    pool_config cfg_;
    server_models & models_;
    pool_classes & classes_;
    uint64_t bus_token_ = 0;

    mutable std::mutex excl_mutex_;
    std::set<std::string> excl_models_;

    // the GET /pool/status counters (WS-16): incremented at accept/intake,
    // decremented at the terminal; cumulative totals since process start
    std::atomic<uint64_t> serving_pending_{0};
    std::atomic<uint64_t> served_total_{0};
    std::atomic<uint64_t> tokens_served_total_{0};
    std::atomic<uint64_t> origin_pending_{0};
    std::atomic<uint64_t> origin_total_{0};
};

} // namespace pool
