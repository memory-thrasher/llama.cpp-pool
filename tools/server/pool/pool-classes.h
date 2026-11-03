#pragma once

#include "pool-config.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace pool {

// a loaded local model carrying a pool tag: one entry per (model, tag)
struct pool_local_member {
    std::string tag;
    std::string name;
    uint64_t n_ctx = 1;
    uint64_t n_slots = 1;
    uint64_t busy_slots = 0;
};

// one class as advertised by the coordinator, per protocol catalog 6
struct pool_class_entry {
    std::string tag;
    uint64_t member_count = 1;
    uint64_t advertised_ctx = 1;
    uint64_t min_ctx = 1;
    uint64_t max_ctx = 1;
};

// one advertised class as merged for the router: the coordinator numbers when
// the tag is in the coordinator state (they already count the local members,
// no double count), the local-only numbers otherwise (D-C8 degradation)
struct pool_class_ad {
    uint64_t member_count = 0;    // loaded members, always >= 1 when listed
    uint64_t advertised_ctx = 0;
    uint64_t min_ctx = 0;
    uint64_t max_ctx = 0;
};

// the merged class advertisement state of the router (D-C1, D-C8): the
// coordinator's GET /pool/classes, pulled with a TTL (LLAMA_POOL_ADVERT_TTL_MS,
// default 2000 ms) and merged with the local members. a failed pull, or no
// coordinator configured, serves the local-only state
class pool_classes {
public:
    // local_members: the loaded models with the pool tags they carry and their
    // slot state, called from the HTTP threads. a disabled cfg = no coordinator
    pool_classes(const pool_config & cfg, std::function<std::vector<pool_local_member>()> local_members);

    // the current advertisement, tag -> entry, refreshed on a cold cache
    std::map<std::string, pool_class_ad> get();

    // the local member to serve the class: a free slot, most headroom over
    // est_ctx first; empty string when none (WS-05 enqueues on the coordinator)
    std::string pick_local_member(const std::string & tag, int64_t est_ctx);

private:
    // pull GET /pool/classes into coord_ and set coord_ok_; metadata only
    bool pull();
    bool stale() const;

    pool_config cfg_;
    std::function<std::vector<pool_local_member>()> local_members_;
    int64_t ttl_ms_;

    std::mutex mutex_;
    std::chrono::steady_clock::time_point last_pull_;
    bool coord_ok_ = false;
    std::vector<pool_class_entry> coord_;
};

// D-E1: the continuity key of a conversation, SHA-256 hex (64 chars) of the
// first 8192 bytes of the raw client payload. no parsing, no template transform
std::string pool_conv_key(const std::string & raw_body);

// D-G5: the intake request id, a deterministic UUID derived from the FULL raw
// request body (SHA-256, first 16 bytes, 8-4-4-4-12 lowercase). same body ->
// same id; different body -> different id
std::string pool_request_id(const std::string & raw_body);

}