#include "pool-classes.h"

#include "common.h"
#include "json.h"
#include "log.h"
#include "hash/hash.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>

using json = common_json;

namespace pool {

namespace {

const int PROTO_VERSION = 1;
const int PULL_CONNECT_TIMEOUT_S = 3;
const int PULL_READ_TIMEOUT_S    = 5;
const int64_t DEFAULT_TTL_MS     = 2000;

uint64_t json_uint(const json & obj, const char * key, uint64_t fallback) {
    if (obj.is_object() && obj.contains(key) && obj[key].is_number()) {
        return obj[key].get<uint64_t>();
    }
    return fallback;
}

} // namespace

std::string pool_conv_key(const std::string & raw_body) {
    const size_t n = std::min(raw_body.size(), size_t(8192));
    return hash_sha256_hex(raw_body.data(), n);
}

pool_classes::pool_classes(const pool_config & cfg, std::function<std::vector<pool_local_member>()> local_members)
        : cfg_(cfg), local_members_(std::move(local_members)), ttl_ms_(DEFAULT_TTL_MS) {
    const std::string ttl = common_get_env("LLAMA_POOL_ADVERT_TTL_MS");
    if (!ttl.empty()) {
        try {
            const int64_t v = std::stoll(ttl);
            if (v > 0) {
                ttl_ms_ = v;
            }
        } catch (const std::exception &) {
            // keep the default on an invalid value
        }
    }
}

bool pool_classes::stale() const {
    return std::chrono::steady_clock::now() - last_pull_ >= std::chrono::milliseconds(ttl_ms_);
}

bool pool_classes::pull() {
    httplib::Client cli(cfg_.server);
    cli.set_connection_timeout(PULL_CONNECT_TIMEOUT_S);
    cli.set_read_timeout(PULL_READ_TIMEOUT_S);
    const httplib::Headers headers = {
        {"Authorization", "Bearer " + cfg_.token},
        {"X-Pool-Proto", std::to_string(PROTO_VERSION)},
    };
    auto res = cli.Get("/pool/classes", headers);
    if (!res) {
        LOG_DBG("pool: classes pull failed: cannot reach %s\n", cfg_.server.c_str());
        return false;
    }
    if (res->status != 200) {
        LOG_DBG("pool: classes pull failed: status=%d\n", res->status);
        return false;
    }
    json body;
    try {
        body = json::parse(res->body);
    } catch (const std::exception &) {
        return false;
    }
    if (!body.is_object() || !body.contains("classes") || !body["classes"].is_array()) {
        return false;
    }
    std::vector<pool_class_entry> out;
    for (auto & item : body["classes"]) {
        pool_class_entry e;
        if (item.is_object() && item.contains("tag") && item["tag"].is_string()) {
            e.tag = item["tag"].get<std::string>();
        }
        e.member_count   = json_uint(item, "member_count", 1);
        e.advertised_ctx = json_uint(item, "advertised_ctx", 1);
        e.min_ctx        = json_uint(item, "min_ctx", 1);
        e.max_ctx        = json_uint(item, "max_ctx", 1);
        // a class with no loaded member is absent, not listed with a zero count
        if (!e.tag.empty() && e.member_count >= 1) {
            out.push_back(std::move(e));
        }
    }
    coord_ = std::move(out);
    return true;
}

std::map<std::string, pool_class_ad> pool_classes::get() {
    std::lock_guard<std::mutex> lk(mutex_);

    // the local state, always available: one entry per tag of the loaded members
    std::map<std::string, pool_class_ad> local;
    if (local_members_) {
        std::vector<pool_local_member> members;
        try {
            members = local_members_();
        } catch (const std::exception &) {
            // the router state is unavailable, keep serving the coordinator state
        }
        for (auto & m : members) {
            pool_class_ad & ad = local[m.tag];
            ad.member_count++;
            if (ad.member_count == 1) {
                ad.advertised_ctx = m.n_ctx;
                ad.min_ctx        = m.n_ctx;
                ad.max_ctx        = m.n_ctx;
            } else {
                ad.advertised_ctx = std::max(ad.advertised_ctx, m.n_ctx);
                ad.min_ctx        = std::min(ad.min_ctx, m.n_ctx);
                ad.max_ctx        = std::max(ad.max_ctx, m.n_ctx);
            }
        }
    }

    if (cfg_.enabled() && stale()) {
        last_pull_ = std::chrono::steady_clock::now(); // one pull attempt per TTL
        coord_ok_ = pull();
    }

    std::map<std::string, pool_class_ad> out;
    for (auto & [tag, ad] : local) {
        out[tag] = ad;
    }
    // the coordinator count already includes the local members, it wins whole
    if (coord_ok_) {
        for (auto & c : coord_) {
            out[c.tag] = {c.member_count, c.advertised_ctx, c.min_ctx, c.max_ctx};
        }
    }
    return out;
}

std::string pool_classes::pick_local_member(const std::string & tag, int64_t est_ctx) {
    std::vector<pool_local_member> members;
    if (local_members_) {
        try {
            members = local_members_();
        } catch (const std::exception &) {
            // no local members available
        }
    }
    const pool_local_member * best = nullptr;
    for (auto & m : members) {
        if (m.tag != tag || m.busy_slots >= m.n_slots) {
            continue;
        }
        if (best == nullptr) {
            best = &m;
            continue;
        }
        // most headroom over the estimated context first
        const int64_t headroom  = (int64_t) m.n_ctx - est_ctx;
        const int64_t best_head = (int64_t) best->n_ctx - est_ctx;
        if (headroom > best_head) {
            best = &m;
        }
    }
    return best ? best->name : "";
}

} // namespace pool
