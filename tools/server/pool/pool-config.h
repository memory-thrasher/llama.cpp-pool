#pragma once

#include "common.h"

#include <string>

namespace pool {

// pool control channel configuration. empty server = pool off, zero behavior change
struct pool_config {
    std::string server;    // coordinator URL, e.g. https://pool.example.com
    std::string name;      // router display name
    std::string token;     // member token, from the LLAMA_POOL_TOKEN env var; never logged
    std::string router_id; // stable UUIDv4, persisted at id_path
    std::string id_path;   // <models_dir>/.pool/router-id

    bool enabled() const { return !server.empty(); }
};

// build the config from CLI params and env vars.
// returns an empty string on success, a startup error message otherwise
std::string pool_config_init(const common_params & params, pool_config & out);

// a fresh random UUIDv4 (lowercase), e.g. the intake request_id
std::string new_uuid();
}
