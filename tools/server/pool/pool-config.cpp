#include "pool-config.h"

#include "common.h"

#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace pool {

namespace {

// generate a random UUIDv4
std::string generate_router_id() {
    std::random_device rd;
    // random_device() returns 32 bits, combine two calls per 64-bit half
    const uint64_t a = ((uint64_t) rd() << 32) | (uint64_t) rd();
    const uint64_t b = ((uint64_t) rd() << 32) | (uint64_t) rd();
    const uint32_t t_low = (uint32_t) (a >> 32);
    const uint16_t t_mid = (uint16_t) ((a >> 16) & 0xFFFF);
    const uint16_t t_hi  = (uint16_t) ((a & 0x0FFF) | 0x4000); // version 4
    const uint16_t c_seq = (uint16_t) (((b >> 48) & 0x3FFF) | 0x8000); // variant 1
    const uint64_t node  = b & 0xFFFFFFFFFFFFULL;
    char buf[37];
    snprintf(buf, sizeof(buf), "%08x-%04x-%04x-%04x-%012llx",
             t_low, t_mid, t_hi, c_seq, (unsigned long long) node);
    return buf;
}

bool is_valid_router_id(const std::string & id) {
    if (id.size() != 36) {
        return false;
    }
    for (size_t i = 0; i < id.size(); i++) {
        const bool is_dash = (i == 8 || i == 13 || i == 18 || i == 23);
        const bool ok = is_dash ? id[i] == '-' : std::isxdigit((unsigned char) id[i]);
        if (!ok) {
            return false;
        }
    }
    return id[14] == '4';
}

std::string default_name() {
    char buf[256] = {0};
#ifdef _WIN32
    DWORD n = sizeof(buf);
    if (GetComputerNameA(buf, &n)) {
        return buf;
    }
#else
    if (gethostname(buf, sizeof(buf)) == 0) {
        return buf;
    }
#endif
    return "llama-router";
}

// read the persisted router id; empty on missing or unreadable file
std::string read_router_id(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        return "";
    }
    std::string id;
    std::getline(f, id);
    // trim whitespace, the file is one line
    while (!id.empty() && std::isspace((unsigned char) id.front())) id.erase(id.begin());
    while (!id.empty() && std::isspace((unsigned char) id.back()))  id.pop_back();
    return id;
}

// load the persisted router id or generate and persist a new one
std::string load_or_create_router_id(const std::string & path) {
    std::string id = read_router_id(path);
    if (is_valid_router_id(id)) {
        return id;
    }
    id = generate_router_id();
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    std::ofstream f(path, std::ios::trunc);
    if (!f) {
        return ""; // caller reports the failure
    }
    f << id;
    return id;
}

} // namespace

std::string new_uuid() {
    return generate_router_id();
}

std::string pool_config_init(const common_params & params, pool_config & out) {
    out = pool_config{};
    if (params.pool_server.empty()) {
        return ""; // pool off
    }

    out.server = params.pool_server;
    while (out.server.size() > 1 && out.server.back() == '/') {
        out.server.pop_back();
    }
    if (out.server.rfind("http://", 0) != 0 && out.server.rfind("https://", 0) != 0) {
        return "invalid --pool-server value: the URL must start with http:// or https://";
    }

    out.name = params.pool_name.empty() ? default_name() : params.pool_name;
    if (out.name.size() > 64) {
        out.name.resize(64);
    }

    out.token = common_get_env("LLAMA_POOL_TOKEN");
    if (out.token.empty()) {
        return "--pool-server is set but the LLAMA_POOL_TOKEN env var is not set";
    }

    if (params.models_dir.empty()) {
        return "--pool-server requires --models-dir, the router id is persisted in <models_dir>/.pool/router-id";
    }
    out.id_path = params.models_dir + "/.pool/router-id";

    out.router_id = load_or_create_router_id(out.id_path);
    if (out.router_id.empty()) {
        return "failed to read or create the router id at " + out.id_path;
    }
    return "";
}

} // namespace pool
