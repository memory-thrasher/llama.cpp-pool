#include "pool-reformat.h"

#include "hash/hash.h"
#include "log.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace pool {

std::string pool_template_hash(const std::string & tmpl_src) {
    return hash_sha256_hex(tmpl_src.data(), tmpl_src.size());
}

bool pool_template_references_reasoning(const std::string & tmpl_src) {
    // the MVP heuristic: the template source mentions reasoning or thinking
    if (tmpl_src.find("reasoning") != std::string::npos) {
        return true;
    }
    if (tmpl_src.find("thinking") != std::string::npos) {
        return true;
    }
    return false;
}

namespace {

// collapse whitespace-only differences: trim the ends, join internal runs to
// one space. for canonical comparison only, never for the served bytes
std::string canon_ws(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    bool in_space = true;
    for (const char c : s) {
        if (std::isspace(static_cast<unsigned char>(c)) != 0) {
            in_space = true;
            continue;
        }
        if (!in_space && !out.empty()) {
            out.push_back(' ');
        }
        out.push_back(c);
        in_space = false;
    }
    if (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

// a normalized copy for identity checks: whitespace collapsed, tool-call ids
// emptied (ids are template-specific and must not break the identity)
common_chat_msg canon_msg(const common_chat_msg & m) {
    common_chat_msg c = m;
    c.content             = canon_ws(m.content);
    c.reasoning_content   = canon_ws(m.reasoning_content);
    for (auto & part : c.content_parts) {
        part.text = canon_ws(part.text);
    }
    for (auto & tc : c.tool_calls) {
        tc.id.clear();
    }
    return c;
}

} // namespace

bool pool_canon_equal(const std::vector<common_chat_msg> & a, const std::vector<common_chat_msg> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (canon_msg(a[i]) != canon_msg(b[i])) {
            return false;
        }
    }
    return true;
}

bool pool_roundtrip_ok(const std::vector<common_chat_msg> & canonical_in) {
    common_json serialized;
    try {
        serialized = common_chat_msgs_to_json_oaicompat(canonical_in);
    } catch (const std::exception & e) {
        LOG_ERR("pool reformat: round trip serialize failed: %s\n", e.what());
        return false;
    }
    std::vector<common_chat_msg> reparsed;
    try {
        reparsed = common_chat_msgs_parse_oaicompat(serialized);
    } catch (const std::exception & e) {
        LOG_ERR("pool reformat: round trip re-parse failed: %s\n", e.what());
        return false;
    }
    if (reparsed.size() != canonical_in.size()) {
        LOG_ERR("pool reformat: round trip message count %zu != %zu\n",
                reparsed.size(), canonical_in.size());
        return false;
    }
    for (size_t i = 0; i < canonical_in.size(); i++) {
        if (canon_msg(canonical_in[i]) != canon_msg(reparsed[i])) {
            LOG_ERR("pool reformat: round trip broke at message %zu\n", i);
            return false;
        }
    }
    return pool_canon_equal(canonical_in, reparsed);
}

std::string pool_reformat_body(const std::string & body,
                               const std::string & origin_template_hash,
                               const std::string & serving_template_hash,
                               const std::string & serving_tmpl_src) {
    // the same template (or the origin had no local member): the body passes
    // byte-identical, no re-parse, no re-serialize. this is the common case
    if (origin_template_hash.empty() || origin_template_hash == serving_template_hash) {
        return body;
    }

    common_json obj;
    try {
        obj = common_json::parse(body);
    } catch (const std::exception & e) {
        LOG_ERR("pool reformat: body does not parse: %s\n", e.what());
        return "";
    }
    if (!obj.is_object() || !obj.contains("messages") || !obj["messages"].is_array()) {
        LOG_ERR("pool reformat: body has no messages array\n");
        return "";
    }

    std::vector<common_chat_msg> canonical;
    try {
        canonical = common_chat_msgs_parse_oaicompat(obj["messages"]);
    } catch (const std::exception & e) {
        LOG_ERR("pool reformat: message parse failed: %s\n", e.what());
        return "";
    }

    const bool keep_reasoning = pool_template_references_reasoning(serving_tmpl_src);
    if (!keep_reasoning) {
        // dropped with a one-line debug log (count only, never the content)
        size_t dropped = 0;
        for (auto & msg : canonical) {
            if (!msg.reasoning_content.empty()) {
                dropped++;
                msg.reasoning_content.clear();
            }
        }
        if (dropped > 0) {
            LOG_DBG("pool reformat: dropped reasoning_content on %zu message(s), the serving template has no reasoning\n", dropped);
        }
    }

    common_json messages;
    try {
        messages = common_chat_msgs_to_json_oaicompat(canonical);
    } catch (const std::exception & e) {
        LOG_ERR("pool reformat: message serialize failed: %s\n", e.what());
        return "";
    }
    obj["messages"] = messages;
    const std::string out = obj.dump();

    // D-M2: under the debug flag, assert the a->b->a identity before serving
    if (std::getenv("LLAMA_POOL_DEBUG_RT") != nullptr) {
        if (!pool_roundtrip_ok(canonical)) {
            LOG_ERR("pool reformat: round trip identity broken; refusing to serve the body\n");
            return "";
        }
    }

    return out;
}

} // namespace pool
