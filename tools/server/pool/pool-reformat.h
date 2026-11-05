#pragma once

// Cross-template chat reformat (WS-17, D-M1/D-M2): the glue that turns the
// origin client's messages into a message structure the SERVING member's
// template renders faithfully. The canonical form is the engine's own
// common_chat_msg list (common/chat.h); this file only glues its public API.
// The same-template case is a byte-identical skip, never a re-serialize.

#include <string>
#include <vector>

#include "chat.h"
#include "json.h"

namespace pool {

// SHA-256 hex of the template source (the empty source hashes too: two
// members without a built-in template are the same template family)
std::string pool_template_hash(const std::string & tmpl_src);

// D-M2 keep heuristic (MVP): the serving template keeps reasoning_content
// only when its source references reasoning/thinking. substring check.
bool pool_template_references_reasoning(const std::string & tmpl_src);

// Reformat the pulled body for the serving member. returns the body to serve,
// or "" on failure (the caller posts a reformat_failed terminal and never
// serves the body):
//  - origin hash empty (origin had no local member) or equal to the serving
//    hash: the body is returned byte-identical (the same-template skip / the
//    pass-through), no re-parse, no re-serialize
//  - otherwise the messages are re-serialized under the serving template
//    (reasoning_content dropped unless the template references it); the rest
//    of the body (model, stream, tools, sampling params, the existing clamp)
//    is untouched
// With LLAMA_POOL_DEBUG_RT=1 the pipeline also asserts the a->b->a identity
// and returns "" when the round trip is broken.
std::string pool_reformat_body(const std::string & body,
                               const std::string & origin_template_hash,
                               const std::string & serving_template_hash,
                               const std::string & serving_tmpl_src);

// The a->b->a identity (D-M2): re-parsing the re-serialized list must come
// back canonical-equal to it. canonicalize = collapse whitespace-only
// differences, normalize tool-call ids (template-specific, must not break the
// identity). canonical_in is the list as it will be serialized: the origin
// parse with the parts the serving template does not keep already dropped, so
// a dropped reasoning part on a non-reasoning B is the expected form.
bool pool_roundtrip_ok(const std::vector<common_chat_msg> & canonical_in);

// canonical equality used by the corpus: whitespace-insensitive on text and
// reasoning, tool-call ids normalized (the re-serializer may re-id them),
// role / content parts / tool calls compared structurally
bool pool_canon_equal(const std::vector<common_chat_msg> & a, const std::vector<common_chat_msg> & b);

} // namespace pool
