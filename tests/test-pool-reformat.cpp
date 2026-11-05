#include "testing.h"
#include "chat.h"
#include "json.h"
#include "pool/pool-reformat.h"

#include <string>
#include <vector>

using json = common_json;

// =============================================================================
// WS-17 corpus: cross-template chat reformat (D-M1/D-M2). The templates below
// are the real template texts of the three families (Qwen3-style, Llama-3.1-
// style, Mistral-style) embedded verbatim from the model files; no model is
// loaded and no GGUF is needed. The pipeline only re-serializes the message
// structure, so the templates are used as identity strings (hash) and for the
// reasoning-keep heuristic (substring check), never rendered.
// =============================================================================

static const char * TPL_QWEN3 =
R"TPL({%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = message.content.split('</think>')[-1].lstrip('\n') %}
                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %})TPL"
;

static const char * TPL_LLAMA31 =
R"TPL({{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
                {%- endfor %}
            {{- ")" }}
        {%- else  %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {#- This means we're in ipython mode #}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
)TPL"
;

static const char * TPL_MISTRAL =
R"TPL({%- if messages[0]["role"] == "system" %}
    {%- set system_message = messages[0]["content"] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}
{%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}

{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}
{%- set ns = namespace() %}
{%- set ns.index = 0 %}
{%- for message in loop_messages %}
    {%- if not (message.role == "tool" or message.role == "tool_results" or (message.tool_calls is defined and message.tool_calls is not none)) %}
        {%- if (message["role"] == "user") != (ns.index % 2 == 0) %}
            {{- raise_exception("After the optional system message, conversation roles must alternate user/assistant/user/assistant/...") }}
        {%- endif %}
        {%- set ns.index = ns.index + 1 %}
    {%- endif %}
{%- endfor %}

{{- bos_token }}
{%- for message in loop_messages %}
    {%- if message["role"] == "user" %}
        {%- if tools is not none and (message == user_messages[-1]) %}
            {{- "[AVAILABLE_TOOLS][" }}
            {%- for tool in tools %}
                {%- set tool = tool.function %}
                {{- '{"type": "function", "function": {' }}
                {%- for key, val in tool.items() if key != "return" %}
                    {%- if val is string %}
                        {{- '"' + key + '": "' + val + '"' }}
                    {%- else %}
                        {{- '"' + key + '": ' + val|tojson }}
                    {%- endif %}
                    {%- if not loop.last %}
                        {{- ", " }}
                    {%- endif %}
                {%- endfor %}
                {{- "}}" }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- else %}
                    {{- "]" }}
                {%- endif %}
            {%- endfor %}
            {{- "[/AVAILABLE_TOOLS]" }}
            {%- endif %}
        {%- if loop.last and system_message is defined %}
            {{- "[INST]" + system_message + "\n\n" + message["content"] + "[/INST]" }}
        {%- else %}
            {{- "[INST]" + message["content"] + "[/INST]" }}
        {%- endif %}
    {%- elif (message.tool_calls is defined and message.tool_calls is not none) %}
        {{- "[TOOL_CALLS][" }}
        {%- for tool_call in message.tool_calls %}
            {%- set out = tool_call.function|tojson %}
            {{- out[:-1] }}
            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}
                {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
            {%- endif %}
            {{- ', "id": "' + tool_call.id + '"}' }}
            {%- if not loop.last %}
                {{- ", " }}
            {%- else %}
                {{- "]" + eos_token }}
            {%- endif %}
        {%- endfor %}
    {%- elif message["role"] == "assistant" %}
        {{- message["content"] + eos_token}}
    {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}
        {%- if message.content is defined and message.content.content is defined %}
            {%- set content = message.content.content %}
        {%- else %}
            {%- set content = message.content %}
        {%- endif %}
        {{- '[TOOL_RESULTS]{"content": ' + content|string + ", " }}
        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}
            {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
        {%- endif %}
        {{- '"call_id": "' + message.tool_call_id + '"}[/TOOL_RESULTS]' }}
    {%- else %}
        {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}
    {%- endif %}
{%- endfor %}
)TPL"
;


struct tmpl_entry { const char * name; const char * src; };
static const std::vector<tmpl_entry> TEMPLATES = {
    {"qwen3",   TPL_QWEN3},
    {"llama31", TPL_LLAMA31},
    {"mistral", TPL_MISTRAL},
};

struct case_entry { const char * id; const char * messages; };
// the corpus: plain multi-turn; assistant tool_call + role:tool result;
// reasoning_content present; image content part (typed text parts standing in
// for the image slot); and a mix of the above
static const std::vector<case_entry> CORPUS = {
    {"plain-multi-turn",
     R"MSG([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you today?"},
        {"role": "assistant", "content": "I am doing well, thank you for asking."},
        {"role": "user", "content": "Tell me a short joke about a cat."}
     ])MSG"},
    {"tool-call+result",
     R"MSG([
        {"role": "system", "content": "You have access to tools."},
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "id": "call-1", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}
        ]},
        {"role": "tool", "tool_call_id": "call-1", "content": "{\"temp\": 18}"}
     ])MSG"},
    {"reasoning",
     R"MSG([
        {"role": "user", "content": "Solve 2 + 2 and show your working."},
        {"role": "assistant", "content": "The answer is 4.", "reasoning_content": "Let me think. Two plus two is four."}
     ])MSG"},
    {"image-content-part",
     R"MSG([
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image: "},
            {"type": "text", "text": "a cat sitting on a windowsill"}
        ]}
     ])MSG"},
    {"mixed",
     R"MSG([
        {"role": "system", "content": "You are a helpful assistant with tools."},
        {"role": "user", "content": "What is 3*7, and what is the weather in Tokyo?"},
        {"role": "assistant", "content": "Let me compute that and check the weather.",
         "reasoning_content": "3*7 is 21. I need the weather tool for Tokyo.",
         "tool_calls": [
            {"type": "function", "id": "call-x", "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}}
        ]},
        {"role": "tool", "tool_call_id": "call-x", "content": "{\"temp\": 22}"},
        {"role": "assistant", "content": "3*7 is 21, and it is 22 degrees in Tokyo."}
     ])MSG"},
};

namespace {

// the canonical list as the serving template keeps it: A with reasoning_content
// dropped when B does not reference reasoning (the expected drop)
std::vector<common_chat_msg> kept_under(const std::vector<common_chat_msg> & a, const char * b_src) {
    std::vector<common_chat_msg> out = a;
    if (!pool::pool_template_references_reasoning(b_src)) {
        for (auto & m : out) {
            m.reasoning_content.clear();
        }
    }
    return out;
}

// one corpus case under a (A, B) pair: parse under A, serialize under B's
// rules, parse under B, assert canonical equality of the kept parts and that a
// reasoning part on a non-reasoning B is dropped (expected). returns false on
// a broken round trip
bool run_case(testing & t, const case_entry & c, const tmpl_entry & a, const tmpl_entry & b) {
    std::vector<common_chat_msg> canonical_a;
    try {
        canonical_a = common_chat_msgs_parse_oaicompat(json::parse(c.messages));
    } catch (const std::exception & e) {
        t.assert_true(std::string(a.name) + " -> " + b.name + " / " + c.id + ": parse failed: " + e.what(), false);
        return false;
    }
    auto kept = kept_under(canonical_a, b.src);
    json serialized = common_chat_msgs_to_json_oaicompat(kept);
    std::vector<common_chat_msg> canonical_b;
    try {
        canonical_b = common_chat_msgs_parse_oaicompat(serialized);
    } catch (const std::exception & e) {
        t.assert_true(std::string(a.name) + " -> " + b.name + " / " + c.id + ": re-parse failed: " + e.what(), false);
        return false;
    }
    bool ok = pool::pool_canon_equal(kept, canonical_b);
    t.assert_true(std::string(a.name) + " -> " + b.name + " / " + c.id + ": canonical equality after reformat", ok);

    // the reasoning drop is expected on a non-reasoning B, kept on a reasoning B
    bool a_has_reasoning = false;
    for (const auto & m : canonical_a) {
        if (!m.reasoning_content.empty()) {
            a_has_reasoning = true;
        }
    }
    bool b_has_reasoning = false;
    for (const auto & m : canonical_b) {
        if (!m.reasoning_content.empty()) {
            b_has_reasoning = true;
        }
    }
    const bool b_refs = pool::pool_template_references_reasoning(b.src);
    if (a_has_reasoning && !b_refs) {
        t.assert_true(std::string(a.name) + " -> " + b.name + " / " + c.id + ": reasoning dropped (expected)", !b_has_reasoning);
    } else if (a_has_reasoning && b_refs) {
        t.assert_true(std::string(a.name) + " -> " + b.name + " / " + c.id + ": reasoning kept", b_has_reasoning);
    }
    return ok;
}

// the full pipeline (as the serving router runs it): same template is a
// byte-identical skip; a different template re-serializes
void test_pipeline_skip_and_reformat(testing & t) {
    const std::string body =
        R"({"model":"pool-md","max_tokens":16,"stream":false,"messages":[{"role":"user","content":"hello"}]})";

    // same-template skip: origin hash == serving hash -> byte-identical
    const std::string h = pool::pool_template_hash(TPL_QWEN3);
    t.assert_equal(64u, h.size());
    const std::string same = pool::pool_reformat_body(body, h, h, TPL_QWEN3);
    t.assert_true("same-template skip is byte-identical", same == body);

    // pass-through: the origin had no local member (null hash) -> byte-identical
    const std::string passthrough = pool::pool_reformat_body(body, "", h, TPL_QWEN3);
    t.assert_true("origin-null pass-through is byte-identical", passthrough == body);

    // different template: the messages are re-serialized (reasoning dropped on
    // a non-reasoning B). here the body has no reasoning, so the structure
    // stands; the hash differs so the pipeline runs (not the skip)
    const std::string h_other = pool::pool_template_hash(TPL_MISTRAL);
    t.assert_true("distinct templates hash distinct", h != h_other);
    const std::string reformatted = pool::pool_reformat_body(body, h, h_other, TPL_MISTRAL);
    t.assert_true("cross-template pipeline produces a body", !reformatted.empty());
    // the model / sampling params are untouched; only the messages move
    json out = json::parse(reformatted);
    t.assert_equal(std::string("pool-md"), out["model"].get<std::string>());
    t.assert_equal(16, out["max_tokens"].get<int>());

    // the a->b->a identity holds for the re-serialized list
    std::vector<common_chat_msg> kept = common_chat_msgs_parse_oaicompat(out["messages"]);
    t.assert_true("cross-template a->b->a identity", pool::pool_roundtrip_ok(kept));
}

// the reasoning keep heuristic result per corpus template (design note)
void test_reasoning_heuristic(testing & t) {
    const bool qwen   = pool::pool_template_references_reasoning(TPL_QWEN3);
    const bool llama  = pool::pool_template_references_reasoning(TPL_LLAMA31);
    const bool mistral = pool::pool_template_references_reasoning(TPL_MISTRAL);
    t.log("reasoning heuristic: qwen3=" + std::string(qwen ? "keep" : "drop") +
          " llama31=" + std::string(llama ? "keep" : "drop") +
          " mistral=" + std::string(mistral ? "keep" : "drop"));
    // Qwen3 references reasoning_content; Llama-3.1 and Mistral do not
    t.assert_true("qwen3 template references reasoning", qwen);
    t.assert_true("llama31 template does not reference reasoning", !llama);
    t.assert_true("mistral template does not reference reasoning", !mistral);
}


// a quiet re-run for the per-template report (no assertions, bool only)
bool run_case_quiet(const tmpl_entry & /* a */, const tmpl_entry & b, const case_entry & c) {
    try {
        std::vector<common_chat_msg> canonical_a = common_chat_msgs_parse_oaicompat(json::parse(c.messages));
        auto kept = kept_under(canonical_a, b.src);
        json serialized = common_chat_msgs_to_json_oaicompat(kept);
        std::vector<common_chat_msg> canonical_b = common_chat_msgs_parse_oaicompat(serialized);
        return pool::pool_canon_equal(kept, canonical_b);
    } catch (const std::exception &) {
        return false;
    }
}

} // namespace

int main() {
    testing t(std::cout);
    t.verbose = true;

    t.test("reasoning keep heuristic per template", test_reasoning_heuristic);
    t.test("pipeline: same-template skip + cross-template reformat", test_pipeline_skip_and_reformat);

    // the (A, B) corpus: every ordered pair of templates, every case
    int pass = 0;
    int total = 0;
    for (const auto & a : TEMPLATES) {
        for (const auto & b : TEMPLATES) {
            for (const auto & c : CORPUS) {
                total++;
                if (run_case(t, c, a, b)) {
                    pass++;
                }
            }
        }
    }
    // per-template pass report (a template is a pass when every case it is a
    // serving target for holds)
    for (const auto & b : TEMPLATES) {
        int b_pass = 0;
        for (const auto & a : TEMPLATES) {
            for (const auto & c : CORPUS) {
                if (run_case_quiet(a, b, c)) {
                    b_pass++;
                }
            }
        }
        t.log(std::string("report: ") + b.name + " (serving) " + std::to_string(b_pass) + "/" + std::to_string(3 * CORPUS.size()) + " pass");
    }
    t.assert_true("corpus: all " + std::to_string(total) + " (A,B,case) round trips hold", pass == total);

    return t.summary();
}
