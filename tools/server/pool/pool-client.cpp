#include "pool-client.h"

#include "common.h"
#include "log.h"
#include "build-info.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <deque>
#include <memory>
#include <thread>

namespace pool {

namespace {

const int PROTO_VERSION = 1;

// the coordinator writes a heartbeat comment roughly every 25 s, so the
// read timeout doubles as the heartbeat silence detector
const int SSE_READ_TIMEOUT_S = 60;
// long-poll: the coordinator holds the response for at most 55 s
const int POLL_READ_TIMEOUT_S = 60;
const int CONNECT_TIMEOUT_S = 10;
// event POSTs must not block the caller long
const int EVENT_TIMEOUT_S = 5;
const int BACKOFF_MAX_S = 60;
// a stream that closes almost right after connecting is treated as a failed
// round, so the backoff keeps growing instead of spinning
const int MIN_ROUND_S = 2;
const int SSE_FAILURES_MAX = 3;
// bound of the outbound event queue; the oldest event is dropped on overflow
const size_t EVENT_QUEUE_MAX = 32;

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

std::string json_str(const json & obj, const char * key) {
    if (obj.is_object() && obj.contains(key) && obj[key].is_string()) {
        return obj[key].get<std::string>();
    }
    return "";
}

} // namespace

//
// pool_event_bus
//

pool_event_bus::token_t pool_event_bus::subscribe(handler_t handler) {
    std::lock_guard<std::mutex> lk(mutex);
    const token_t token = next_token++;
    handlers.push_back({token, std::move(handler)});
    return token;
}

void pool_event_bus::unsubscribe(token_t token) {
    std::lock_guard<std::mutex> lk(mutex);
    handlers.erase(std::remove_if(handlers.begin(), handlers.end(),
        [token](const std::pair<token_t, handler_t> & h) { return h.first == token; }),
        handlers.end());
}

void pool_event_bus::post(const json & envelope) {
    std::vector<std::pair<token_t, handler_t>> copy;
    {
        std::lock_guard<std::mutex> lk(mutex);
        copy = handlers;
    }
    for (auto & h : copy) {
        h.second(envelope);
    }
}

//
// pool_client::state
//

struct pool_client::state {
    pool_config cfg;
    std::function<json()> capacity_snapshot;
    pool_event_bus * bus = nullptr;

    std::thread subscribe_thread;
    std::thread sender_thread;
    std::atomic<bool> running{false};
    std::atomic<bool> stopping{false};

    struct event_t {
        std::string type;
        json payload;
    };
    std::mutex ev_mutex;
    std::condition_variable ev_cv;
    std::deque<event_t> events;

    // the client currently holding the subscribe stream, so stop() can wake a blocked read
    std::mutex cli_mutex;
    std::shared_ptr<httplib::Client> current_cli;

    //
    // outbound events
    //

    void queue_event(const std::string & type, const json & payload) {
        std::lock_guard<std::mutex> lk(ev_mutex);
        if (events.size() >= EVENT_QUEUE_MAX) {
            events.pop_front(); // drop the oldest
        }
        events.push_back({type, std::move(payload)});
        ev_cv.notify_one();
    }

    // metadata only, no payload (anti-memory)
    bool post_event(const std::string & type, const json & payload) {
        const json env = {
            {"type", type},
            {"payload", payload},
        };
        httplib::Client cli(cfg.server);
        cli.set_connection_timeout(CONNECT_TIMEOUT_S);
        cli.set_read_timeout(EVENT_TIMEOUT_S);
        cli.set_write_timeout(EVENT_TIMEOUT_S);
        const httplib::Headers headers = {
            {"Authorization", "Bearer " + cfg.token},
            {"X-Pool-Proto", std::to_string(PROTO_VERSION)},
        };
        auto res = cli.Post("/pool/events", headers, env.dump(), "application/json");
        if (!res) {
            LOG_WRN("pool: event post failed: type=%s (no response)\n", type.c_str());
            return false;
        }
        if (res->status != 200) {
            LOG_WRN("pool: event post failed: type=%s status=%d\n", type.c_str(), res->status);
            return false;
        }
        return true;
    }

    void sender_loop() {
        while (true) {
            event_t ev;
            {
                std::unique_lock<std::mutex> lk(ev_mutex);
                ev_cv.wait(lk, [this]() { return stopping || !events.empty(); });
                if (stopping) {
                    return; // dropping the rest, the router is going away
                }
                ev = std::move(events.front());
                events.pop_front();
            }
            post_event(ev.type, ev.payload);
        }
    }

    //
    // capacity snapshot
    //

    json snapshot() {
        json models;
        try {
            models = capacity_snapshot();
        } catch (const std::exception & e) {
            LOG_WRN("pool: capacity snapshot failed: %s\n", e.what());
        }
        if (!models.is_array()) {
            models = json::array();
        }
        return models;
    }

    void on_capacity_change() {
        if (!running.load()) {
            return;
        }
        const json payload = {
            {"router_id", cfg.router_id},
            {"models", snapshot()},
            {"updated_at", rfc3339_now()},
        };
        queue_event("capacity", std::move(payload));
    }

    json handshake_payload() {
        return json {
            {"proto", PROTO_VERSION},
            {"router_id", cfg.router_id},
            {"router_name", cfg.name},
            {"software", std::string(llama_build_info())},
            {"models", snapshot()},
        };
    }

    //
    // subscribe loop
    //

    void sleep_interruptible(int seconds) {
        for (int i = 0; i < seconds && !stopping.load(); i++) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    void subscribe_loop() {
        // LLAMA_POOL_LINK_MODE=sse|poll, default sse; after consecutive SSE
        // failures we stay on long-poll for the rest of the process lifetime
        bool use_poll = common_get_env("LLAMA_POOL_LINK_MODE") == "poll";
        // the handshake goes out on (re)connect, before any other event
        bool announced = false;
        int sse_failures = 0;
        int backoff_s = 1;

        while (!stopping.load()) {
            const bool ok = use_poll ? poll_round(announced) : sse_round(announced);
            if (stopping.load()) {
                break;
            }
            if (use_poll) {
                // a successful round already waited on the coordinator side
                backoff_s = ok ? 1 : std::min(backoff_s * 2, BACKOFF_MAX_S);
            } else if (ok) {
                sse_failures = 0;
                backoff_s = 1;
            } else {
                sse_failures++;
                backoff_s = std::min(backoff_s * 2, BACKOFF_MAX_S);
                if (sse_failures >= SSE_FAILURES_MAX) {
                    use_poll = true;
                    LOG_WRN("pool: %d consecutive SSE failures, switching to long-poll mode\n", sse_failures);
                }
            }
            if (ok) {
                announced = true;
            } else {
                announced = false;
            }
            sleep_interruptible(backoff_s);
        }
    }

    bool sse_round(bool & announced) {
        auto cli = std::make_shared<httplib::Client>(cfg.server);
        cli->set_connection_timeout(CONNECT_TIMEOUT_S);
        cli->set_read_timeout(SSE_READ_TIMEOUT_S);
        cli->set_write_timeout(CONNECT_TIMEOUT_S);
        {
            std::lock_guard<std::mutex> lk(cli_mutex);
            current_cli = cli;
        }

        bool connected = false;
        std::string buf;   // partial line
        std::string data;  // accumulated data of the current event

        httplib::Headers headers = {
            {"Accept", "text/event-stream"},
            {"Authorization", "Bearer " + cfg.token},
            {"X-Pool-Proto", std::to_string(PROTO_VERSION)},
        };

        const auto t0 = std::chrono::steady_clock::now();
        auto res = cli->Get("/pool/subscribe", headers,
            [this, &connected, &announced](const httplib::Response & r) -> bool {
                if (r.status != 200) {
                    LOG_WRN("pool: subscribe failed: status=%d\n", r.status);
                    return false;
                }
                connected = true;
                // first message after (re)connect, before any other
                if (!announced) {
                    queue_event("handshake", handshake_payload());
                    announced = true;
                }
                return true;
            },
            [&buf, &data, this](const char * d, size_t len) -> bool {
                return sse_feed(d, len, buf, data);
            });
        const auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0).count();

        {
            std::lock_guard<std::mutex> lk(cli_mutex);
            current_cli.reset();
        }

        if (!res) {
            if (!connected) {
                LOG_WRN("pool: subscribe failed: cannot reach %s\n", cfg.server.c_str());
            }
            return false;
        }
        if (res->status != 200) {
            return false; // the response handler already logged it
        }
        // the shutdown closes the stream itself, no point in reporting it
        if (stopping.load()) {
            return true;
        }
        if (elapsed_s < MIN_ROUND_S) {
            LOG_INF("pool: subscribe closed after %llds, treating as failed\n", (long long) elapsed_s);
            return false;
        }
        // stream ended: the coordinator closed it, or the heartbeat went silent
        LOG_INF("pool: subscribe stream closed, reconnecting\n");
        return true;
    }

    // feed raw bytes of the SSE stream, dispatch each completed event
    bool sse_feed(const char * data, size_t len, std::string & buf, std::string & event_data) {
        buf.append(data, len);
        size_t start = 0;
        while (true) {
            const size_t nl = buf.find('\n', start);
            if (nl == std::string::npos) {
                break;
            }
            std::string line = buf.substr(start, nl - start);
            start = nl + 1;
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                // the end of the event
                if (!event_data.empty()) {
                    dispatch_envelope(event_data);
                    event_data.clear();
                }
            } else if (line[0] == ':') {
                // comment; the coordinator sends ": hb <unix_ms>" heartbeats
            } else if (line.rfind("data:", 0) == 0) {
                std::string payload = line.substr(5);
                if (!payload.empty() && payload[0] == ' ') {
                    payload.erase(0, 1);
                }
                if (!event_data.empty()) {
                    event_data += "\n";
                }
                event_data += payload;
            }
            // event:, id: and retry: fields are not used by the protocol
        }
        buf.erase(0, start);
        return true;
    }

    void dispatch_envelope(const std::string & body) {
        json env;
        try {
            env = json::parse(body);
        } catch (const std::exception &) {
            LOG_WRN("%s", "pool: dropping an event with an invalid body\n");
            return;
        }
        if (!env.is_object()) {
            return;
        }
        bus->post(env);
    }

    bool poll_round(bool & announced) {
        httplib::Client cli(cfg.server);
        cli.set_connection_timeout(CONNECT_TIMEOUT_S);
        cli.set_read_timeout(POLL_READ_TIMEOUT_S);
        cli.set_write_timeout(CONNECT_TIMEOUT_S);
        httplib::Headers headers = {
            {"Accept", "application/json"},
            {"Authorization", "Bearer " + cfg.token},
            {"X-Pool-Proto", std::to_string(PROTO_VERSION)},
        };
        auto res = cli.Get("/pool/subscribe", headers);
        if (!res) {
            LOG_WRN("pool: long-poll failed: cannot reach %s\n", cfg.server.c_str());
            return false;
        }
        if (res->status != 200) {
            LOG_WRN("pool: long-poll failed: status=%d\n", res->status);
            return false;
        }
        // first message after (re)connect, before any other
        if (!announced) {
            queue_event("handshake", handshake_payload());
            announced = true;
        }
        json offers;
        try {
            offers = json::parse(res->body);
        } catch (const std::exception &) {
            LOG_WRN("%s", "pool: long-poll returned an invalid body\n");
            return true;
        }
        if (!offers.is_array()) {
            LOG_WRN("%s", "pool: long-poll returned an unexpected body\n");
            return true;
        }
        for (size_t i = 0; i < offers.size(); i++) {
            if (offers[i].is_object()) {
                bus->post(offers[i]);
            }
        }
        return true;
    }

    void start() {
        running = true;
        bool sender_started = false;
        try {
            sender_thread = std::thread([this]() { sender_loop(); });
            sender_started = true;
            subscribe_thread = std::thread([this]() { subscribe_loop(); });
        } catch (const std::exception &) {
            stopping = true;
            if (sender_started && sender_thread.joinable()) {
                sender_thread.join();
            }
            if (subscribe_thread.joinable()) {
                subscribe_thread.join();
            }
            throw;
        }
    }

    void stop() {
        stopping = true;
        // wake a read blocked on the subscribe stream
        std::shared_ptr<httplib::Client> cli;
        {
            std::lock_guard<std::mutex> lk(cli_mutex);
            cli = current_cli;
        }
        if (cli) {
            cli->stop();
        }
        ev_cv.notify_all();
        if (sender_thread.joinable()) {
            sender_thread.join();
        }
        if (subscribe_thread.joinable()) {
            subscribe_thread.join();
        }
    }
};

//
// pool_client
//

pool_client::pool_client() = default;

pool_client::~pool_client() {
    stop();
}

bool pool_client::start(const pool_config & cfg, std::function<json()> capacity_snapshot, std::string & err) {
    if (st) {
        err = "pool client is already running";
        return false;
    }
    st = std::make_unique<state>();
    st->cfg = cfg;
    st->capacity_snapshot = std::move(capacity_snapshot);
    st->bus = &bus_;

    // WS-03 marker handler for work_offer; WS-05 replaces it with the real dispatch
    bus_.subscribe([](const json & env) {
        if (json_str(env, "type") != "work_offer") {
            return;
        }
        std::string job_id, tag;
        if (env.contains("payload") && env["payload"].is_object()) {
            job_id = json_str(env["payload"], "job_id");
            tag    = json_str(env["payload"], "tag");
        }
        LOG_INF("pool: work_offer job_id=%s tag=%s\n", job_id.c_str(), tag.c_str());
    });

    try {
        st->start();
    } catch (const std::exception & e) {
        st.reset();
        err = string_format("failed to start the pool threads: %s", e.what());
        return false;
    }
    LOG_INF("pool: control channel started: server=%s name=%s router_id=%s\n",
            cfg.server.c_str(), cfg.name.c_str(), cfg.router_id.c_str());
    return true;
}

void pool_client::stop() {
    state * st = this->st.get();
    if (!st || !st->running.exchange(false)) {
        return; // not started, or already stopped
    }
    st->stop();
    this->st.reset();
    LOG_INF("%s", "pool: stopped\n");
}

void pool_client::on_capacity_change() {
    if (st) {
        st->on_capacity_change();
    }
}

const pool_config * pool_client::config() const {
    return st ? &st->cfg : nullptr;
}

pool_client & pool_instance() {
    static pool_client inst;
    return inst;
}

void pool_on_capacity_change() {
    pool_instance().on_capacity_change();
}

} // namespace pool
