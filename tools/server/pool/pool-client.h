#pragma once

#include "pool-config.h"
#include "json.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using json = common_json;

namespace pool {

// in-process pub/sub for coordinator -> router envelope events.
// the pool thread delivers each envelope once to every subscriber; WS-05
// (dispatch) registers handlers for work_offer / exclusion_update / eviction_notice here
class pool_event_bus {
public:
    using handler_t = std::function<void(const json & envelope)>;
    using token_t   = uint64_t;

    // register a handler, returns the token for unsubscribe
    token_t subscribe(handler_t handler);
    void unsubscribe(token_t token);

    // deliver to all subscribers; may be called from any thread
    void post(const json & envelope);

private:
    std::mutex mutex;
    uint64_t next_token = 1;
    std::vector<std::pair<token_t, handler_t>> handlers;
};

// router side of the pool control channel: subscribe loop (SSE or long-poll),
// event POST, handshake and capacity emission. background threads start only
// when the pool is configured; everything is a no-op otherwise
class pool_client {
public:
    pool_client();
    ~pool_client();

    pool_client(const pool_client &) = delete;
    pool_client & operator=(const pool_client &) = delete;

    // start the background threads. capacity_snapshot returns the model array of
    // capacity.payload.models (catalog 3.2); it is called off the HTTP threads
    bool start(const pool_config & cfg, std::function<json()> capacity_snapshot, std::string & err);
    void stop();

    // fire and forget: queue a fresh capacity snapshot. safe from any thread, no-op when not running
    void on_capacity_change();

    // the subscribe link (SSE stream or held long-poll) to the coordinator is
    // open right now; false when the pool is off or between failed rounds
    bool link_open() const;
    // milliseconds since the last subscribe event or heartbeat; 0 when never in contact
    uint64_t ms_since_last_contact() const;
    // latched on the first successful subscribe round (SSE or long-poll); the
    // router's startup gate waits for this when a coordinator is configured
    bool established() const;

    pool_event_bus & bus() { return bus_; }

    // the running config (router_id, name, token); null when the pool is off
    const pool_config * config() const;

private:
    struct state;
    std::unique_ptr<state> st;
    pool_event_bus bus_;
};

// global instance, owned by the server
pool_client & pool_instance();

// hook for the router state machine: a local model became loaded or unloaded.
// builds the capacity snapshot and queues the event; no-op when the pool is off
void pool_on_capacity_change();
}
