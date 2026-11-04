/**
 * poolStore - Local pool state of this router (WS-16)
 *
 * Owns the GET /pool/status snapshot and its refresh cadence. The router
 * reports its own pool membership: the coordinator link, the class
 * advertisement and the serving/originating counters. The pool-wide sharing
 * policy is not owned here; it is served by the coordinator portal on its
 * own origin with cookie auth.
 *
 * Degrades gracefully: a 404 (a router build without the endpoint) or
 * `pool: false` (no coordinator configured) both settle into the
 * "pool not configured" state instead of an error.
 */

import { PoolService } from '$lib/services/pool.service';
import { ApiError } from '$lib/utils';

/** The status cadence while the pool page is open: the link and the counters move slowly. */
const REFRESH_INTERVAL_MS = 5000;
/** The settled "pool not configured" snapshot; a 404 endpoint maps onto it too. */
const NOT_CONFIGURED_STATUS: ApiPoolStatusResponse = {
	classes: [],
	connected: false,
	last_contact_ms: 0,
	originating: { pending: 0, total: 0 },
	pool: false,
	router_id: '',
	serving: { pending: 0, served_total: 0, tokens_served_total: 0 }
};

class PoolStore {
	error = $state<string | null>(null);
	loading = $state(false);
	status = $state<ApiPoolStatusResponse | null>(null);

	private inflight: Promise<void> | null = null;
	private timer: ReturnType<typeof setInterval> | null = null;

	get classes(): ApiPoolClassEntry[] {
		return this.status?.classes ?? [];
	}

	/** The endpoint answered and the router has a coordinator. */
	get configured(): boolean {
		return this.status?.pool ?? false;
	}

	get connected(): boolean {
		return this.status?.connected ?? false;
	}

	/** The endpoint answered with `pool: false`, or was missing (404). */
	get notConfigured(): boolean {
		return this.status !== null && this.status.pool === false;
	}

	/**
	 * Fetch the status. Resolves with the stored snapshot (the last good one
	 * on a failed refresh). A 404 endpoint settles into the not-configured
	 * snapshot, so the caller never needs error handling for the pool-off case.
	 */
	async fetch(): Promise<ApiPoolStatusResponse | null> {
		if (this.inflight) return this.inflight;

		this.inflight = (async () => {
			this.loading = true;
			this.error = null;

			try {
				this.status = await PoolService.status();
			} catch (e) {
				if (e instanceof ApiError && e.status === 404) {
					this.status = NOT_CONFIGURED_STATUS;
				} else {
					this.error = e instanceof Error ? e.message : 'pool status unavailable';
				}
			} finally {
				this.loading = false;
			}
		})();

		try {
			await this.inflight;
		} finally {
			this.inflight = null;
		}

		return this.status;
	}

	/** Start the refresh cadence; idempotent. */
	startPolling(): void {
		if (this.timer) return;

		void this.fetch();

		this.timer = setInterval(() => {
			void this.fetch();
		}, REFRESH_INTERVAL_MS);
	}

	/** Stop the refresh cadence. */
	stopPolling(): void {
		if (this.timer) {
			clearInterval(this.timer);
			this.timer = null;
		}
	}
}

export const poolStore = new PoolStore();
