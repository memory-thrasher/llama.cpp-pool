/**
 * PoolService - Local pool state API layer (WS-16)
 *
 * Wraps GET /pool/status, the read-only view of this router's pool state:
 * coordinator link, class advertisement and the serving/originating
 * counters. The pool-wide sharing policy is not here; it is served by the
 * coordinator portal on its own origin with cookie auth.
 */

import { API_POOL } from '$lib/constants';
import { apiFetch } from '$lib/utils';

export class PoolService {
	/**
	 * Fetch the local pool state of this router.
	 *
	 * @returns The pinned status object; `pool: false` on a router without a coordinator
	 */
	static async status(): Promise<ApiPoolStatusResponse> {
		return apiFetch<ApiPoolStatusResponse>(API_POOL.STATUS);
	}
}
