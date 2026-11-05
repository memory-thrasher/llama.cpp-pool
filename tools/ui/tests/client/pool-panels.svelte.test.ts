// Guards the pool panels (WS-16): the status panel renders the pinned
// /pool/status snapshot, and the member panel offers one toggle per loaded
// pool member - offered (loaded) and withheld (excluded) - while plain models
// keep the existing load/unload semantics untouched.

import PoolMembersPanel from '$lib/components/app/pool/PoolMembersPanel.svelte';
import PoolStatusPanel from '$lib/components/app/pool/PoolStatusPanel.svelte';
import { ServerModelStatus } from '$lib/enums';
import { ModelsService } from '$lib/services/models.service';
import { modelsStore, poolStore } from '$lib/stores';
import type { ApiModelDataEntry, ApiPoolStatusResponse } from '$lib/types/api';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

/** A router listing entry with a fixed status. */
function modelEntry(id: string, status: ServerModelStatus, tags: string[] = []) {
	return {
		created: 0,
		id,
		in_cache: true,
		object: 'model',
		owned_by: 'llamacpp',
		path: `/models/${id}`,
		status: { value: status },
		tags
	} as unknown as ApiModelDataEntry;
}

const POOLED = 'org/Pooled-Model-GGUF:Q4_K_M';
const POOLED_WITHHELD = 'org/Pooled-Withheld-GGUF:Q4_K_M';
const PLAIN = 'org/Plain-Model-GGUF:Q4_K_M';

function statusSnapshot(overrides: Partial<ApiPoolStatusResponse> = {}): ApiPoolStatusResponse {
	return {
		classes: [{ member_count: 3, tag: 'pool-text-xs' }],
		connected: true,
		last_contact_ms: 12_000,
		originating: { pending: 0, total: 7 },
		pool: true,
		router_id: '3f2b8c1e-9a4d-4e7f-8c2a-1d5e6f7a8b9c',
		serving: { pending: 1, served_total: 42, tokens_served_total: 12_345 },
		...overrides
	};
}

beforeEach(() => {
	vi.restoreAllMocks();
	poolStore.status = null;
	modelsStore.routerModels = [];
});

describe('PoolStatusPanel', () => {
	it('renders the link state, classes and counters of the snapshot', async () => {
		poolStore.status = statusSnapshot();

		const screen = await render(PoolStatusPanel);

		// the link is up, with the human-rendered last contact and the router id
		await expect.element(screen.getByText('Up')).toBeVisible();
		await expect.element(screen.getByText('12 s ago')).toBeVisible();
		await expect.element(screen.getByText(/3f2b8c1e/)).toBeVisible();

		// the class advertisement: tag and member count
		await expect.element(screen.getByText('pool-text-xs')).toBeVisible();
		await expect(screen.getByText('3')).not.toBeNull();

		// the queue counts
		await expect.element(screen.getByText('Serving now')).toBeVisible();
		await expect.element(screen.getByText('Waiting for the pool')).toBeVisible();

		// the local contribution, cumulative since process start
		await expect.element(screen.getByText('Jobs served')).toBeVisible();
		await expect.element(screen.getByText('Tokens served')).toBeVisible();
		await expect(screen.getByText('42')).not.toBeNull();
		await expect(screen.getByText('12,345')).not.toBeNull();
	});

	it('shows the down state when the link is closed', async () => {
		poolStore.status = statusSnapshot({ connected: false, last_contact_ms: 420_000 });

		const screen = await render(PoolStatusPanel);

		await expect.element(screen.getByText('Down')).toBeVisible();
	});
});

describe('PoolMembersPanel', () => {
	function stubMembers() {
		modelsStore.routerModels = [
			modelEntry(POOLED, ServerModelStatus.LOADED, ['pool-text-xs']),
			modelEntry(POOLED_WITHHELD, ServerModelStatus.EXCLUDED, ['pool-text-md']),
			modelEntry(PLAIN, ServerModelStatus.LOADED, [])
		];

		const load = vi.spyOn(ModelsService, 'load').mockResolvedValue({ success: true });
		const unload = vi.spyOn(ModelsService, 'unload').mockResolvedValue({ success: true });

		vi.spyOn(modelsStore, 'fetchRouterModels').mockResolvedValue(undefined);

		return { load, unload };
	}

	it('lists the loaded pool members, offered and withheld, with their class tags', async () => {
		stubMembers();

		const screen = await render(PoolMembersPanel);

		// scoped to the members card: the ids also appear in the Local models card below
		const card = screen.container.querySelector('[aria-label="Pool members"]');
		if (!card) {
			throw new Error('the Pool members card is not rendered');
		}

		// the two pool members, in their states
		expect(card?.textContent).toContain(POOLED);
		expect(card?.textContent).toContain(POOLED_WITHHELD);

		// the state badges: one offered, one withheld ('Withheld' is also a
		// substring of the withheld model's id, so match the badge text exactly)
		const badges = [...card.querySelectorAll('[data-slot="badge"]')].map((b) => b.textContent);
		expect(badges).toContain('Offered');
		expect(badges).toContain('Withheld');

		// the class labels (D-B6) are read-only badges next to the rows
		expect(card?.textContent).toContain('pool-text-xs');
		expect(card?.textContent).toContain('pool-text-md');
	});

	it('offers a withheld member: POST /models/load (the unmark), then re-lists', async () => {
		const { load, unload } = stubMembers();
		const screen = await render(PoolMembersPanel);
		const offered = screen.getByRole('switch', { name: `Offer ${POOLED_WITHHELD} to the pool` });

		await userEvent.click(offered);

		expect(load).toHaveBeenCalledWith(POOLED_WITHHELD);
		expect(unload).not.toHaveBeenCalled();
		expect(modelsStore.fetchRouterModels).toHaveBeenCalled();
	});

	it('withholds an offered member: POST /models/unload (the mark), then re-lists', async () => {
		const { load, unload } = stubMembers();
		const screen = await render(PoolMembersPanel);
		const withheld = screen.getByRole('switch', { name: `Withhold ${POOLED} from the pool` });

		await userEvent.click(withheld);

		expect(unload).toHaveBeenCalledWith(POOLED);
		expect(load).not.toHaveBeenCalled();
		expect(modelsStore.fetchRouterModels).toHaveBeenCalled();
	});

	it('renders no toggle for a plain model, and none for an unloaded pool member', async () => {
		modelsStore.routerModels = [
			modelEntry(POOLED, ServerModelStatus.LOADED, ['pool-text-xs']),
			modelEntry(POOLED_WITHHELD, ServerModelStatus.UNLOADED, ['pool-text-md']),
			modelEntry(PLAIN, ServerModelStatus.LOADED, [])
		];

		const load = vi.spyOn(ModelsService, 'load').mockResolvedValue({ success: true });
		const unload = vi.spyOn(ModelsService, 'unload').mockResolvedValue({ success: true });

		vi.spyOn(modelsStore, 'fetchRouterModels').mockResolvedValue(undefined);

		const screen = await render(PoolMembersPanel);

		const card = screen.container.querySelector('[aria-label="Pool members"]');

		// only the loaded pool member is a row; the unloaded pool member and the
		// plain model are not listed (they do appear in the Local models card)
		expect(card?.textContent).toContain(POOLED);
		expect(card?.textContent).not.toContain(POOLED_WITHHELD);
		expect(card?.textContent).not.toContain(PLAIN);

		// exactly one toggle exists, and the plain model never drives it
		expect(screen.container.querySelectorAll('[role="switch"]')).toHaveLength(1);

		await userEvent.click(screen.getByRole('switch'));

		expect(unload).toHaveBeenCalledWith(POOLED);
		expect(load).not.toHaveBeenCalled();
	});
});
