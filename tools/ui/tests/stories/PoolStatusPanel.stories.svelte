<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import PoolStatusPanel from '$lib/components/app/pool/PoolStatusPanel.svelte';
	import { poolStore } from '$lib/stores/pool.svelte';
	import { expect } from 'storybook/test';

	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Components/PoolStatusPanel'
	});

	const routerId = '3f2b8c1e-9a4d-4e7f-8c2a-1d5e6f7a8b9c';

	// the panel reads the store, as on the pool page; the snapshot is the
	// pinned GET /pool/status shape (WS-16)
	poolStore.status = {
		classes: [
			{ member_count: 2, tag: 'pool-text-md' },
			{ member_count: 1, tag: 'pool-text-xs' }
		],
		connected: true,
		last_contact_ms: 12_000,
		originating: { pending: 1, total: 7 },
		pool: true,
		router_id: routerId,
		serving: { pending: 1, served_total: 42, tokens_served_total: 12_345 }
	};
</script>

<Story
	name="Status"
	play={async ({ canvas }) => {
		// the link state, the classes and the local counters render from the snapshot
		await expect(canvas.getByText('Up')).toBeInTheDocument();
		await expect(canvas.getByText('pool-text-md')).toBeInTheDocument();
		await expect(canvas.getByText('pool-text-xs')).toBeInTheDocument();
		await expect(canvas.getByText('12 s ago')).toBeInTheDocument();
		await expect(canvas.getByText('Jobs served')).toBeInTheDocument();
		await expect(canvas.getByText('Tokens served')).toBeInTheDocument();
	}}
>
	<div class="mx-auto max-w-md space-y-4 p-4">
		<PoolStatusPanel />
	</div>
</Story>
