<script lang="ts">
	import { Network } from '@lucide/svelte';
	import { PoolMembersPanel, PoolStatusPanel } from '$lib/components/app';
	import { Skeleton } from '$lib/components/ui/skeleton';
	import { deviceStore, modelsStore, poolStore } from '$lib/stores';

	// the local models list feeds the member rows, and the status refreshes on
	// a cadence while the page is open; both stop when it unmounts
	$effect(() => {
		void modelsStore.fetchRouterModels();

		poolStore.startPolling();

		return () => {
			poolStore.stopPolling();
		};
	});

	let isMobile = $derived(deviceStore.isMobile);
	let ready = $derived(poolStore.status !== null);
	let notConfigured = $derived(poolStore.notConfigured);
</script>

<svelte:head>
	<title>Pool · llama.cpp</title>
</svelte:head>

<div class="mx-auto max-w-4xl p-4">
	<header class="mb-4 flex items-center gap-2">
		<Network class="size-5 text-muted-foreground" />

		<h1 class="text-lg font-semibold">Pool</h1>
	</header>

	{#if !ready}
		<div aria-busy="true" aria-label="Loading pool status" class="space-y-4">
			<Skeleton class="h-44 w-full" />

			<Skeleton class="h-24 w-full" />

			<Skeleton class="h-40 w-full" />
		</div>
	{:else if notConfigured}
		<div class="rounded-xl bg-card p-6 shadow-sm">
			<p class="text-sm text-muted-foreground">
				Pool is not configured on this router. Start it with a coordinator (--pool-server).
			</p>
		</div>
	{:else}
		<div class={isMobile ? 'space-y-4' : 'grid items-start gap-4 lg:grid-cols-2'}>
			<div class="space-y-4">
				<PoolStatusPanel />
			</div>

			<div class="space-y-4">
				<PoolMembersPanel />
			</div>
		</div>
	{/if}
</div>
