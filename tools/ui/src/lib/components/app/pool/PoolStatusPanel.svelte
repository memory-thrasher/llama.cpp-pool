<script lang="ts">
	import { CircleCheck, CircleX, Waypoints } from '@lucide/svelte';
	import * as Card from '$lib/components/ui/card';
	import * as Table from '$lib/components/ui/table';
	import { poolStore } from '$lib/stores';

	/** The router id is long; the row shows the head and tail, the title carries the full id. */
	function shortId(id: string): string {
		return id.length <= 13 ? id : `${id.slice(0, 8)}…${id.slice(-4)}`;
	}

	/** "12 s ago", "1 m 05 s ago"; 0 means never in contact. */
	function contactAge(ms: number): string {
		if (ms === 0) return 'never';

		const s = Math.floor(ms / 1000);

		if (s < 10) return `${s} s ago`;

		const m = Math.floor(s / 60);
		const rest = s % 60;

		return m > 0 ? `${m} m ${String(rest).padStart(2, '0')} s ago` : `${s} s ago`;
	}

	let status = $derived(poolStore.status);
	let routerId = $derived(status?.router_id ?? '');
	let connected = $derived(poolStore.connected);
	let lastContact = $derived(status?.last_contact_ms ?? 0);
	let classes = $derived(poolStore.classes);
	let serving = $derived(status?.serving ?? null);
	let originating = $derived(status?.originating ?? null);
</script>

<!-- Pool status: the link to the coordinator and the class advertisement -->
<Card.Root>
	<Card.Header>
		<Card.Title>Pool status</Card.Title>

		<Card.Action>
			{#if connected}
				<span
					aria-label="Connected to the pool coordinator"
					class="inline-flex items-center gap-1 text-xs font-medium text-emerald-600 dark:text-emerald-400"
				>
					<CircleCheck class="size-3.5" />
					Up
				</span>
			{:else}
				<span
					aria-label="Not connected to the pool coordinator"
					class="inline-flex items-center gap-1 text-xs font-medium text-red-600 dark:text-red-400"
				>
					<CircleX class="size-3.5" />
					Down
				</span>
			{/if}
		</Card.Action>
	</Card.Header>

	<Card.Content class="space-y-3">
		<dl class="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
			<dt class="text-muted-foreground">Last contact</dt>

			<dd>{contactAge(lastContact)}</dd>

			{#if routerId}
				<dt class="text-muted-foreground">Router</dt>

				<dd class="truncate font-mono text-xs" title={routerId}>{shortId(routerId)}</dd>
			{/if}
		</dl>

		<div class="border-t pt-3">
			<div class="mb-2 flex items-center gap-2 text-sm font-medium">
				<Waypoints class="size-4 text-muted-foreground" />

				Classes
			</div>

			{#if classes.length > 0}
				<Table.Root>
					<Table.Header>
						<Table.Row>
							<Table.Head>Tag</Table.Head>

							<Table.Head class="text-right">Members</Table.Head>
						</Table.Row>
					</Table.Header>

					<Table.Body>
						{#each classes as entry (entry.tag)}
							<Table.Row>
								<Table.Cell class="font-mono text-xs">{entry.tag}</Table.Cell>

								<Table.Cell class="text-right">{entry.member_count}</Table.Cell>
							</Table.Row>
						{/each}
					</Table.Body>
				</Table.Root>
			{:else}
				<p class="text-sm text-muted-foreground">No classes advertised on this router.</p>
			{/if}
		</div>
	</Card.Content>
</Card.Root>

<!-- Queue: what this router is serving right now and is waiting for -->
<Card.Root>
	<Card.Header>
		<Card.Title>Queue</Card.Title>
	</Card.Header>

	<Card.Content class="space-y-3">
		<div class="grid grid-cols-2 gap-3">
			<div class="rounded-lg bg-muted/50 p-3">
				<div class="text-xs text-muted-foreground">Serving now</div>

				<div class="text-2xl font-semibold">{serving?.pending ?? 0}</div>
			</div>

			<div class="rounded-lg bg-muted/50 p-3">
				<div class="text-xs text-muted-foreground">Waiting for the pool</div>

				<div class="text-2xl font-semibold">{originating?.pending ?? 0}</div>
			</div>
		</div>

		<p class="text-xs text-muted-foreground">
			Offered and running jobs cannot be cancelled from here (MVP).
		</p>
	</Card.Content>
</Card.Root>

<!-- Contribution: this router's share, cumulative since process start -->
<Card.Root>
	<Card.Header>
		<Card.Title>Contribution (local)</Card.Title>
	</Card.Header>

	<Card.Content class="space-y-3">
		<div class="grid grid-cols-2 gap-3">
			<div class="rounded-lg bg-muted/50 p-3">
				<div class="text-xs text-muted-foreground">Jobs served</div>

				<div class="text-2xl font-semibold">{serving?.served_total ?? 0}</div>
			</div>

			<div class="rounded-lg bg-muted/50 p-3">
				<div class="text-xs text-muted-foreground">Tokens served</div>

				<div class="text-2xl font-semibold">
					{(serving?.tokens_served_total ?? 0).toLocaleString()}
				</div>
			</div>
		</div>

		<p class="text-xs text-muted-foreground">
			Counts for this router only. Pool-wide sharing policy: see the coordinator portal.
		</p>
	</Card.Content>
</Card.Root>
