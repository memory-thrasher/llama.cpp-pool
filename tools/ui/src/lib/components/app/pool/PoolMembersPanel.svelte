<script lang="ts">
	import { Cpu } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import * as Card from '$lib/components/ui/card';
	import { Switch } from '$lib/components/ui/switch';
	import { ServerModelStatus } from '$lib/enums';
	import { ModelsService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import { SvelteSet } from 'svelte/reactivity';
	import { toast } from 'svelte-sonner';

	/** The pool- prefix is reserved for classes (server-models.cpp). */
	const POOL_TAG_PREFIX = 'pool-';

	/** The load states of the local models list, in display order. */
	const STATUS_LABELS: Record<string, string> = {
		[ServerModelStatus.EXCLUDED]: 'Withheld',
		[ServerModelStatus.FAILED]: 'Failed',
		[ServerModelStatus.LOADED]: 'Loaded',
		[ServerModelStatus.LOADING]: 'Loading',
		[ServerModelStatus.SLEEPING]: 'Sleeping',
		[ServerModelStatus.UNLOADED]: 'Unloaded'
	};

	function poolTags(entry: ApiModelDataEntry): string[] {
		return (entry.tags ?? []).filter((tag: string) => tag.startsWith(POOL_TAG_PREFIX));
	}

	/** A row is a loaded pool member: loaded = offered, excluded = withheld (D-F1).
	 *  Plain models (no pool tag) keep the existing load/unload semantics untouched. */
	let members = $derived(
		modelsStore.routerModels.filter(
			(m) =>
				poolTags(m).length > 0 &&
				(m.status.value === ServerModelStatus.LOADED ||
					m.status.value === ServerModelStatus.EXCLUDED)
		)
	);

	/** Every local model with its load state; the pool rows are a subset of this. */
	let localModels = $derived(modelsStore.routerModels);

	let busyIds = new SvelteSet<string>();

	async function toggleOffer(entry: ApiModelDataEntry) {
		const id = entry.id;

		if (busyIds.has(id)) return;

		const withheld = entry.status.value === ServerModelStatus.EXCLUDED;

		busyIds.add(id);

		try {
			if (withheld) {
				// the unmark: the member re-enters the pool; the model stays loaded
				await ModelsService.load(id);
			} else {
				// the mark: the member is withheld; the model stays loaded (WS-12)
				await ModelsService.unload(id);
			}

			// the mark is not tracked by the /models/sse feed, so re-list to refresh the panel
			await modelsStore.fetchRouterModels();
		} catch (error) {
			const what = withheld ? 'offer' : 'withhold';

			toast.error(
				`Failed to ${what} ${id}: ${error instanceof Error ? error.message : 'request failed'}`
			);
		} finally {
			busyIds.delete(id);
		}
	}
</script>

<!-- Offer/withhold: one click pulls a GPU out of the pool (D-O1) -->
<Card.Root aria-label="Pool members">
	<Card.Header>
		<Card.Title>Offer / withhold</Card.Title>
	</Card.Header>

	<Card.Content class="space-y-2">
		{#if members.length > 0}
			<div class="space-y-2">
				{#each members as entry (entry.id)}
					{@const tags = poolTags(entry)}
					{@const withheld = entry.status.value === ServerModelStatus.EXCLUDED}
					<div class="flex items-center justify-between gap-3 rounded-lg bg-muted/50 p-3">
						<div class="min-w-0">
							<div class="flex items-center gap-2">
								<span class="truncate font-mono text-xs" title={entry.id}>{entry.id}</span>

								<Badge
									class={withheld
										? 'bg-amber-500/15 text-amber-700 dark:text-amber-400'
										: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400'}
								>
									{withheld ? 'Withheld' : 'Offered'}
								</Badge>
							</div>

							{#if tags.length > 0}
								<div class="mt-1 flex flex-wrap gap-1">
									{#each tags as tag (tag)}
										<Badge class="font-mono text-[11px]" variant="secondary">{tag}</Badge>
									{/each}
								</div>
							{/if}
						</div>

						<Switch
							aria-label={withheld
								? `Offer ${entry.id} to the pool`
								: `Withhold ${entry.id} from the pool`}
							checked={!withheld}
							disabled={busyIds.has(entry.id)}
							onCheckedChange={() => toggleOffer(entry)}
						/>
					</div>
				{/each}
			</div>
		{:else}
			<p class="text-sm text-muted-foreground">No loaded pool members on this router.</p>
		{/if}
	</Card.Content>
</Card.Root>

<!-- Local models: what is loaded on this router, by state -->
<Card.Root aria-label="Local models">
	<Card.Header>
		<Card.Title>Local models</Card.Title>
	</Card.Header>

	<Card.Content class="space-y-2">
		{#if localModels.length > 0}
			<div class="space-y-1">
				{#each localModels as entry (entry.id)}
					<div class="flex items-center justify-between gap-3 px-1 py-0.5">
						<span class="min-w-0 truncate font-mono text-xs" title={entry.id}>{entry.id}</span>

						<Badge
							variant={entry.status.value === ServerModelStatus.LOADED ||
							entry.status.value === ServerModelStatus.EXCLUDED
								? 'secondary'
								: 'outline'}
						>
							{STATUS_LABELS[entry.status.value] ?? entry.status.value}
						</Badge>
					</div>
				{/each}
			</div>
		{:else}
			<p class="flex items-center gap-2 text-sm text-muted-foreground">
				<Cpu class="size-4" />
				No models registered on this router.
			</p>
		{/if}
	</Card.Content>
</Card.Root>
