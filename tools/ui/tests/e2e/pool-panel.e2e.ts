// Browser-level pin for the pool page (WS-16): the panels are covered by
// vitest components, but nothing rendered the page in a real browser. The
// built app is served same-origin by the webServer, and the UI's apiFetch
// uses relative paths, so the API routes are intercepted with page.route.
// The app is hash-routed, so the page lives at /#/pool.
// The shapes below are the pinned GET /pool/status and /v1/models bodies
// (tools/server/server-models.cpp: pool_status(), the router model list).

import { expect, type Page, test } from '@playwright/test';

const MODEL_OFFERED = 'ggml-org/E2E-Pool-A-GGUF:Q8_0';
const MODEL_WITHHELD = 'ggml-org/E2E-Pool-B-GGUF:Q8_0';
/** The connected snapshot the panel renders (counts, tags and ids only). */
const STATUS_CONNECTED = {
	classes: [{ member_count: 2, tag: 'pool-text-xs' }],
	connected: true,
	last_contact_ms: 12_000,
	originating: { pending: 0, total: 8 },
	pool: true,
	router_id: 'r-e2e',
	serving: { pending: 1, served_total: 12, tokens_served_total: 3400 }
};
/** The router without a coordinator: the settled not-configured state. */
const STATUS_NOT_CONFIGURED = {
	classes: [],
	connected: false,
	last_contact_ms: 0,
	originating: { pending: 0, total: 0 },
	pool: false,
	router_id: '',
	serving: { pending: 0, served_total: 0, tokens_served_total: 0 }
};

function modelEntry(id: string, status: string) {
	return {
		created: 0,
		id,
		in_cache: true,
		object: 'model',
		owned_by: 'llamacpp',
		path: `/models/${id}`,
		status: { value: status },
		tags: ['pool-text-xs']
	};
}

/**
 * The routes the page touches: /props (the role detection - ROUTER, so the
 * member list fetch runs - and the per-model modality fetch), /pool/status
 * (the snapshot), and /v1/models (the member rows).
 */
async function mockPoolApis(page: Page, status: object, models: object[]) {
	await page.route('**/props*', (route) => route.fulfill({ json: { role: 'router' } }));
	await page.route('**/pool/status', (route) => route.fulfill({ json: status }));
	await page.route('**/v1/models', (route) => route.fulfill({ json: { data: models } }));
}

test.describe('Pool page', () => {
	test('connected: link state, classes, queue and member rows', async ({ page }) => {
		await mockPoolApis(page, STATUS_CONNECTED, [
			modelEntry(MODEL_OFFERED, 'loaded'),
			modelEntry(MODEL_WITHHELD, 'excluded')
		]);

		const errors: string[] = [];

		page.on('pageerror', (err) => errors.push(err.message));

		await page.goto('/#/pool');

		// the link is up, with the human-rendered last contact and the router id
		await expect(page.getByText('Up', { exact: true })).toBeVisible();
		await expect(page.getByText('Down', { exact: true })).toHaveCount(0);
		await expect(page.getByText('12 s ago')).toBeVisible();
		await expect(page.getByText('r-e2e')).toBeVisible();

		// the class advertisement row (the tag also appears on the member badges
		// below, so scope to the table cell)
		await expect(page.getByRole('cell', { name: 'pool-text-xs' })).toBeVisible();

		// the member rows: both models listed, the excluded one withheld. the ids
		// also appear in the Local models card, so scope to the members card
		const members = page.getByLabel('Pool members');

		await expect(members.getByText(MODEL_OFFERED)).toBeVisible();
		await expect(members.getByText(MODEL_WITHHELD)).toBeVisible();
		await expect(members.getByText('Offered', { exact: true })).toBeVisible();
		await expect(members.getByText('Withheld', { exact: true })).toBeVisible();

		// the queue and contribution numbers of the snapshot: serving pending
		// 1, class member count 2, jobs served 12, tokens 3400 (locale-formatted)
		await expect(page.getByText('Serving now')).toBeVisible();
		await expect(page.getByText('Waiting for the pool')).toBeVisible();
		await expect(page.getByText('Jobs served')).toBeVisible();
		await expect(page.getByText('Tokens served')).toBeVisible();
		await expect(page.getByText('1', { exact: true })).toBeVisible();
		await expect(page.getByText('2', { exact: true })).toBeVisible();
		await expect(page.getByText('12', { exact: true })).toBeVisible();
		await expect(page.getByText('3,400')).toBeVisible();

		expect(errors).toEqual([]);
	});

	test('not configured: the coordinator-less state, no member table', async ({ page }) => {
		await mockPoolApis(page, STATUS_NOT_CONFIGURED, []);

		await page.goto('/#/pool');

		await expect(page.getByText('Pool is not configured on this router.')).toBeVisible();
		// the panels never render in the not-configured state
		await expect(page.getByText('Pool status')).toHaveCount(0);
		await expect(page.getByText('Offer / withhold')).toHaveCount(0);
	});

	test('degraded: a 500 /pool/status leaves the page functional', async ({ page }) => {
		await page.route('**/props*', (route) => route.fulfill({ json: { role: 'router' } }));
		await page.route('**/pool/status', (route) =>
			route.fulfill({ body: '{"error":"boom"}', contentType: 'application/json', status: 500 })
		);
		await page.route('**/v1/models', (route) => route.fulfill({ json: { data: [] } }));

		const errors: string[] = [];

		page.on('pageerror', (err) => errors.push(err.message));

		await page.goto('/#/pool');

		// the snapshot never settles, so the page holds its loading state
		// (skeleton) - no uncaught-error blank screen, no panels
		await expect(page.getByLabel('Loading pool status')).toBeVisible();
		await expect(page.getByText('Pool status')).toHaveCount(0);
		await expect(page.getByText('Offer / withhold')).toHaveCount(0);

		expect(errors).toEqual([]);
	});
});
