// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original classical WASM workflow execution
// Authentic public workflow runtime from Git a23c96cf6dcb9b08245d6a11d1c002e988b31597.
// Original project sources compiled without edits; unchanged pinned SDK packages stay external.
//#region \0rolldown/runtime.js
var __defProp = Object.defineProperty;
var __esmMin = (fn, res, err) => () => {
	if (err) throw err[0];
	try {
		return fn && (res = fn(fn = 0)), res;
	} catch (e) {
		throw err = [e], e;
	}
};
var __exportAll = (all, no_symbols) => {
	let target = {};
	for (var name in all) __defProp(target, name, {
		get: all[name],
		enumerable: true
	});
	if (!no_symbols) __defProp(target, Symbol.toStringTag, { value: "Module" });
	return target;
};
var init_wasmAssets = __esmMin((() => {
	new URL("../wasm/scpn_quantum_studio_wasm_kernel.wasm", "" + import.meta.url).href;
	new URL("../wasm/scpn_quantum_studio_program_ad_wasm.wasm", "" + import.meta.url).href;
}));
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/contracts/canonical.ts
/** Validate Unicode scalar sequences without normalising their spelling. */
function scalarString(value, path = "$") {
	if (/[\ud800-\udfff]/u.test(value)) throw new Error(`${path}: invalid Unicode scalar`);
	return value;
}
/**
* Identify C0 and DEL characters without interpreting source text as a pattern.
* @param value Original string, with no normalization or replacement.
* @param includeC1 Also reject the C1 control range required by operator metadata.
* @returns Whether any of the selected control characters occurs in the source.
*/
function containsControlCharacters(value, includeC1 = false) {
	for (const character of value) {
		const code = character.charCodeAt(0);
		if (code <= 31 || code === 127 || includeC1 && code >= 128 && code <= 159) return true;
	}
	return false;
}
/** Inspect plain data members without invoking getters or prototype methods. */
function dataEntries(value, path = "$") {
	const prototype = Object.getPrototypeOf(value);
	if (prototype !== Object.prototype && prototype !== null) throw new Error(`${path}: unsupported object`);
	return Reflect.ownKeys(value).map((key) => {
		if (typeof key !== "string") throw new Error(`${path}: symbol key unsupported`);
		const descriptor = Object.getOwnPropertyDescriptor(value, key);
		if (descriptor === void 0 || !("value" in descriptor) || !descriptor.enumerable) throw new Error(`${path}: data member required`);
		return [scalarString(key, path), descriptor.value];
	});
}
function byteOrder(left, right) {
	const encoder = new TextEncoder();
	const a = encoder.encode(left);
	const b = encoder.encode(right);
	for (let index = 0; index < Math.min(a.length, b.length); index++) {
		const difference = a[index] - b[index];
		if (difference !== 0) return difference;
	}
	return a.length - b.length;
}
function tag(value, path, depth, ancestors) {
	if (depth > 64 || depth === 64 && typeof value === "object" && value !== null) throw new Error(`${path}: depth exceeded`);
	if (value === null || typeof value === "boolean") return value;
	if (typeof value === "string") return scalarString(value, path);
	if (typeof value === "bigint") {
		const decimal = value.toString();
		if (decimal.replace(/^-/, "").length > 4096) throw new Error(`${path}: integer scalar too large`);
		return ["integer", decimal];
	}
	if (typeof value === "number") {
		if (!Number.isFinite(value)) throw new Error(`${path}: non-finite float`);
		const bytes = /* @__PURE__ */ new Uint8Array(8);
		new DataView(bytes.buffer).setFloat64(0, value, false);
		return ["float64", Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")];
	}
	if (typeof value !== "object") throw new Error(`${path}: unsupported value type`);
	if (ancestors.has(value)) throw new Error(`${path}: ancestor cycle`);
	ancestors.add(value);
	try {
		if (Array.isArray(value)) {
			if (Object.getPrototypeOf(value) !== Array.prototype) throw new Error(`${path}: unsupported array prototype`);
			if (Reflect.ownKeys(value).length !== value.length + 1) throw new Error(`${path}: sparse or decorated array`);
			const items = [];
			for (let index = 0; index < value.length; index++) {
				const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
				if (!descriptor || !("value" in descriptor)) throw new Error(`${path}: array data element required`);
				items.push(tag(descriptor.value, `${path}[${index}]`, depth + 1, ancestors));
			}
			return ["array", items];
		}
		return ["object", dataEntries(value, path).sort(([a], [b]) => byteOrder(a, b)).map(([key, item]) => [key, tag(item, `${path}.${key}`, depth + 1, ancestors)])];
	} finally {
		ancestors.delete(value);
	}
}
/** Encode schema + LF + compact typed JSON, preserving integer/float identity. */
function canonicalBytes(schema, body) {
	if (!schema || /[\r\n]/.test(schema)) throw new Error("schema: invalid domain prefix");
	scalarString(schema, "schema");
	return new TextEncoder().encode(`${schema}\n${JSON.stringify(tag(body, "$", 0, /* @__PURE__ */ new Set()))}`);
}
/** Hash the complete canonical byte sequence; existing raw evidence uses its own codec. */
async function canonicalDigest(schema, body) {
	const digest = await crypto.subtle.digest("SHA-256", canonicalBytes(schema, body));
	return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
var init_canonical = __esmMin((() => {}));
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/resources/admission.ts
function integer$2(value, name, allowZero = false) {
	if (typeof value !== "bigint" || value < (allowZero ? 0n : 1n)) throw new Error(`${name}: ${allowZero ? "nonnegative" : "positive"} integer required`);
	return value;
}
function label(value, name) {
	if (typeof value !== "string" || !value.trim() || value.length > 4096) throw new Error(`${name}: bounded source identity required`);
	return scalarString(value, name);
}
function record$1(value, keys) {
	if (typeof value !== "object" || value === null || Reflect.ownKeys(value).length !== keys.length) throw new Error("resource record fields are incomplete or unsupported");
	if (dataEntries(value).some(([key]) => !keys.includes(key))) throw new Error("unsupported resource record field");
}
function array$1(value, minimum, maximum) {
	if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype || value.length < minimum || value.length > maximum || Reflect.ownKeys(value).length !== value.length + 1) throw new Error("bounded plain resource array required");
	for (let index = 0; index < value.length; index++) {
		const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
		if (!descriptor || !("value" in descriptor)) throw new Error("resource array data member required");
	}
}
function product(left, right, limit) {
	if (right > limit / left) throw new Error("declared storage exceeds addressability");
	return left * right;
}
/** Sum fixed-width live declarations exactly without constructing their buffers. */
function estimateResourcePlan(plan, addressableBytes) {
	record$1(plan, [
		"backend",
		"method",
		"buffers",
		"concurrency",
		"workUnits"
	]);
	const limit = integer$2(addressableBytes, "addressableBytes");
	if (limit > 18446744073709551615n) throw new Error("addressable declaration exceeds 64 bits");
	const concurrency = integer$2(plan.concurrency, "concurrency");
	label(plan.backend, "backend");
	label(plan.method, "method");
	array$1(plan.buffers, 1, 1e3);
	const names = /* @__PURE__ */ new Set();
	let total = 0n;
	const components = plan.buffers.map((buffer) => {
		record$1(buffer, [
			"name",
			"role",
			"shape",
			"dtype",
			"count"
		]);
		label(buffer.name, "buffer name");
		if (names.has(buffer.name)) throw new Error("duplicate buffer name");
		names.add(buffer.name);
		if (!roles.includes(buffer.role)) throw new Error("unsupported buffer role");
		if (!Object.hasOwn(itemBytes, buffer.dtype)) throw new Error("unsupported fixed-width dtype");
		array$1(buffer.shape, 1, 64);
		let bytes = itemBytes[buffer.dtype];
		const shape = buffer.shape.map((dimension) => {
			integer$2(dimension, "dimension");
			bytes = product(bytes, dimension, limit);
			return dimension;
		});
		bytes = product(bytes, integer$2(buffer.count, "count"), limit);
		if (bytes > limit - total) throw new Error("declared storage exceeds addressability");
		total += bytes;
		return Object.freeze({
			name: buffer.name,
			role: buffer.role,
			dtype: buffer.dtype,
			count: buffer.count,
			shape: Object.freeze(shape),
			bytes
		});
	});
	const payloadBytes = product(total, concurrency, limit);
	const workUnits = plan.workUnits === null ? null : integer$2(plan.workUnits, "workUnits") * concurrency;
	return Object.freeze({
		backend: plan.backend,
		method: plan.method,
		components: Object.freeze(components),
		concurrency,
		payloadBytes,
		workUnits
	});
}
/** Reproject the complete plan against supplied policy; unknown limits refuse. */
function checkResourcePlan(plan, supplied, requestedWallMs = null) {
	record$1(supplied, [
		"source",
		"addressableBytes",
		"memoryBytes",
		"workUnits",
		"overheadBytes"
	]);
	label(supplied.source, "policy source");
	const limit = integer$2(supplied.addressableBytes, "addressableBytes");
	const memoryBytes = supplied.memoryBytes === null ? null : integer$2(supplied.memoryBytes, "memoryBytes", true);
	const workLimit = supplied.workUnits === null ? null : integer$2(supplied.workUnits, "work ceiling", true);
	const overhead = supplied.overheadBytes === null ? null : integer$2(supplied.overheadBytes, "overheadBytes", true);
	const policy = Object.freeze({
		source: supplied.source,
		addressableBytes: limit,
		memoryBytes,
		workUnits: workLimit,
		overheadBytes: overhead
	});
	const estimate = estimateResourcePlan(plan, limit);
	const blockers = [];
	if (requestedWallMs !== null) {
		integer$2(requestedWallMs, "wall-clock deadline");
		blockers.push("wall_clock_admission_unavailable");
	}
	if (memoryBytes === null) blockers.push("memory_limit_unknown");
	if (workLimit === null || estimate.workUnits === null) blockers.push("work_limit_unknown");
	if (overhead === null) blockers.push("backend_overhead_unknown");
	let bytesRequired = null;
	if (overhead !== null) if (overhead > (limit - estimate.payloadBytes) / estimate.concurrency) blockers.push("total_storage_exceeds_addressability");
	else {
		bytesRequired = estimate.payloadBytes + overhead * estimate.concurrency;
		if (memoryBytes !== null && bytesRequired > memoryBytes) blockers.push("declared_storage_exceeds_budget");
	}
	if (workLimit !== null && estimate.workUnits !== null && estimate.workUnits > workLimit) blockers.push("declared_work_exceeds_budget");
	return Object.freeze({
		allowed: blockers.length === 0,
		estimate,
		policy,
		bytesRequired,
		blockers: Object.freeze(blockers),
		claimBoundary: boundary,
		requestedWallMs
	});
}
var itemBytes, roles, boundary;
var init_admission = __esmMin((() => {
	init_canonical();
	itemBytes = {
		float32: 4n,
		float64: 8n,
		complex64: 8n,
		complex128: 16n,
		uint8: 1n
	};
	roles = [
		"statevector",
		"density",
		"intermediate",
		"adjoint",
		"transfer",
		"graph"
	];
	boundary = "Declared payload and workload policy only; original backend admission remains required. No host capacity, elapsed time, reservation or OOM guarantee.";
}));
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/resources/kuramotoResources.ts
/** Browser product ceilings for declared numeric payload, not discovered free RAM. */
function browserResourcePolicy(bounds) {
	validateBounds(bounds);
	return Object.freeze({
		source: "browser declared numeric-payload ceiling; allocator/object overhead excluded; kernel limits remain mandatory",
		addressableBytes: 4294967295n,
		memoryBytes: 4n * 1024n * 1024n,
		workUnits: 4n * BigInt(bounds.maxSteps) * BigInt(bounds.maxOscillators) ** 2n,
		overheadBytes: 0n
	});
}
function count(value, name) {
	if (!Number.isSafeInteger(value) || value < 1 || value > 4294967295) throw new Error(`${name}: bounded positive count required`);
	return BigInt(value);
}
function validateBounds(bounds) {
	count(bounds.maxOscillators, "kernel oscillator limit");
	count(bounds.maxSteps, "kernel step limit");
}
/** Project the existing Rust RK4 buffer chain; its equations and ABI remain unchanged. */
function admitKuramotoResources(request, bounds, policy = browserResourcePolicy(bounds), requestedWallMs = null) {
	validateBounds(bounds);
	const n = count(request.n, "oscillators");
	const steps = count(request.steps, "steps");
	if (request.n > bounds.maxOscillators || request.steps > bounds.maxSteps) throw new Error("request exceeds declared kernel bounds");
	if (request.mode !== "mean-field" && request.mode !== "networked") throw new Error("unsupported Kuramoto method");
	const matrix = request.mode === "networked" ? n * n : 0n;
	const payload = 2n * n + matrix;
	const output = steps + 1n + n;
	const buffers = [
		{
			name: "caller_numeric_values",
			role: "intermediate",
			shape: [payload],
			dtype: "float64",
			count: 1n
		},
		{
			name: "encoded_host_and_guest_input",
			role: "transfer",
			shape: [32n + 8n * payload],
			dtype: "uint8",
			count: 2n
		},
		{
			name: "parsed_kernel_input",
			role: "intermediate",
			shape: [payload],
			dtype: "float64",
			count: 1n
		},
		{
			name: "theta_rk4_stages",
			role: "intermediate",
			shape: [n],
			dtype: "float64",
			count: 6n
		},
		{
			name: "kernel_result_guest_output_host_result",
			role: "transfer",
			shape: [output],
			dtype: "float64",
			count: 3n
		}
	];
	return checkResourcePlan({
		backend: "shipped-kuramoto-wasm-float64",
		method: request.mode,
		buffers,
		concurrency: 1n,
		workUnits: 4n * steps * (request.mode === "networked" ? n * n : n)
	}, policy, requestedWallMs);
}
/** Admit original numeric buffers plus explicit worker transport copies before allocation. */
function admitOwnedKuramotoResources(request, bounds, binaryBytes, policy = browserResourcePolicy(bounds), requestedWallMs = null) {
	if (!Number.isSafeInteger(binaryBytes) || binaryBytes < 1 || binaryBytes > 2 * 1024 * 1024) throw new Error("bounded worker binary bytes required");
	const original = admitKuramotoResources(request, bounds, policy);
	const n = BigInt(request.n);
	const values = 2n * n + (request.mode === "networked" ? n * n : 0n);
	const buffers = original.estimate.components.map((component) => ({
		name: component.name,
		role: component.role,
		shape: component.shape,
		dtype: component.dtype,
		count: component.count
	}));
	buffers.push({
		name: "owned_worker_transfer_codec_vectors_and_source_request",
		role: "transfer",
		shape: [values],
		dtype: "float64",
		count: 3n
	}, {
		name: "parent_source_codec_validation",
		role: "transfer",
		shape: [32n + 8n * values],
		dtype: "uint8",
		count: 1n
	}, {
		name: "retained_and_transferred_kernel_binary",
		role: "transfer",
		shape: [BigInt(binaryBytes)],
		dtype: "uint8",
		count: 2n
	});
	return checkResourcePlan({
		backend: "shipped-kuramoto-wasm-float64-owned-worker",
		method: request.mode,
		buffers,
		concurrency: 1n,
		workUnits: original.estimate.workUnits
	}, policy, requestedWallMs);
}
var init_kuramotoResources = __esmMin((() => {
	init_admission();
}));
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/workers/kernelProtocol.ts
/** Read a platform getter without consulting a caller's shadow property. */
function nativeGetter(prototype, name) {
	const getter = Object.getOwnPropertyDescriptor(prototype, name)?.get;
	return typeof getter === "function" ? getter : null;
}
/** Inspect native typed-array branding and backing storage across realms. */
function ownedView(value, brand) {
	if (!ArrayBuffer.isView(value)) return false;
	const prototype = Object.getPrototypeOf(Uint8Array.prototype);
	const tag = nativeGetter(prototype, Symbol.toStringTag);
	const backingBuffer = nativeGetter(prototype, "buffer");
	const bufferSize = nativeGetter(ArrayBuffer.prototype, "byteLength");
	if (tag === null || backingBuffer === null || bufferSize === null) return false;
	if (Reflect.apply(tag, value, []) !== brand) return false;
	const buffer = Reflect.apply(backingBuffer, value, []);
	try {
		Reflect.apply(bufferSize, buffer, []);
		return true;
	} catch {
		return false;
	}
}
/** Require an actual float64 view backed by a transferable ArrayBuffer, across native realms. */
function ownedWorkerVector(value) {
	return ownedView(value, "Float64Array");
}
/** Require actual uint8 binary bytes with an owned transferable backing buffer. */
function ownedWorkerBinary(value) {
	return ownedView(value, "Uint8Array");
}
/** Read the actual typed binary byte count without invoking a caller's shadow accessor. */
function kernelBinarySize(bytes) {
	const byteLength = nativeGetter(Object.getPrototypeOf(Uint8Array.prototype), "byteLength");
	if (byteLength === null) throw new Error("native typed-array byte length unavailable");
	return Reflect.apply(byteLength, bytes, []);
}
/** Read only ordinary own data properties; refuse accessors and unknown fields. */
function workerData(value, fields) {
	if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("worker object required");
	const entries = dataEntries(value);
	if (entries.length !== fields.length || entries.some(([key]) => !fields.includes(key))) throw new Error("worker fields differ from v1");
	return Object.fromEntries(entries);
}
/** Require one canonical lower-case source/revision/plan SHA-256 digest. */
function workerDigest(value) {
	if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) throw new Error("canonical SHA-256 required");
	return value;
}
/** Require a bounded, nonempty run identifier without control characters. */
function workerRunId(value) {
	if (typeof value !== "string" || value.length < 1 || value.length > 128 || containsControlCharacters(value)) throw new Error("bounded worker run identity required");
	return value;
}
/** Validate sequence and event shape before any outcome can reach a consumer. */
function readWorkerEvent(value) {
	const raw = workerData(value, [
		"version",
		"run_id",
		"sequence",
		"kind",
		"payload"
	]);
	if (raw["version"] !== 1 || !Number.isSafeInteger(raw["sequence"]) || Number(raw["sequence"]) < 1 || typeof raw["kind"] !== "string" || ![
		"accepted",
		"progress",
		"result",
		"failed",
		"cancelled"
	].includes(raw["kind"])) throw new Error("unsupported worker event");
	const payload = raw["payload"];
	if (typeof payload !== "object" || payload === null || Array.isArray(payload)) throw new Error("worker event payload required");
	dataEntries(payload);
	return {
		version: 1,
		run_id: workerRunId(raw["run_id"]),
		sequence: Number(raw["sequence"]),
		kind: raw["kind"],
		payload
	};
}
var init_kernelProtocol = __esmMin((() => {
	init_canonical();
}));
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/workers/kernelClient.ts
function nativeWorker() {
	const worker = new Worker(new URL(
		/* @vite-ignore */
		"/assets/kernelWorker-BYxnMHNc.js",
		"" + import.meta.url
	), { type: "module" });
	const port = {
		onmessage: null,
		onerror: null,
		postMessage(message, transfers = []) {
			worker.postMessage(message, transfers);
		},
		terminate() {
			return worker.terminate();
		}
	};
	worker.onmessage = (event) => port.onmessage?.({ data: event.data });
	worker.onerror = (event) => port.onerror?.({ message: event.message });
	return port;
}
function finiteVector(value, expectedLength) {
	if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype || Object.hasOwn(value, Symbol.iterator)) throw new Error("ordinary saved numeric vector required");
	if (value.length !== expectedLength) throw new Error("saved vector shape changed after admission");
	const copy = new Float64Array(expectedLength);
	for (let index = 0; index < expectedLength; index++) {
		const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
		if (!descriptor || !("value" in descriptor) || typeof descriptor.value !== "number" || !Number.isFinite(descriptor.value)) throw new Error("finite numeric vector data required");
		copy[index] = descriptor.value;
	}
	return copy;
}
/** Run the original WASM through the frozen v1 transport without changing saved caller state. */
function createOwnedKuramotoRun(options) {
	let worker = null;
	let active = true;
	let settled = false;
	let sequence = 0;
	let accepted = false;
	let timer = null;
	let cleanup = null;
	let pending;
	let identity = null;
	let snapshot = null;
	let terminalEvent = null;
	let resolveOutcome;
	const result = new Promise((resolve) => {
		resolveOutcome = resolve;
	});
	const errorText = (error) => error instanceof Error ? error.message : "owned kernel execution failed";
	const finish = async (outcome) => {
		if (settled) return;
		pending = outcome;
		active = false;
		if (timer !== null) {
			clearTimeout(timer);
			timer = null;
		}
		if (cleanup !== null) {
			await cleanup;
			return;
		}
		cleanup = (async () => {
			let disposed = false;
			let disposalReason = "";
			try {
				if (worker !== null) {
					worker.onmessage = null;
					worker.onerror = null;
					await worker.terminate();
					worker = null;
				}
				disposed = true;
			} catch (error) {
				disposalReason = errorText(error);
			}
			let terminal = pending;
			if (!disposed) terminal = {
				ok: false,
				code: "failed",
				reason: `worker disposal failed: ${disposalReason}`,
				disposed: false
			};
			else if (!terminal.ok) terminal = {
				...terminal,
				disposed: true
			};
			settled = true;
			if (terminal.ok && terminalEvent !== null) try {
				options.onEvent?.({
					...terminalEvent,
					payload: {
						...terminalEvent.payload,
						disposed: true
					}
				});
			} catch (error) {
				terminal = {
					ok: false,
					code: "failed",
					reason: errorText(error),
					disposed: true
				};
			}
			if (!terminal.ok && terminal.code === "cancelled" && identity !== null) try {
				options.onEvent?.({
					version: 1,
					run_id: identity.runId,
					sequence: ++sequence,
					kind: "cancelled",
					payload: {
						revision_hash: identity.revisionHash,
						plan_hash: identity.planHash,
						build_fingerprint: identity.buildFingerprint,
						disposed: true
					}
				});
			} catch (error) {
				terminal = {
					ok: false,
					code: "failed",
					reason: errorText(error),
					disposed: true
				};
			}
			resolveOutcome(terminal);
		})();
		await cleanup;
	};
	const cancel = () => finish({
		ok: false,
		code: "cancelled",
		reason: "owned run cancelled after worker disposal",
		disposed: false
	});
	const handle = {
		result,
		cancel,
		dispose: cancel
	};
	try {
		dataEntries(options);
		identity = {
			runId: workerRunId(options.runId),
			revisionHash: workerDigest(options.revisionHash),
			planHash: workerDigest(options.planHash),
			buildFingerprint: workerDigest(options.buildFingerprint)
		};
		if (!Number.isSafeInteger(options.deadlineMs) || options.deadlineMs < 1 || options.deadlineMs > 6e4) throw new Error("bounded operational worker deadline required");
		if (!ownedWorkerBinary(options.wasmBytes)) throw new Error("bounded original WASM binary is unavailable");
		const binaryBytes = kernelBinarySize(options.wasmBytes);
		if (binaryBytes < 1 || binaryBytes > 2097152) throw new Error("bounded original WASM binary is unavailable");
		dataEntries(options.request);
		dataEntries(options.bounds);
		const request = options.request;
		const bounds = { ...options.bounds };
		const n = request.omega.length;
		const policy = options.resourcePolicy ?? browserResourcePolicy(bounds);
		const admission = admitOwnedKuramotoResources({
			n,
			steps: request.steps,
			mode: request.mode
		}, bounds, binaryBytes, policy);
		if (!admission.allowed) throw new Error(`worker resource policy refused: ${admission.blockers.join(", ")}`);
		snapshot = {
			mode: request.mode,
			omega: finiteVector(request.omega, n),
			theta0: finiteVector(request.theta0, n),
			kNm: finiteVector(request.kNm ?? [], request.mode === "networked" ? n * n : 0),
			steps: request.steps,
			dt: request.dt,
			coupling: request.coupling
		};
		if (encodeKuramotoInput({
			...snapshot,
			omega: Array.from(snapshot.omega),
			theta0: Array.from(snapshot.theta0),
			kNm: Array.from(snapshot.kNm)
		}) === null) throw new Error("original source request refused before worker allocation");
		const binary = new Uint8Array(options.wasmBytes);
		const frozen = identity;
		const input = snapshot;
		worker = (options.workerFactory ?? nativeWorker)();
		timer = setTimeout(() => {
			finish({
				ok: false,
				code: "timeout",
				reason: "owned kernel operational deadline reached",
				disposed: false
			});
		}, options.deadlineMs);
		worker.onerror = (event) => {
			if (active) finish({
				ok: false,
				code: "failed",
				reason: event.message,
				disposed: false
			});
		};
		worker.onmessage = (event) => {
			if (!active) return;
			try {
				const message = readWorkerEvent(event.data);
				if (message.run_id !== frozen.runId || message.sequence <= sequence) return;
				if (message.payload["revision_hash"] !== frozen.revisionHash || message.payload["plan_hash"] !== frozen.planHash) return;
				if (message.kind !== "failed" && message.payload["build_fingerprint"] !== frozen.buildFingerprint) throw new Error("worker build identity differs from the frozen run");
				sequence = message.sequence;
				if (message.kind === "failed") {
					if (typeof message.payload["reason"] !== "string") throw new Error("worker failure reason missing");
					finish({
						ok: false,
						code: "failed",
						reason: message.payload["reason"],
						disposed: false
					});
					return;
				}
				if (message.kind === "cancelled") throw new Error("worker cannot acknowledge host disposal");
				if (message.kind === "accepted") {
					if (accepted) throw new Error("worker accepted the same run twice");
					const limits = message.payload["bounds"];
					if (typeof limits !== "object" || limits === null || Array.isArray(limits)) throw new Error("worker acceptance limits missing");
					const values = Object.fromEntries(dataEntries(limits));
					if (values["maxOscillators"] !== bounds.maxOscillators || values["maxSteps"] !== bounds.maxSteps) throw new Error("worker acceptance limits differ from the frozen run");
					accepted = true;
					options.onEvent?.(message);
					if (active) worker?.postMessage({
						version: 1,
						run_id: frozen.runId,
						revision_hash: frozen.revisionHash,
						plan_hash: frozen.planHash,
						command: "run",
						payload: null
					});
					return;
				}
				if (!accepted) throw new Error("worker event preceded source acceptance");
				if (message.kind === "progress") {
					options.onEvent?.(message);
					return;
				}
				const orderParameter = message.payload["orderParameter"];
				const thetaFinal = message.payload["thetaFinal"];
				if (!ownedWorkerVector(orderParameter) || !ownedWorkerVector(thetaFinal) || orderParameter.length !== input.steps + 1 || thetaFinal.length !== n || !orderParameter.every(Number.isFinite) || !thetaFinal.every(Number.isFinite)) throw new Error("worker source output shape or values refused");
				terminalEvent = message;
				finish({
					...frozen,
					ok: true,
					run: {
						orderParameter,
						thetaFinal
					},
					disposed: true
				});
			} catch (error) {
				finish({
					ok: false,
					code: "failed",
					reason: errorText(error),
					disposed: false
				});
			}
		};
		const payload = {
			wasm: binary,
			build_fingerprint: frozen.buildFingerprint,
			input,
			bounds,
			policy: admission.policy
		};
		worker.postMessage({
			version: 1,
			run_id: frozen.runId,
			revision_hash: frozen.revisionHash,
			plan_hash: frozen.planHash,
			command: "validate",
			payload
		}, [
			binary.buffer,
			input.omega.buffer,
			input.theta0.buffer,
			input.kNm.buffer
		]);
	} catch (error) {
		finish({
			ok: false,
			code: "refused",
			reason: errorText(error),
			disposed: false
		});
	}
	return handle;
}
var init_kernelClient = __esmMin((() => {
	init_kuramoto();
	init_canonical();
	init_kuramotoResources();
	init_kernelProtocol();
}));
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/data/studio/kuramoto_scenario_meanfield_20260708.json
var artifact_id, boundaries, expected, generated_by, input_hex, scenario, schema, kuramoto_scenario_meanfield_20260708_default;
var init_kuramoto_scenario_meanfield_20260708 = __esmMin((() => {
	artifact_id = "studio-kuramoto-scenario-meanfield-20260708";
	boundaries = {
		"max_oscillators": 128,
		"max_steps": 4096
	};
	expected = {
		"order_parameter": [
			.6116919540310324,
			.6186414061153401,
			.625627761031269,
			.6326458560829619,
			.6396903121493714,
			.6467555443321523,
			.6538357744776088,
			.6609250456040002,
			.6680172382443404,
			.6751060886911658,
			.6821852091037786,
			.6892481094105082,
			.6962882209089913,
			.7032989214368274,
			.7102735619538647,
			.7172054943464625,
			.7240881002341679,
			.730914820531136,
			.7376791854891903,
			.7443748449275096,
			.7509955983363934,
			.7575354245301399,
			.7639885105174565,
			.7703492792575214,
			.77661241597618,
			.7827728927299783,
			.7888259909256917,
			.7947673215294853,
			.8005928427322658,
			.8062988748754897,
			.8118821124837028,
			.8173396332953882,
			.8226689042310553,
			.8278677842856786,
			.8329345243803026,
			.8378677642536279,
			.8426665265175229,
			.8473302080396158,
			.8518585688505693,
			.8562517188026156,
			.8605101022289743,
			.8646344808706425,
			.8686259153476602,
			.8724857454564982,
			.8762155695739898,
			.8798172234417403,
			.8832927585937436,
			.8866444206747646,
			.8898746278785631,
			.8929859497140612,
			.8959810862847736,
			.8988628482429915,
			.901634137555949,
			.9042979291971169,
			.9068572538523679,
			.9093151817084519,
			.9116748073703643,
			.9139392359350169,
			.9161115702313339,
			.9181948992215595,
			.9201922875452463,
			.9221067661760711,
			.9239413241522206,
			.9256989013335215,
			.9273823821326309,
			.9289945901632958,
			.9305382837458025,
			.9320161522080749,
			.9334308129203196,
			.9347848090014697,
			.9360806076368045,
			.9373205989478965,
			.9385070953582961,
			.9396423314010295,
			.9407284639169226,
			.9417675725959003,
			.9427616608166511,
			.943712656743348,
			.9446224146413941,
			.9454927163773917,
			.9463252730716679,
			.9471217268747032,
			.9478836528416842,
			.9486125608821097,
			.9493098977639329,
			.9499770491540888,
			.9506153416794598,
			.9512260449943535,
			.9518103738424241,
			.9523694901026567,
			.9529045048105719,
			.9534164801471862,
			.9539064313895121,
			.9543753288174908,
			.9548240995732425,
			.9552536294693894,
			.955664764743991,
			.9560583137602954,
			.956435048650111,
			.9567957069001092,
			.9571409928808176,
			.9574715793184281,
			.9577881087098817,
			.9580911946819471,
			.9583814232952417,
			.9586593542943262,
			.9589255223051548,
			.9591804379812766,
			.9594245891002793,
			.9596584416120311,
			.9598824406403247,
			.960097011439555,
			.960302560308076,
			.9604994754598909,
			.9606881278563052,
			.9608688719991711,
			.9610420466873134,
			.9612079757377006,
			.9613669686728855,
			.9615193213762023,
			.9616653167161596,
			.9618052251414204,
			.9619393052477262,
			.9620678043180517,
			.9621909588372493,
			.9623089949823799,
			.9624221290898821,
			.962530568100685,
			.9626345099843201,
			.9627341441430418,
			.9628296517969224,
			.9629212063508431,
			.9630089737442565,
			.9630931127845601,
			.9631737754648783,
			.9632511072670089,
			.9633252474502617,
			.963396329326871,
			.9634644805246401,
			.9635298232374345,
			.9635924744641197,
			.963652546236499,
			.963710145836788,
			.9637653760051319,
			.9638183351376435,
			.9638691174754226,
			.9639178132849878,
			.9639645090305331,
			.9640092875384019,
			.9640522281541468,
			.964093406892532,
			.9641328965808085,
			.9641707669955858,
			.9642070849935966,
			.9642419146366464,
			.9642753173110188,
			.964307351841593,
			.9643380746009262,
			.9643675396135297,
			.9643957986555611,
			.9644229013501486,
			.9644488952585445,
			.9644738259672998,
			.9644977371716439,
			.9645206707552397,
			.9645426668664822,
			.9645637639914927,
			.9645839990239633,
			.9646034073319881,
			.9646220228220215,
			.9646398780000887,
			.964657004030372,
			.9646734307912928,
			.9646891869291929,
			.9647042999097327,
			.9647187960670959,
			.9647327006511048,
			.9647460378723364,
			.9647588309453226,
			.964771102129928,
			.9647828727709725,
			.9647941633361858,
			.9648049934525611,
			.9648153819411762,
			.9648253468505531,
			.9648349054886154,
			.9648440744533043,
			.9648528696619125,
			.9648613063791905,
			.964869399244276,
			.9648771622964999,
			.9648846090001147,
			.9648917522679906,
			.9648986044843265,
			.9649051775264151,
			.9649114827854999,
			.964917531186771,
			.9649233332085254,
			.9649288989005347,
			.9649342379016512,
			.9649393594566845,
			.9649442724325785,
			.9649489853339195,
			.9649535063178029,
			.9649578432080863,
			.9649620035090526,
			.9649659944185096,
			.9649698228403492,
			.9649734953965886,
			.9649770184389165,
			.9649803980597597,
			.9649836401028993,
			.964986750173646,
			.9649897336485983,
			.9649925956849993,
			.9649953412297086,
			.9649979750278062,
			.965000501630841,
			.965002925404744,
			.9650052505374135,
			.96500748104599,
			.9650096207838341,
			.9650116734472155,
			.9650136425817308,
			.9650155315884559,
			.9650173437298477,
			.9650190821354023,
			.9650207498070851,
			.9650223496245334,
			.9650238843500509,
			.965025356633395,
			.9650267690163684,
			.9650281239372245,
			.9650294237348906,
			.9650306706530202,
			.9650318668438775,
			.9650330143720657,
			.9650341152181012,
			.9650351712818419,
			.965036184385776,
			.9650371562781758,
			.9650380886361232,
			.9650389830684121,
			.9650398411183309,
			.9650406642663323,
			.9650414539325943,
			.9650422114794763,
			.9650429382138749,
			.9650436353894836,
			.965044304208961,
			.9650449458260089,
			.9650455613473679,
			.9650461518347295,
			.9650467183065715,
			.9650472617399177,
			.9650477830720269,
			.9650482832020117,
			.9650487629923933,
			.9650492232705897,
			.9650496648303467,
			.9650500884331085,
			.9650504948093328,
			.9650508846597535,
			.9650512586565888,
			.9650516174447044,
			.9650519616427258,
			.9650522918441071,
			.9650526086181552,
			.9650529125110135,
			.9650532040466046,
			.9650534837275336,
			.9650537520359582,
			.9650540094344185,
			.9650542563666376,
			.9650544932582857,
			.965054720517716,
			.965054938536669,
			.9650551476909488,
			.9650553483410719,
			.9650555408328884,
			.9650557254981793,
			.9650559026552301,
			.9650560726093788,
			.9650562356535426,
			.9650563920687238,
			.9650565421244947,
			.9650566860794628,
			.9650568241817167,
			.9650569566692534,
			.9650570837703902,
			.9650572057041579,
			.9650573226806785,
			.9650574349015278,
			.965057542560083,
			.9650576458418569,
			.9650577449248166,
			.9650578399796913,
			.9650579311702672,
			.9650580186536687,
			.9650581025806304,
			.9650581830957566
		],
		"theta_final": [
			1.0726114007555314,
			1.154008385327912,
			1.2330752458676941,
			1.3104631309006871,
			1.3867155513630016,
			1.4623103322089812,
			1.5376896677910188,
			1.6132844486369986,
			1.6895368690993129,
			1.766924754132306,
			1.8459916146720883,
			1.9273885992444686
		]
	};
	generated_by = "python -m scpn_quantum_control.studio.kuramoto_scenario_artifact --write";
	input_hex = "01000000000000000c0000002c0100007b14ae47e17a943f0000000000000440000000000000f0bf2ebae8a28b2eeabf5d74d145175de4bf185d74d14517ddbf74d145175d74d1bf40175d74d145b7bf40175d74d145b73f74d145175d74d13f185d74d14517dd3f5e74d145175de43f30bae8a28b2eea3f000000000000f03f000000000000000074d145175d74d13f74d145175d74e13f2ebae8a28b2eea3f74d145175d74f13fd145175d74d1f53f2ebae8a28b2efa3f8b2ebae8a28bfe3f74d145175d740140a28b2ebae8a20340d145175d74d105400000000000000840";
	scenario = {
		"coupling": 2.5,
		"dt": .02,
		"mode": "mean-field",
		"n": 12,
		"omega": [
			-1,
			-.8181818181818181,
			-.6363636363636364,
			-.4545454545454546,
			-.2727272727272727,
			-.09090909090909083,
			.09090909090909083,
			.2727272727272727,
			.4545454545454546,
			.6363636363636365,
			.8181818181818183,
			1
		],
		"steps": 300,
		"theta0": [
			0,
			.2727272727272727,
			.5454545454545454,
			.8181818181818181,
			1.0909090909090908,
			1.3636363636363635,
			1.6363636363636362,
			1.909090909090909,
			2.1818181818181817,
			2.454545454545454,
			2.727272727272727,
			3
		]
	};
	schema = "scpn_qc_studio_kuramoto_scenario_v1";
	kuramoto_scenario_meanfield_20260708_default = {
		artifact_id,
		boundaries,
		expected,
		generated_by,
		input_hex,
		scenario,
		schema
	};
}));
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/browser/kuramoto.ts
function isRecord(value) {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}
/** Pack a request into the kernel's canonical little-endian input, or fail closed. */
function encodeKuramotoInput(request) {
	const n = request.omega.length;
	if (n < 1 || request.theta0.length !== n) return null;
	if (!Number.isSafeInteger(request.steps) || request.steps < 1 || request.steps > 4294967295) return null;
	if (!Number.isFinite(request.dt) || request.dt <= 0 || !Number.isFinite(request.coupling)) return null;
	if (request.mode !== "mean-field" && request.mode !== "networked") return null;
	const modeCode = MODE_CODES[request.mode];
	const kLen = request.mode === "networked" ? n * n : 0;
	if (!Number.isSafeInteger(kLen)) return null;
	if (request.mode === "networked" && (request.kNm?.length ?? -1) !== kLen) return null;
	if (kLen === 0 && request.kNm !== void 0 && request.kNm.length !== 0) return null;
	const byteLength = HEADER_LEN + (2 * n + kLen) * 8;
	if (!Number.isSafeInteger(byteLength) || byteLength > 4294967295) return null;
	const fields = [
		request.omega,
		request.theta0,
		request.kNm ?? []
	];
	for (const field of fields) for (const value of field) if (!Number.isFinite(value)) return null;
	try {
		const buffer = new ArrayBuffer(byteLength);
		const view = new DataView(buffer);
		view.setUint32(0, 1, true);
		view.setUint32(4, modeCode, true);
		view.setUint32(8, n, true);
		view.setUint32(12, request.steps, true);
		view.setFloat64(16, request.dt, true);
		view.setFloat64(24, request.coupling, true);
		let offset = HEADER_LEN;
		for (const field of fields) for (const value of field) {
			if (!Number.isFinite(value)) return null;
			view.setFloat64(offset, value, true);
			offset += 8;
		}
		return new Uint8Array(buffer);
	} catch {
		return null;
	}
}
function numberArray(value) {
	if (!Array.isArray(value) || value.some((entry) => typeof entry !== "number")) return null;
	return value;
}
/** Parse the committed scenario artefact into a guarded value (exported for tests). */
function parseScenario(raw) {
	if (!isRecord(raw) || !isRecord(raw["scenario"]) || !isRecord(raw["expected"])) return {
		ok: false,
		reason: "scenario artefact is missing its blocks"
	};
	const scenario = raw["scenario"];
	const boundaries = raw["boundaries"];
	const omega = numberArray(scenario["omega"]);
	const theta0 = numberArray(scenario["theta0"]);
	const expected = numberArray(raw["expected"]["order_parameter"]);
	const mode = scenario["mode"];
	if (omega === null || theta0 === null || expected === null || mode !== "mean-field" && mode !== "networked" || !isRecord(boundaries) || typeof boundaries["max_oscillators"] !== "number" || typeof boundaries["max_steps"] !== "number" || typeof scenario["n"] !== "number" || typeof scenario["steps"] !== "number" || typeof scenario["dt"] !== "number" || typeof scenario["coupling"] !== "number" || typeof raw["artifact_id"] !== "string") return {
		ok: false,
		reason: "scenario artefact fields are malformed"
	};
	return {
		ok: true,
		value: {
			artifactId: raw["artifact_id"],
			boundaries: {
				maxOscillators: boundaries["max_oscillators"],
				maxSteps: boundaries["max_steps"]
			},
			mode,
			n: scenario["n"],
			steps: scenario["steps"],
			dt: scenario["dt"],
			coupling: scenario["coupling"],
			omega,
			theta0,
			expectedOrderParameter: expected
		}
	};
}
var HEADER_LEN, MODE_CODES;
var init_kuramoto = __esmMin((() => {
	init_wasmAssets();
	init_kernelClient();
	init_kuramoto_scenario_meanfield_20260708();
	HEADER_LEN = 32;
	MODE_CODES = {
		"mean-field": 0,
		networked: 1
	};
	parseScenario(kuramoto_scenario_meanfield_20260708_default);
}));
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/contracts/jsonTransport.ts
function boundedText(text) {
	if (text.length > 134217728) throw new Error("$: JSON byte limit exceeded");
	scalarString(text);
	if (new TextEncoder().encode(text).length > 134217728) throw new Error("$: JSON byte limit exceeded");
}
/** Read exact integer and float tokens, refusing duplicate keys and invalid syntax. */
function readJson(text) {
	boundedText(text);
	let position = 0;
	const fail = (reason) => {
		throw new Error(`$ at ${position}: ${reason}`);
	};
	const whitespace = () => {
		while (/[ \t\r\n]/.test(text[position] ?? "")) position++;
	};
	const expect = (character) => {
		whitespace();
		if (text[position] !== character) fail(`expected ${character}`);
		position++;
	};
	const string = () => {
		whitespace();
		if (text[position] !== "\"") fail("string required");
		const start = position++;
		const delimiters = /["\\]/g;
		while (true) {
			delimiters.lastIndex = position;
			const delimiter = delimiters.exec(text);
			if (delimiter === null) {
				position = text.length;
				return fail("unterminated string");
			}
			position = delimiter.index + 1;
			if (delimiter[0] === "\\") {
				position = Math.min(position + 1, text.length);
				continue;
			}
			return scalarString(JSON.parse(text.slice(start, position)));
		}
	};
	const value = (depth) => {
		whitespace();
		const first = text[position];
		if (depth > 64 || depth === 64 && (first === "[" || first === "{")) fail("depth exceeded");
		if (first === "\"") return string();
		if (first === "[") {
			position++;
			const items = [];
			whitespace();
			if (text[position] !== "]") while (true) {
				items.push(value(depth + 1));
				whitespace();
				if (text[position] !== ",") break;
				position++;
			}
			expect("]");
			return items;
		}
		if (first === "{") {
			position++;
			const record = Object.create(null);
			whitespace();
			if (text[position] !== "}") while (true) {
				const key = string();
				if (Object.hasOwn(record, key)) fail("duplicate key");
				expect(":");
				record[key] = value(depth + 1);
				whitespace();
				if (text[position] !== ",") break;
				position++;
			}
			expect("}");
			return record;
		}
		for (const [token, result] of [
			["true", true],
			["false", false],
			["null", null]
		]) if (text.startsWith(token, position)) {
			position += token.length;
			return result;
		}
		const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(text.slice(position));
		if (!match) return fail("value required");
		const token = match[0];
		position += token.length;
		if (token === "-0") return -0;
		if (!/[.eE]/.test(token)) {
			if (token.replace(/^-/, "").length > 4096) fail("integer scalar too large");
			return BigInt(token);
		}
		const number = Number(token);
		if (!Number.isFinite(number)) fail("non-finite float");
		return number;
	};
	const parsed = value(0);
	whitespace();
	if (position !== text.length) fail("trailing input");
	return parsed;
}
function encode$1(value) {
	if (typeof value === "bigint") return value.toString();
	if (typeof value === "number") {
		if (Object.is(value, -0)) return "-0.0";
		const decimal = value.toString();
		return /[.eE]/.test(decimal) ? decimal : `${decimal}.0`;
	}
	if (value === null || typeof value === "boolean" || typeof value === "string") return JSON.stringify(value);
	if (Array.isArray(value)) return `[${value.map(encode$1).join(",")}]`;
	return "{" + dataEntries(value).map(([key, item]) => `${JSON.stringify(key)}:${encode$1(item)}`).join(",") + "}";
}
/** Serialize admitted values with float markers, retaining negative zero and bigint precision. */
function writeJson(value) {
	canonicalBytes("workspace_json.v1", value);
	const text = encode$1(value);
	boundedText(text);
	return text;
}
var init_jsonTransport = __esmMin((() => {
	init_canonical();
}));
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/contracts/workspace.ts
function fail(path, message) {
	throw new ContractError(path, message);
}
function full(pattern, text) {
	return pattern.exec(text)?.[0] === text;
}
function object$7(value, path) {
	if (typeof value !== "object" || value === null || Array.isArray(value)) return fail(path, "object required");
	return Object.fromEntries(dataEntries(value, path));
}
function keys(value, required, path, optional = []) {
	if (required.some((key) => !Object.hasOwn(value, key)) || Object.keys(value).some((key) => !required.includes(key) && !optional.includes(key))) fail(path, "missing or unknown field");
}
function text$3(value, path) {
	if (typeof value !== "string" || Array.from(value).every((character) => blankCharacters.includes(character))) return fail(path, "nonempty string required");
	return value;
}
function list(value, path) {
	if (!Array.isArray(value)) return fail(path, "array required");
	return [...value];
}
function strings(value, path) {
	const result = list(value, path).map((item) => text$3(item, path));
	if (new Set(result).size !== result.length) fail(path, "duplicate entry");
	return result;
}
function integer$1(value, path) {
	if (typeof value === "number" && Number.isSafeInteger(value) && !Object.is(value, -0)) value = BigInt(value);
	if (typeof value !== "bigint" || value < 0n || value > safeInteger) return fail(path, "safe nonnegative integer required");
	return value;
}
function shape(value, path) {
	const dimensions = list(value, path).map((item) => integer$1(item, path));
	let size = dimensions.includes(0n) ? 0n : 1n;
	for (const dimension of dimensions) {
		size *= dimension;
		if (size > safeInteger) fail(path, "shape product overflow");
	}
	return {
		dimensions,
		size
	};
}
function hash$1(value, path) {
	const result = text$3(value, path);
	if (!full(/[0-9a-f]{64}/, result)) fail(path, "lowercase SHA-256 required");
	return result;
}
function uuid(value, path) {
	const result = text$3(value, path);
	if (!full(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/, result)) fail(path, "canonical UUID required");
	return result;
}
function reference(value, path) {
	const ref = object$7(value, path);
	keys(ref, [
		"schema",
		"sha256",
		"media_type"
	], path, ["name"]);
	if (!full(/[A-Za-z0-9_.-]+\.v[1-9][0-9]*/, text$3(ref["schema"], path))) fail(`${path}.schema`, "versioned schema required");
	hash$1(ref["sha256"], `${path}.sha256`);
	if (!full(/[A-Za-z0-9.+-]+\/[A-Za-z0-9.+-]+/, text$3(ref["media_type"], path))) fail(`${path}.media_type`, "media type required");
	if ("name" in ref) {
		const name = text$3(ref["name"], `${path}.name`);
		if (/[\\:%?#@]/.test(name) || Array.from(name).some((character) => character.charCodeAt(0) < 32) || name.split("/").some((part) => [
			"",
			".",
			".."
		].includes(part))) fail(`${path}.name`, "safe project-relative name required");
	}
	return ref;
}
function references(value, path) {
	const refs = list(value, path).map((item) => reference(item, path));
	if (new Set(refs.map((ref) => `${String(ref["schema"])}:${String(ref["sha256"])}`)).size !== refs.length) fail(path, "duplicate reference");
	return refs;
}
function dtype(value, path) {
	const result = text$3(value, path);
	if (![
		"float64",
		"int64",
		"uint64"
	].includes(result)) fail(path, "unsupported dtype");
	return result;
}
function element$1(value, kind, path) {
	const literal = text$3(value, path);
	if (kind === "float64") {
		if (!full(/[0-9a-f]{16}/, literal)) fail(path, "16-digit lowercase IEEE hex required");
		const bytes = Uint8Array.from({ length: 8 }, (_, index) => parseInt(literal.slice(2 * index, 2 * index + 2), 16));
		const result = new DataView(bytes.buffer).getFloat64(0, false);
		if (!Number.isFinite(result)) fail(path, "non-finite element");
		return result;
	}
	if (!full(/0|-?[1-9][0-9]{0,19}/, literal)) fail(path, "canonical bounded decimal required");
	const result = BigInt(literal);
	const [lower, upper] = kind === "int64" ? [-(2n ** 63n), 2n ** 63n - 1n] : [0n, 2n ** 64n - 1n];
	if (result < lower || result > upper) fail(path, "integer element overflow");
	return result;
}
function typed(value, path) {
	const result = object$7(value, path);
	keys(result, [
		"dtype",
		"shape",
		"values"
	], path);
	const kind = dtype(result["dtype"], `${path}.dtype`);
	const dimensions = shape(result["shape"], `${path}.shape`);
	const values = list(result["values"], `${path}.values`);
	if (BigInt(values.length) !== dimensions.size) fail(path, "shape/value cardinality mismatch");
	values.forEach((item, index) => {
		element$1(item, kind, `${path}.values[${index}]`);
	});
	return {
		dtype: kind,
		shape: dimensions.dimensions,
		values
	};
}
function domain(value, kind, path) {
	const result = object$7(value, path);
	if (result["kind"] === "finite") keys(result, ["kind"], path);
	else if (result["kind"] === "closed_interval") {
		keys(result, [
			"kind",
			"lower",
			"upper"
		], path);
		if (element$1(result["lower"], kind, path) > element$1(result["upper"], kind, path)) fail(path, "reversed interval");
	} else if (result["kind"] === "enumerated") {
		keys(result, ["kind", "values"], path);
		const values = strings(result["values"], path);
		if (values.length === 0) fail(path, "empty domain");
		values.forEach((item) => {
			element$1(item, kind, path);
		});
	} else fail(path, "unsupported domain");
	return result;
}
function timestamp(value, path) {
	const literal = text$3(value, path);
	const match = /([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,9}))?Z/.exec(literal);
	if (!match || match[0] !== literal) return fail(path, "UTC timestamp required");
	const year = Number(match[1]), month = Number(match[2]), day = Number(match[3]);
	const hour = Number(match[4]), minute = Number(match[5]), second = Number(match[6]);
	const date = /* @__PURE__ */ new Date(0);
	date.setUTCFullYear(year, month - 1, day);
	date.setUTCHours(hour, minute, second, 0);
	if (year < 1 || date.getUTCFullYear() !== year || date.getUTCMonth() !== month - 1 || date.getUTCDate() !== day || date.getUTCHours() !== hour || date.getUTCMinutes() !== minute || date.getUTCSeconds() !== second) fail(path, "invalid calendar timestamp");
	return `${literal.slice(0, 19)}.${(match[7] ?? "").padEnd(9, "0")}`;
}
function validate(schema, body) {
	keys(body, fields[schema], "$.body");
	if (schema === "quantum_workspace.v1" || schema === "experiment_revision.v1") uuid(body["project_id"], "$.body.project_id");
	if (schema === "quantum_workspace.v1") {
		body["revision_refs"] = references(body["revision_refs"], "$.body.revision_refs");
		body["artefact_refs"] = references(body["artefact_refs"], "$.body.artefact_refs");
		if (body["draft_ref"] !== null) body["draft_ref"] = reference(body["draft_ref"], "$.body.draft_ref");
		if (timestamp(body["updated_at"], "$.body.updated_at") < timestamp(body["created_at"], "$.body.created_at")) fail("$.body.updated_at", "precedes creation");
	} else if (schema === "experiment_revision.v1") {
		strings(body["parent_revision_hashes"], "$.body.parent_revision_hashes").forEach((parent) => {
			hash$1(parent, "$.body.parent_revision_hashes");
		});
		for (const key of [
			"problem_ref",
			"program_ref",
			"semantic_settings_ref"
		]) body[key] = reference(body[key], `$.body.${key}`);
		body["input_refs"] = references(body["input_refs"], "$.body.input_refs");
		const parameters = object$7(body["parameters"], "$.body.parameters");
		body["parameters"] = Object.fromEntries(Object.entries(parameters).map(([key, item]) => [text$3(key, "$.body.parameters"), typed(item, `$.body.parameters.${key}`)]));
	} else if (schema === "parameter_spec.v1") {
		for (const key of [
			"key",
			"unit",
			"default_source"
		]) text$3(body[key], `$.body.${key}`);
		const kind = dtype(body["dtype"], "$.body.dtype");
		body["shape"] = shape(body["shape"], "$.body.shape").dimensions;
		body["domain"] = domain(body["domain"], kind, "$.body.domain");
		if (typeof body["trainable"] !== "boolean") fail("$.body.trainable", "boolean required");
		strings(body["dependency_keys"], "$.body.dependency_keys");
	} else if (schema === "resolved_settings.v1") {
		object$7(body["requested"], "$.body.requested");
		const effective = object$7(body["effective"], "$.body.effective");
		const origins = object$7(body["origins"], "$.body.origins");
		if (Object.keys(origins).length !== Object.keys(effective).length || Object.keys(effective).some((key) => !Object.hasOwn(origins, key))) fail("$.body.origins", "must cover effective fields exactly");
		for (const [key, origin] of Object.entries(origins)) if (typeof origin === "string") text$3(origin, `$.body.origins.${key}`);
		else if (Object.keys(object$7(origin, `$.body.origins.${key}`)).length === 0) fail("$.body.origins", "empty provenance");
		for (const key of ["policy_ref", "environment_ref"]) body[key] = reference(body[key], `$.body.${key}`);
		strings(body["rejected_fields"], "$.body.rejected_fields");
	} else {
		for (const key of ["run_id", "attempt_id"]) uuid(body[key], `$.body.${key}`);
		for (const key of ["revision_hash", "plan_hash"]) hash$1(body[key], `$.body.${key}`);
		if (body["mode"] !== "local") fail("$.body.mode", "unsupported mode");
		let previous = -1n;
		body["events"] = list(body["events"], "$.body.events").map((raw) => {
			const event = object$7(raw, "$.body.events");
			keys(event, [
				"version",
				"run_id",
				"sequence",
				"kind",
				"payload"
			], "$.body.events");
			const version = integer$1(event["version"], "$.body.events.version");
			const sequence = integer$1(event["sequence"], "$.body.events.sequence");
			if (version !== 1n || event["run_id"] !== body["run_id"] || typeof event["kind"] !== "string" || ![
				"accepted",
				"progress",
				"result",
				"failed",
				"cancelled"
			].includes(event["kind"])) fail("$.body.events", "version, run or event kind mismatch");
			if (sequence <= previous) fail("$.body.events.sequence", "must increase");
			previous = sequence;
			object$7(event["payload"], "$.body.events.payload");
			return {
				...event,
				version,
				sequence
			};
		});
		body["output_refs"] = references(body["output_refs"], "$.body.output_refs");
	}
}
function snapshot(value, freeze) {
	if (typeof value !== "object" || value === null) return value;
	const result = Array.isArray(value) ? value.map((item) => snapshot(item, freeze)) : Object.fromEntries(dataEntries(value).map(([key, item]) => [key, snapshot(item, freeze)]));
	return freeze ? Object.freeze(result) : result;
}
function refusal(error) {
	return {
		ok: false,
		code: "invalid_document",
		path: error instanceof ContractError ? error.path : "$",
		message: error instanceof Error ? error.message : "Document refused"
	};
}
/** Parse one of five envelopes and freeze its recursive structural snapshot.
* A snapshot created by this parser retains its immutable identity on readmission.
* Caller objects and wire exports always receive complete validation; no graph or
* producer admission is inferred from a structurally parsed snapshot.
*/
function parseDocument(payload) {
	const known = typeof payload === "object" && payload !== null ? parsedDocuments.get(payload) : void 0;
	if (known !== void 0) return {
		ok: true,
		value: known
	};
	try {
		canonicalBytes("workspace_input.v1", payload);
		const record = object$7(payload, "$");
		keys(record, [
			"schema",
			"body",
			"extensions"
		], "$");
		const schema = text$3(record["schema"], "$.schema");
		if (!Object.hasOwn(fields, schema)) fail("$.schema", "unsupported workspace schema");
		const body = object$7(record["body"], "$.body");
		const extensions = object$7(record["extensions"], "$.extensions");
		validate(schema, body);
		const value = snapshot({
			schema,
			body,
			extensions
		}, true);
		parsedDocuments.set(value, value);
		return {
			ok: true,
			value
		};
	} catch (error) {
		return refusal(error);
	}
}
function named(schema, payload) {
	const parsed = parseDocument(payload);
	if (!parsed.ok) return parsed;
	if (parsed.value.schema !== schema) return {
		ok: false,
		code: "schema_mismatch",
		path: "$.schema",
		message: `${schema} required`
	};
	return {
		ok: true,
		value: parsed.value
	};
}
/** Parse a workspace root without claiming reference admission. */
function parseWorkspaceManifest(payload) {
	return named("quantum_workspace.v1", payload);
}
/** Parse immutable experiment inputs without resolving parents. */
function parseExperimentRevision(payload) {
	return named("experiment_revision.v1", payload);
}
/** Parse a typed parameter's exact unit, shape and domain. */
function parseParameterSpec(payload) {
	return named("parameter_spec.v1", payload);
}
/** Parse recorded setting provenance without executing policy. */
function parseResolvedSettings(payload) {
	return named("resolved_settings.v1", payload);
}
/** Parse local event metadata without a successful-run claim. */
function parseLocalRunRecord(payload) {
	return named("local_run_record.v1", payload);
}
/** Revalidate and address a whole immutable document, including its extensions. */
function documentDigest(document) {
	const parsed = parseDocument(document);
	if (!parsed.ok) throw new Error(`${parsed.path}: ${parsed.message}`);
	return canonicalDigest(parsed.value.schema, parsed.value);
}
/** Check supplied typed values against a spec without implicit unit conversion. */
function validateParameterBinding(spec, payload, unit) {
	try {
		const parsed = parseParameterSpec(spec);
		if (!parsed.ok) return parsed;
		const values = typed(payload, "$.parameters");
		const expected = parsed.value.body;
		const dimensions = values["shape"];
		const expectedDimensions = expected["shape"];
		if (unit !== expected["unit"] || values["dtype"] !== expected["dtype"] || dimensions.length !== expectedDimensions.length || dimensions.some((value, index) => value !== expectedDimensions[index])) fail("$.parameters", "dtype, shape or unit mismatch");
		const kind = values["dtype"];
		const restriction = expected["domain"];
		for (const value of values["values"]) {
			const number = element$1(value, kind, "$.parameters.values");
			if (restriction["kind"] === "closed_interval" && (number < element$1(restriction["lower"], kind, "$.domain.lower") || number > element$1(restriction["upper"], kind, "$.domain.upper"))) fail("$.parameters.values", "outside interval");
			if (restriction["kind"] === "enumerated" && !restriction["values"].includes(value)) fail("$.parameters.values", "outside enumerated domain");
		}
		return {
			ok: true,
			value: null
		};
	} catch (error) {
		return refusal(error);
	}
}
var workspaceSchemas, fields, parsedDocuments, blankCharacters, safeInteger, ContractError;
var init_workspace = __esmMin((() => {
	init_canonical();
	workspaceSchemas = Object.freeze([
		"quantum_workspace.v1",
		"experiment_revision.v1",
		"parameter_spec.v1",
		"resolved_settings.v1",
		"local_run_record.v1"
	]);
	fields = {
		"quantum_workspace.v1": [
			"project_id",
			"revision_refs",
			"draft_ref",
			"created_at",
			"updated_at",
			"artefact_refs"
		],
		"experiment_revision.v1": [
			"project_id",
			"parent_revision_hashes",
			"problem_ref",
			"program_ref",
			"parameters",
			"semantic_settings_ref",
			"input_refs"
		],
		"parameter_spec.v1": [
			"key",
			"dtype",
			"shape",
			"unit",
			"domain",
			"default_source",
			"trainable",
			"dependency_keys"
		],
		"resolved_settings.v1": [
			"requested",
			"effective",
			"origins",
			"policy_ref",
			"environment_ref",
			"rejected_fields"
		],
		"local_run_record.v1": [
			"run_id",
			"attempt_id",
			"revision_hash",
			"plan_hash",
			"mode",
			"events",
			"output_refs"
		]
	};
	parsedDocuments = /* @__PURE__ */ new WeakMap();
	blankCharacters = "	\n\v\f\r \xA0            \u2028\u2029  　";
	safeInteger = 9007199254740991n;
	ContractError = class extends Error {
		path;
		constructor(path, message) {
			super(message);
			this.path = path;
		}
	};
}));
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/contracts/graph.ts
function take$3(result) {
	if (!result.ok) throw new Error(`${result.path}: ${result.message}`);
	return result.value;
}
function acyclic(edges, path) {
	const degrees = new Map([...edges].map(([key, parents]) => [key, new Set(parents).size]));
	const children = new Map([...edges.keys()].map((key) => [key, []]));
	for (const [key, parents] of edges) for (const parent of new Set(parents)) {
		const descendants = children.get(parent);
		if (!descendants) throw new Error(`${path}: dangling dependency ${parent}`);
		descendants.push(key);
	}
	const ready = [...degrees].filter(([, degree]) => degree === 0).map(([key]) => key);
	let visited = 0;
	while (ready.length > 0) {
		const key = ready.pop();
		visited++;
		for (const child of children.get(key)) {
			const degree = degrees.get(child) - 1;
			degrees.set(child, degree);
			if (degree === 0) ready.push(child);
		}
	}
	if (visited !== edges.size) throw new Error(`${path}: dependency cycle`);
}
/** Admit complete reference indexes without fetching, writing or launching a worker. */
async function admitWorkspace(manifest, documents, rawRecords, rawCodecs, parameterSpecs, parameterUnits) {
	try {
		const root = take$3(parseWorkspaceManifest(manifest));
		const projectId = root.body["project_id"];
		const index = new Map([...documents].map(([digest, value]) => [digest, take$3(parseDocument(value))]));
		const rawSnapshot = new Map([...rawRecords].map(([digest, value]) => [digest, {
			schema: value.schema,
			content: new Uint8Array(value.content)
		}]));
		const specs = new Map([...parameterSpecs].map(([key, value]) => [key, take$3(parseParameterSpec(value))]));
		const units = new Map(parameterUnits);
		const codecs = new Map(rawCodecs);
		const rootDigest = await documentDigest(root);
		for (const [digest, record] of index) {
			if (await documentDigest(record) !== digest) throw new Error("$.documents: digest mismatch");
			if (record.schema === "quantum_workspace.v1" && digest !== rootDigest) throw new Error("$.documents: unexpected workspace root");
			if (record.schema === "experiment_revision.v1" && record.body["project_id"] !== projectId) throw new Error("$.documents: cross-project revision");
			if (rawSnapshot.has(digest)) throw new Error("$.documents: ambiguous raw/document identity");
		}
		if (specs.size !== units.size || [...specs.keys()].some((key) => !units.has(key))) throw new Error("$.parameter_units: keys must match specifications");
		const specDigests = /* @__PURE__ */ new Map();
		for (const [key, spec] of specs) {
			const digest = await documentDigest(spec);
			if (key !== spec.body["key"] || index.get(digest)?.schema !== "parameter_spec.v1") throw new Error("$.parameter_specs: key or indexed identity mismatch");
			specDigests.set(key, digest);
		}
		acyclic(new Map([...specs].map(([key, spec]) => [key, spec.body["dependency_keys"]])), "$.parameter_specs");
		const verified = /* @__PURE__ */ new Map();
		const rawIdentity = async (digest) => {
			const cached = verified.get(digest);
			if (cached) return cached;
			const artifact = rawSnapshot.get(digest);
			if (!artifact) throw new Error("$.references: dangling raw reference");
			const codec = codecs.get(artifact.schema);
			if (!codec) throw new Error("$.references: unsupported raw producer/schema");
			const identity = await codec(new Uint8Array(artifact.content));
			if (identity.schema !== artifact.schema || identity.digest !== digest || typeof identity.kind !== "string" || identity.kind === "") throw new Error("$.references: raw producer identity mismatch");
			const snapshot = Object.freeze({
				schema: identity.schema,
				kind: identity.kind,
				digest: identity.digest
			});
			verified.set(digest, snapshot);
			return snapshot;
		};
		const resolve = async (ref, expected) => {
			const digest = ref["sha256"];
			const document = index.get(digest);
			if (!document && workspaceSchemas.some((schema) => schema === ref["schema"])) throw new Error("$.references: workspace reference requires indexed document");
			const identity = document ? {
				schema: document.schema,
				kind: document.schema
			} : await rawIdentity(digest);
			if (identity.schema !== ref["schema"] || expected !== void 0 && identity.kind !== expected) throw new Error("$.references: schema or kind mismatch");
		};
		for (const ref of root.body["revision_refs"]) await resolve(ref, "experiment_revision.v1");
		for (const ref of root.body["artefact_refs"]) await resolve(ref);
		if (root.body["draft_ref"] !== null) await resolve(root.body["draft_ref"], "experiment_revision.v1");
		const parents = /* @__PURE__ */ new Map();
		for (const [digest, record] of index) {
			const body = record.body;
			if (record.schema === "experiment_revision.v1") {
				parents.set(digest, body["parent_revision_hashes"]);
				await resolve(body["problem_ref"], "problem");
				await resolve(body["program_ref"], "program");
				await resolve(body["semantic_settings_ref"], "resolved_settings.v1");
				const inputs = body["input_refs"];
				for (const ref of inputs) await resolve(ref);
				const inputHashes = new Set(inputs.map((ref) => ref["sha256"]));
				for (const [key, values] of Object.entries(body["parameters"])) {
					const spec = specs.get(key);
					if (!spec || !inputHashes.has(specDigests.get(key))) throw new Error("$.parameters: missing immutable specification reference");
					take$3(validateParameterBinding(spec, values, units.get(key)));
				}
			} else if (record.schema === "resolved_settings.v1") {
				await resolve(body["policy_ref"], "policy");
				await resolve(body["environment_ref"], "environment");
			} else if (record.schema === "local_run_record.v1") {
				if (index.get(body["revision_hash"])?.schema !== "experiment_revision.v1") throw new Error("$.run.revision_hash: missing revision");
				if ((await rawIdentity(body["plan_hash"])).kind !== "plan") throw new Error("$.run.plan_hash: wrong raw kind");
				for (const ref of body["output_refs"]) await resolve(ref);
			}
		}
		acyclic(parents, "$.revisions");
		return {
			ok: true,
			value: Object.freeze({
				projectId,
				workspaceHash: rootDigest,
				documentHashes: Object.freeze([...index.keys()].sort()),
				rawHashes: Object.freeze([...verified.keys()].sort())
			})
		};
	} catch (error) {
		return {
			ok: false,
			code: "invalid_graph",
			path: "$",
			message: error instanceof Error ? error.message : "Workspace admission refused"
		};
	}
}
var init_graph = __esmMin((() => {
	init_workspace();
}));
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/contracts/index.ts
var init_contracts = __esmMin((() => {
	init_canonical();
	init_jsonTransport();
	init_workspace();
	init_graph();
}));
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/storage/workspaceArchive.ts
init_graph();
init_contracts();
/** Maximum decoded member bytes retained by the portable archive. */
var maxExpandedBytes = 128 * 1024 * 1024;
/** Canonical string domain for an exact archive snapshot, including its source formatting. */
var archiveSnapshotDomain = "quantum_workspace_archive_source.v1";
function take$2(value) {
	if (!value.ok) throw new Error(`${value.path}: ${value.message}`);
	return value.value;
}
function record(value, keys, name) {
	if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`${name}: object required`);
	const result = value;
	if (Object.keys(result).length !== keys.length || keys.some((key) => !Object.hasOwn(result, key))) throw new Error(`${name} fields are incomplete or unsupported`);
	return result;
}
function text$2(value, name) {
	if (typeof value !== "string" || value.length === 0) throw new Error(`${name}: nonempty string required`);
	return value;
}
function safeName(value) {
	const name = text$2(value, "member name");
	if (name.length > 4096 || /[\\:%?#@]/.test(name) || Array.from(name).some((character) => character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127) || name.split("/").some((part) => [
		"",
		".",
		".."
	].includes(part))) throw new Error("unsafe member name");
	return name;
}
function digest$1(value) {
	const result = text$2(value, "member digest");
	if (result.length !== 64 || !/^[0-9a-f]{64}$/.test(result)) throw new Error("lowercase SHA-256 member digest required");
	return result;
}
/** Preview bounded JSON and admit every reference; callers may lower the expanded byte budget. */
async function admitWorkspaceArchive(json, rawCodecs = /* @__PURE__ */ new Map(), expandedByteLimit = maxExpandedBytes) {
	if (!Number.isSafeInteger(expandedByteLimit) || expandedByteLimit <= 0 || expandedByteLimit > 134217728) throw new Error("expanded byte budget must be a positive integer within the product bound");
	if (typeof json !== "string" || json.length > 67108864 || new TextEncoder().encode(json).byteLength > 67108864) throw new Error("encoded archive limit exceeded");
	const archive = record(readJson(json), [
		"schema",
		"manifest",
		"members",
		"parameter_units"
	], "archive");
	if (archive["schema"] !== "quantum_workspace_archive.v1") throw new Error("Unsupported archive schema; explicit supported migration is required before import");
	const manifest = take$2(parseWorkspaceManifest(archive["manifest"]));
	const members = archive["members"];
	if (!Array.isArray(members) || members.length + 1 > 1e3) throw new Error("archive member limit exceeded");
	const names = /* @__PURE__ */ new Set();
	const hashes = /* @__PURE__ */ new Set();
	const documents = /* @__PURE__ */ new Map();
	const raw = /* @__PURE__ */ new Map();
	const originalCodecs = new Map(rawCodecs);
	let expanded = new TextEncoder().encode(writeJson(manifest)).byteLength;
	if (expanded > expandedByteLimit) throw new Error("expanded archive limit exceeded");
	const entries = members.map((value) => {
		const member = record(value, [
			"name",
			"kind",
			"schema",
			"sha256",
			"content"
		], "member");
		const name = safeName(member["name"]);
		if (names.has(name)) throw new Error("duplicate archive member name");
		names.add(name);
		const hash = digest$1(member["sha256"]);
		if (hashes.has(hash)) throw new Error("duplicate archive identity");
		hashes.add(hash);
		const schema = text$2(member["schema"], "member schema");
		const content = member["content"];
		if (typeof content !== "string") throw new Error("member content: string required");
		const kind = member["kind"];
		if (kind !== "document" && kind !== "raw") throw new Error("unsupported archive member kind; executable members and links are refused");
		if (kind === "raw" && (content.length % 2 !== 0 || /[^0-9a-f]/.test(content))) throw new Error("canonical lowercase hex bytes required");
		expanded += kind === "raw" ? content.length / 2 : new TextEncoder().encode(content).byteLength;
		if (expanded > expandedByteLimit) throw new Error("expanded archive limit exceeded");
		return Object.freeze({
			name,
			sha256: hash,
			schema,
			kind,
			content
		});
	});
	for (const entry of entries) if (entry.kind === "document") {
		const document = take$2(parseDocument(readJson(entry.content)));
		if (document.schema !== entry.schema) throw new Error("member document schema mismatch");
		documents.set(entry.sha256, document);
	} else {
		const nativeDecoder = Uint8Array.fromHex;
		const content = typeof nativeDecoder === "function" ? nativeDecoder(entry.content) : new Uint8Array(entry.content.length / 2);
		if (typeof nativeDecoder !== "function") for (let index = 0; index < content.length; index++) content[index] = parseInt(entry.content.slice(2 * index, 2 * index + 2), 16);
		raw.set(entry.sha256, {
			schema: entry.schema,
			content
		});
		const codec = originalCodecs.get(entry.schema);
		if (!codec) throw new Error("unsupported raw producer/schema");
		const identity = await codec(new Uint8Array(content));
		if (identity.digest !== entry.sha256 || identity.schema !== entry.schema || typeof identity.kind !== "string" || identity.kind.length === 0) throw new Error("raw producer identity mismatch");
	}
	const specs = /* @__PURE__ */ new Map();
	for (const document of documents.values()) {
		if (document.schema !== "parameter_spec.v1") continue;
		const spec = take$2(parseParameterSpec(document));
		const key = spec.body["key"];
		if (specs.has(key)) throw new Error("duplicate parameter specification key");
		specs.set(key, spec);
	}
	const unitPayload = archive["parameter_units"];
	if (typeof unitPayload !== "object" || unitPayload === null || Array.isArray(unitPayload)) throw new Error("parameter units object required");
	const units = new Map(Object.entries(unitPayload).map(([key, value]) => [key, text$2(value, "parameter unit")]));
	const admission = take$2(await admitWorkspace(manifest, documents, raw, originalCodecs, specs, units));
	const preview = Object.freeze({
		schema: "quantum_workspace_archive.v1",
		projectId: admission.projectId,
		archiveDigest: await canonicalDigest(archiveSnapshotDomain, json),
		workspaceHash: admission.workspaceHash,
		documentHashes: admission.documentHashes,
		rawHashes: Object.freeze([...raw.keys()].sort()),
		memberNames: Object.freeze([...names]),
		json
	});
	const revisions = [...documents].filter((entry) => entry[1].schema === "experiment_revision.v1");
	return Object.freeze({
		preview,
		manifest,
		members: Object.freeze(entries),
		documents: Object.freeze(Object.fromEntries(documents)),
		parameterUnits: Object.freeze(Object.fromEntries(units)),
		revisions: Object.freeze(Object.fromEntries(revisions)),
		parameterSpecs: Object.freeze(Object.fromEntries(specs))
	});
}
/** Preview the original bounded archive without exposing editing inputs or execution authority. */
async function previewWorkspaceArchive(json, rawCodecs = /* @__PURE__ */ new Map(), expandedByteLimit = maxExpandedBytes) {
	return (await admitWorkspaceArchive(json, rawCodecs, expandedByteLimit)).preview;
}
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/experiments/kuramotoArtifacts.ts
/** Exact raw source digest; no workspace canonical re-encoding changes its identity. */
async function artifactBytesDigest(bytes) {
	const digest = await crypto.subtle.digest("SHA-256", new Uint8Array(bytes));
	return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
/** Convert a finite backend scalar to the workspace's original big-endian IEEE spelling. */
function encodeFloat64(value) {
	if (!Number.isFinite(value)) throw new ExperimentRefusal("finite float64 artifact value required");
	const bytes = /* @__PURE__ */ new Uint8Array(8);
	new DataView(bytes.buffer).setFloat64(0, value, false);
	return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}
/** Decode an exact finite scalar without changing its signed zero or source unit. */
function decodeFloat64(value) {
	if (typeof value !== "string" || !/^[0-9a-f]{16}$/.test(value)) throw new ExperimentRefusal("16 lowercase IEEE float64 digits required");
	const bytes = Uint8Array.from({ length: 8 }, (_, index) => parseInt(value.slice(2 * index, 2 * index + 2), 16));
	const number = new DataView(bytes.buffer).getFloat64(0, false);
	if (!Number.isFinite(number)) throw new ExperimentRefusal("finite float64 artifact value required");
	return number;
}
/** Decode the original 32-byte little-endian input with bounded vectors, without solving. */
function decodeKernelInput(content) {
	if (content.byteLength < 32 || content.byteLength > byteLimit) throw new ExperimentRefusal("bounded complete kernel input required");
	const bytes = new Uint8Array(content);
	const view = new DataView(bytes.buffer);
	const version = view.getUint32(0, true), mode = view.getUint32(4, true), n = view.getUint32(8, true), steps = view.getUint32(12, true);
	if (version !== 1 || mode > 1 || n < 1 || n > 128 || steps < 1 || steps > 4096) throw new ExperimentRefusal("unsupported kernel version, method or browser input shape");
	const matrix = mode === 1 ? n * n : 0;
	if (bytes.byteLength !== 32 + 8 * (2 * n + matrix)) throw new ExperimentRefusal("kernel input byte shape mismatch");
	const vector = (start, length) => Object.freeze(Array.from({ length }, (_, index) => view.getFloat64(32 + 8 * (start + index), true)));
	const request = Object.freeze({
		mode: mode === 0 ? "mean-field" : "networked",
		omega: vector(0, n),
		theta0: vector(n, n),
		steps,
		dt: view.getFloat64(16, true),
		coupling: view.getFloat64(24, true),
		...mode === 1 ? { kNm: vector(2 * n, matrix) } : {}
	});
	if (encodeKuramotoInput(request) === null) throw new ExperimentRefusal("original kernel input admission refused");
	return request;
}
function object$6(value, keys) {
	if (typeof value !== "object" || value === null || Array.isArray(value)) throw new ExperimentRefusal("artifact object required");
	const entries = dataEntries(value);
	if (entries.length !== keys.length || entries.some(([key]) => !keys.includes(key))) throw new ExperimentRefusal("unsupported or missing artifact fields");
	return Object.fromEntries(entries);
}
function hash(value) {
	if (typeof value !== "string" || value.length !== 64 || !/^[0-9a-f]{64}$/.test(value)) throw new ExperimentRefusal("exact lowercase source SHA256 required");
	return value;
}
function bounds(value) {
	const fields = object$6(value, ["maxOscillators", "maxSteps"]);
	const result = {
		maxOscillators: fields["maxOscillators"],
		maxSteps: fields["maxSteps"]
	};
	browserResourcePolicy(result);
	return Object.freeze(result);
}
function policy(value) {
	return object$6(value, [
		"source",
		"addressableBytes",
		"memoryBytes",
		"workUnits",
		"overheadBytes"
	]);
}
function vector(value, maximum) {
	if (!Array.isArray(value) || value.length < 1 || value.length > maximum) throw new ExperimentRefusal("bounded output vector required");
	value.forEach(decodeFloat64);
}
/** Read one producer's exact metadata after its independent offline verifier succeeds. */
async function readExperimentArtifact(kind, bytes) {
	if (bytes.byteLength < 1 || bytes.byteLength > byteLimit) throw new ExperimentRefusal("bounded artifact metadata required");
	const payload = object$6(readJson(new TextDecoder("utf-8", { fatal: true }).decode(bytes)), kind === "policy" ? [
		"source",
		"addressableBytes",
		"memoryBytes",
		"workUnits",
		"overheadBytes"
	] : kind === "environment" ? [
		"backend",
		"integrator",
		"dtype",
		"phase_unit",
		"frequency_unit",
		"time_unit",
		"seed",
		"kernel_sha256",
		"bounds",
		"browser_user_agent"
	] : kind === "plan" ? [
		"revision_hash",
		"kernel_sha256",
		"input_sha256",
		"environment_sha256",
		"policy_sha256",
		"shape",
		"bounds",
		"binary_bytes",
		"policy",
		"requested_memory_bytes",
		"deadline_ms",
		"admission"
	] : [
		"revision_hash",
		"plan_hash",
		"kernel_sha256",
		"order_parameter",
		"theta_final"
	]);
	if (kind === "policy") admitOwnedKuramotoResources({
		n: 1,
		steps: 1,
		mode: "mean-field"
	}, {
		maxOscillators: 128,
		maxSteps: 4096
	}, 8, policy(payload));
	else if (kind === "environment") {
		if (payload["backend"] !== "shipped-kuramoto-wasm-float64-owned-worker" || payload["integrator"] !== "Rust fixed-step RK4" || payload["dtype"] !== "float64" || payload["phase_unit"] !== "rad" || payload["frequency_unit"] !== "rad/model-time" || payload["time_unit"] !== "model-time" || payload["seed"] !== null) throw new ExperimentRefusal("unsupported numerical environment");
		hash(payload["kernel_sha256"]);
		bounds(payload["bounds"]);
		if (typeof payload["browser_user_agent"] !== "string" || payload["browser_user_agent"].length > 4096) throw new ExperimentRefusal("bounded observed browser environment required");
	} else if (kind === "plan") {
		for (const key of [
			"revision_hash",
			"kernel_sha256",
			"input_sha256",
			"environment_sha256",
			"policy_sha256"
		]) hash(payload[key]);
		const shape = object$6(payload["shape"], [
			"n",
			"steps",
			"mode"
		]);
		const actual = admitOwnedKuramotoResources({
			n: shape["n"],
			steps: shape["steps"],
			mode: shape["mode"]
		}, bounds(payload["bounds"]), payload["binary_bytes"], policy(payload["policy"]));
		const requested = payload["requested_memory_bytes"];
		if (requested !== null && (typeof requested !== "bigint" || requested < 0n || requested !== actual.policy.memoryBytes)) throw new ExperimentRefusal("requested and effective run byte ceiling differ");
		if (await canonicalDigest("studio.kuramoto-admission.v1", actual) !== await canonicalDigest("studio.kuramoto-admission.v1", payload["admission"])) throw new ExperimentRefusal("resource admission differs from the original owner");
		if (!Number.isSafeInteger(payload["deadline_ms"]) || payload["deadline_ms"] < 1 || payload["deadline_ms"] > 6e4) throw new ExperimentRefusal("bounded operational deadline required");
	} else {
		for (const key of [
			"revision_hash",
			"plan_hash",
			"kernel_sha256"
		]) hash(payload[key]);
		vector(payload["order_parameter"], 4097);
		vector(payload["theta_final"], 128);
	}
	return Object.freeze(payload);
}
async function verify(kind, content) {
	if (kind === "input") {
		decodeKernelInput(content);
		return;
	}
	if (kind === "kernel") {
		if (content.length < 8 || content.length > byteLimit || [
			0,
			97,
			115,
			109,
			1,
			0,
			0,
			0
		].some((byte, index) => content[index] !== byte)) throw new ExperimentRefusal("bounded original WASM v1 source required");
		return;
	}
	await readExperimentArtifact(kind, content);
}
/** Create a verified original raw member, preserving supplied bytes and their SHA256. */
async function makeArtifact(kind, value) {
	const bytes = value instanceof Uint8Array ? new Uint8Array(value) : new TextEncoder().encode(writeJson(value));
	const verifier = localExperimentCodecs.get(experimentSchemas[kind]);
	if (verifier === void 0) throw new ExperimentRefusal("original artifact producer unavailable");
	const identity = await verifier(bytes);
	return Object.freeze({
		name: `raw/${identity.digest}.bin`,
		kind: "raw",
		schema: identity.schema,
		sha256: identity.digest,
		content: Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")
	});
}
/** Restore bounded original bytes from an already graph-admitted raw archive member. */
function artifactContent(member) {
	if (member.kind !== "raw" || member.content.length > 2 * byteLimit || member.content.length % 2 !== 0 || /[^0-9a-f]/.test(member.content)) throw new ExperimentRefusal("bounded admitted original raw member required");
	const nativeDecoder = Uint8Array.fromHex;
	if (typeof nativeDecoder === "function") return nativeDecoder(member.content);
	return Uint8Array.from({ length: member.content.length / 2 }, (_, index) => parseInt(member.content.slice(2 * index, 2 * index + 2), 16));
}
var ExperimentRefusal, experimentSchemas, producerKinds, byteLimit, localExperimentCodecs;
var init_kuramotoArtifacts = __esmMin((() => {
	init_kuramoto();
	init_contracts();
	init_canonical();
	init_kuramotoResources();
	ExperimentRefusal = class extends Error {};
	experimentSchemas = Object.freeze({
		/** Original bounded little-endian kernel input bytes. */ input: "studio.kuramoto-input.v1",
		/** Original WASM v1 program bytes. */ kernel: "studio.kuramoto-kernel.v1",
		/** Declared original numeric-payload and work ceilings. */ policy: "studio.kuramoto-policy.v1",
		/** Recorded source backend, method, precision, units and build. */ environment: "studio.kuramoto-environment.v1",
		/** Immutable source-bound effective numerical admission. */ plan: "studio.kuramoto-plan.v1",
		/** Source-bound actual order parameter and final phase bits. */ output: "studio.kuramoto-output.v1"
	});
	producerKinds = {
		input: "problem",
		kernel: "program",
		policy: "policy",
		environment: "environment",
		plan: "plan",
		output: "output"
	};
	byteLimit = 2 * 1024 * 1024;
	localExperimentCodecs = new Map(Object.entries(experimentSchemas).map(([name, schema]) => {
		const kind = name;
		const codec = async (content) => {
			await verify(kind, content);
			return {
				schema,
				kind: producerKinds[kind],
				digest: await artifactBytesDigest(content)
			};
		};
		return [schema, codec];
	}));
}));
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/experiments/experimentArchive.ts
init_kuramoto();
init_contracts();
init_kuramotoArtifacts();
/** Verify source-bound event ordering; a recorded terminal event is never independent numerical evidence. */
function validateExperimentLifecycle(events, runId, revisionHash, planHash, kernelHash, terminal) {
	let previous = 0n, accepted = false;
	if (events.length === 0 || events.at(-1).kind !== terminal || events.at(-1).payload["disposed"] !== true) throw new ExperimentRefusal("disposed original terminal lifecycle required");
	for (let index = 0; index < events.length; index++) {
		const event = events[index];
		if (BigInt(event.version) !== 1n || BigInt(event.sequence) <= previous || event.run_id !== runId || event.payload["revision_hash"] !== revisionHash || event.payload["plan_hash"] !== planHash || event.payload["build_fingerprint"] !== kernelHash) throw new ExperimentRefusal("original attempt lifecycle identity or sequence differs");
		previous = BigInt(event.sequence);
		if (event.kind === "accepted") {
			if (accepted || index !== 0) throw new ExperimentRefusal("original worker acceptance must occur once before progress");
			accepted = true;
		} else if (event.kind === "progress") {
			if (!accepted) throw new ExperimentRefusal("original worker progress precedes admission");
		} else if (index !== events.length - 1) {
			if (!(event.kind === "result" && index === events.length - 2 && terminal === "failed" && events.at(-1).payload["origin"] === "original client outcome")) throw new ExperimentRefusal("original terminal event precedes another event");
		}
	}
	if (terminal === "result" && !accepted) throw new ExperimentRefusal("successful original attempt requires worker acceptance");
}
function take$1(parsed) {
	if (!parsed.ok) throw new ExperimentRefusal("original workspace document refused");
	return parsed.value;
}
function ref(member) {
	return {
		schema: member.schema,
		sha256: member.sha256,
		media_type: member.schema === experimentSchemas.kernel || member.schema === experimentSchemas.input ? "application/octet-stream" : "application/json"
	};
}
async function documentMember(document) {
	const sha256 = await documentDigest(document);
	return {
		name: `documents/${sha256}.json`,
		kind: "document",
		schema: document.schema,
		sha256,
		content: writeJson(document)
	};
}
function referenced(archive, reference, expected) {
	const value = reference;
	const member = archive.members.find((item) => item.sha256 === value.sha256 && item.schema === value.schema);
	if (member.schema !== expected) throw new ExperimentRefusal("local experiment source reference is unsupported");
	return member;
}
/** Admit the original graph and read an explicit immutable revision or its selected draft, without executing or changing the archive. */
async function readLocalExperiment(json, rawCodecs, revisionHash) {
	const archive = await admitWorkspaceArchive(json, rawCodecs);
	const selection = revisionHash === void 0 ? archive.manifest.body["draft_ref"] : { sha256: revisionHash };
	if (selection === null) throw new ExperimentRefusal("no local experiment revision selected");
	const revision = archive.revisions[selection.sha256];
	if (revision === void 0) throw new ExperimentRefusal("immutable local experiment revision is absent from the admitted archive");
	const extension = revision.extensions["local_experiment"];
	if (extension?.["version"] !== 1n || extension["model"] !== "classical Kuramoto") throw new ExperimentRefusal("unsupported local experiment source version or model");
	const base = decodeKernelInput(artifactContent(referenced(archive, revision.body["problem_ref"], experimentSchemas.input)));
	const kernel = referenced(archive, revision.body["program_ref"], experimentSchemas.kernel);
	const settings = take$1(parseResolvedSettings(readJson(referenced(archive, revision.body["semantic_settings_ref"], "resolved_settings.v1").content)));
	const environmentMember = referenced(archive, settings.body["environment_ref"], experimentSchemas.environment);
	const environment = await readExperimentArtifact("environment", artifactContent(environmentMember));
	if (environment["kernel_sha256"] !== kernel.sha256) throw new ExperimentRefusal("environment and original source program differ");
	const sourcePolicy = await readExperimentArtifact("policy", artifactContent(referenced(archive, settings.body["policy_ref"], experimentSchemas.policy)));
	const parameters = revision.body["parameters"];
	const keys = [
		"omega",
		"theta0",
		"coupling",
		"dt",
		"steps",
		...base.mode === "networked" ? ["k_nm"] : []
	];
	if (Object.keys(parameters).length !== keys.length || keys.some((key) => parameters[key] === void 0)) throw new ExperimentRefusal("local source parameter fields differ from the original method");
	const numeric = (key, shape, unit) => {
		const value = parameters[key];
		if (value.dtype !== "float64" || value.shape.length !== shape.length || value.shape.some((dimension, index) => dimension !== shape[index]) || archive.parameterUnits[key] !== unit) throw new ExperimentRefusal("original source dtype, shape or unit differs");
		return Object.freeze(value.values.map(decodeFloat64));
	};
	const n = BigInt(base.omega.length), stepsValue = parameters["steps"];
	if (stepsValue.dtype !== "uint64" || stepsValue.shape.length !== 0 || archive.parameterUnits["steps"] !== "1") throw new ExperimentRefusal("original step count dtype, shape or unit differs");
	const steps = BigInt(stepsValue.values[0]);
	if (steps < 1n || steps > 4096n) throw new ExperimentRefusal("bounded original step count required");
	const request = Object.freeze({
		mode: base.mode,
		omega: numeric("omega", [n], "rad/model-time"),
		theta0: numeric("theta0", [n], "rad"),
		coupling: numeric("coupling", [], "rad/model-time")[0],
		dt: numeric("dt", [], "model-time")[0],
		steps: Number(steps),
		...base.mode === "networked" ? { kNm: numeric("k_nm", [n, n], "rad/model-time") } : {}
	});
	if (encodeKuramotoInput(request) === null) throw new ExperimentRefusal("effective original source request refused");
	const effective = settings.body["effective"];
	const semanticHash = await canonicalDigest("studio.kuramoto-settings.v1", {
		backend: environment["backend"],
		method: request.mode,
		precision: "float64",
		seed: null,
		time_unit: "model-time"
	});
	if (await canonicalDigest("studio.kuramoto-settings.v1", effective) !== semanticHash || await canonicalDigest("studio.kuramoto-settings.v1", settings.body["requested"]) !== semanticHash || settings.body["rejected_fields"].length !== 0) throw new ExperimentRefusal("recorded requested/effective semantic settings and original source differ");
	return Object.freeze({
		archive,
		revision,
		revisionHash: selection.sha256,
		request,
		kernelBytes: artifactContent(kernel),
		kernelHash: kernel.sha256,
		environment,
		environmentMember,
		policy: sourcePolicy
	});
}
/** Append one genuinely disposed attempt to its unchanged selected revision through original admission. */
async function appendExperimentAttempt(plan, runId, attemptId, events, outcome, rawCodecs) {
	if (!outcome.disposed || events.length === 0) throw new ExperimentRefusal("disposed original attempt diagnostics required");
	const source = await readLocalExperiment(plan.sourceJson, rawCodecs);
	if (source.revisionHash !== plan.revisionHash) throw new ExperimentRefusal("attempt revision differs from the selected source");
	const terminal = outcome.ok ? "result" : outcome.code === "cancelled" ? "cancelled" : "failed";
	validateExperimentLifecycle(events, runId, plan.revisionHash, plan.planHash, plan.buildFingerprint, terminal);
	const additional = [
		plan.inputMember,
		plan.policyMember,
		plan.planMember
	];
	const outputRefs = [];
	if (outcome.ok) {
		if (outcome.runId !== runId || outcome.revisionHash !== plan.revisionHash || outcome.planHash !== plan.planHash || outcome.buildFingerprint !== plan.buildFingerprint || outcome.run.orderParameter.length !== plan.request.steps + 1 || outcome.run.thetaFinal.length !== plan.request.omega.length || !events.some((event) => event.kind === "accepted")) throw new ExperimentRefusal("successful result does not belong to the original admitted plan");
		const output = await makeArtifact("output", {
			revision_hash: plan.revisionHash,
			plan_hash: plan.planHash,
			kernel_sha256: plan.buildFingerprint,
			order_parameter: Array.from(outcome.run.orderParameter, encodeFloat64),
			theta_final: Array.from(outcome.run.thetaFinal, encodeFloat64)
		});
		additional.push(output);
		outputRefs.push(ref(output));
	}
	const wireEvents = events.map((event) => {
		const payload = { ...event.payload };
		for (const key of ["orderParameter", "thetaFinal"]) {
			const value = payload[key];
			if (ArrayBuffer.isView(value) && Object.prototype.toString.call(value) === "[object Float64Array]") {
				const vector = value;
				payload[key] = {
					dtype: "float64",
					shape: [BigInt(vector.length)],
					values: Array.from(vector, encodeFloat64)
				};
			}
		}
		return {
			...event,
			version: 1n,
			sequence: BigInt(event.sequence),
			payload
		};
	});
	const recordMember = await documentMember(take$1(parseLocalRunRecord({
		schema: "local_run_record.v1",
		body: {
			run_id: runId,
			attempt_id: attemptId,
			revision_hash: plan.revisionHash,
			plan_hash: plan.planHash,
			mode: "local",
			events: wireEvents,
			output_refs: outputRefs
		},
		extensions: {}
	})));
	additional.push(recordMember);
	const original = source.archive;
	const existing = new Set(original.members.map((member) => member.sha256));
	const appended = additional.filter((member) => {
		if (existing.has(member.sha256)) return false;
		existing.add(member.sha256);
		return true;
	});
	const manifest = take$1(parseWorkspaceManifest({
		...original.manifest,
		body: {
			...original.manifest.body,
			artefact_refs: [...original.manifest.body["artefact_refs"], ref(recordMember)],
			updated_at: (/* @__PURE__ */ new Date()).toISOString()
		}
	}));
	const preview = await previewWorkspaceArchive(writeJson({
		schema: original.preview.schema,
		manifest,
		members: [...original.members, ...appended],
		parameter_units: original.parameterUnits
	}), rawCodecs);
	return Object.freeze({
		...preview,
		runRecordHash: recordMember.sha256
	});
}
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/experiments/experimentPlan.ts
init_kuramoto();
init_contracts();
init_kuramotoResources();
init_kuramotoArtifacts();
/** Reuse original resource admission and bind one source revision to the actually shipped kernel. */
async function prepareExperimentPlan(source, kernel, options = {}) {
	if (!kernel.sourceBytes || kernel.sourceBytes.length < 1 || await artifactBytesDigest(kernel.sourceBytes) !== source.kernelHash) throw new ExperimentRefusal("currently shipped kernel differs from the original source; replay unavailable");
	const recorded = source.environment["bounds"];
	if (recorded.maxOscillators !== kernel.bounds.maxOscillators || recorded.maxSteps !== kernel.bounds.maxSteps) throw new ExperimentRefusal("native kernel limits differ from the original environment");
	const ceiling = browserResourcePolicy(kernel.bounds), original = source.policy;
	if (original.addressableBytes > ceiling.addressableBytes || original.memoryBytes !== null && original.memoryBytes > ceiling.memoryBytes || original.workUnits !== null && original.workUnits > ceiling.workUnits) throw new ExperimentRefusal("source policy ceiling exceeds the current original product policy");
	let requestedMemory = null;
	let policy = original;
	if (options.memoryBudget !== void 0) {
		if (!/^(?:0|[1-9][0-9]{0,19})$/.test(options.memoryBudget)) throw new ExperimentRefusal("canonical nonnegative bounded byte count required");
		requestedMemory = BigInt(options.memoryBudget);
		if (original.memoryBytes === null || requestedMemory > original.memoryBytes) throw new ExperimentRefusal("requested byte budget exceeds the original policy ceiling");
		policy = {
			...original,
			memoryBytes: requestedMemory,
			source: "explicit run numeric-payload ceiling below the original source policy"
		};
	}
	const admission = admitOwnedKuramotoResources({
		n: source.request.omega.length,
		steps: source.request.steps,
		mode: source.request.mode
	}, kernel.bounds, kernel.sourceBytes.length, policy);
	const deadlineMs = options.deadlineMs ?? 5e3;
	if (!Number.isSafeInteger(deadlineMs) || deadlineMs < 1 || deadlineMs > 6e4) throw new ExperimentRefusal("bounded operational deadline required");
	const input = encodeKuramotoInput(source.request);
	if (input === null) throw new ExperimentRefusal("effective original input refused before worker allocation");
	const inputMember = await makeArtifact("input", input), policyMember = await makeArtifact("policy", { ...admission.policy });
	const planMember = await makeArtifact("plan", {
		revision_hash: source.revisionHash,
		kernel_sha256: source.kernelHash,
		input_sha256: inputMember.sha256,
		environment_sha256: source.environmentMember.sha256,
		policy_sha256: policyMember.sha256,
		shape: {
			n: source.request.omega.length,
			steps: source.request.steps,
			mode: source.request.mode
		},
		bounds: kernel.bounds,
		binary_bytes: kernel.sourceBytes.length,
		policy: admission.policy,
		requested_memory_bytes: requestedMemory,
		deadline_ms: deadlineMs,
		admission
	});
	return Object.freeze({
		sourceJson: source.archive.preview.json,
		revisionHash: source.revisionHash,
		planHash: planMember.sha256,
		buildFingerprint: source.kernelHash,
		wasmBytes: new Uint8Array(kernel.sourceBytes),
		bounds: Object.freeze({ ...kernel.bounds }),
		request: source.request,
		admission,
		policy: admission.policy,
		deadlineMs,
		requestedMemoryBytes: requestedMemory,
		inputMember,
		policyMember,
		planMember
	});
}
/** Re-admit an explicitly listed completed record and reprepare its exact original numerical plan. */
async function prepareSavedReplay(json, kernel, rawCodecs) {
	const source = await readLocalExperiment(json, rawCodecs);
	const references = source.archive.manifest.body["artefact_refs"];
	const members = new Map(source.archive.members.map((member) => [member.sha256, member]));
	for (const reference of [...references].reverse()) {
		const member = members.get(reference.sha256);
		if (member.schema !== "local_run_record.v1") continue;
		const record = readJson(member.content).body;
		const events = record["events"];
		if (record["revision_hash"] !== source.revisionHash || events.at(-1)?.kind !== "result" || events.at(-1)?.payload["disposed"] !== true) continue;
		const storedPlan = members.get(record["plan_hash"]);
		if (storedPlan === void 0 || storedPlan.schema !== experimentSchemas.plan) throw new ExperimentRefusal("original replay plan source format is unsupported");
		const payload = await readExperimentArtifact("plan", artifactContent(storedPlan));
		const requested = payload["requested_memory_bytes"];
		const plan = await prepareExperimentPlan(source, kernel, {
			deadlineMs: payload["deadline_ms"],
			...requested === null ? {} : { memoryBudget: requested.toString() }
		});
		if (plan.planHash !== record["plan_hash"]) throw new ExperimentRefusal("original replay plan identity differs from the current source");
		validateExperimentLifecycle(events, record["run_id"], plan.revisionHash, plan.planHash, plan.buildFingerprint, "result");
		const outputs = record["output_refs"];
		const outputReference = outputs[0];
		if (outputs.length !== 1 || outputReference?.schema !== experimentSchemas.output) throw new ExperimentRefusal("one original source output artifact required for replay");
		const output = await readExperimentArtifact("output", artifactContent(members.get(outputReference.sha256)));
		if (output["revision_hash"] !== plan.revisionHash || output["plan_hash"] !== plan.planHash || output["kernel_sha256"] !== plan.buildFingerprint) throw new ExperimentRefusal("original replay output belongs to another source or plan");
		const orderParameter = output["order_parameter"], thetaFinal = output["theta_final"];
		if (orderParameter.length !== plan.request.steps + 1 || thetaFinal.length !== plan.request.omega.length) throw new ExperimentRefusal("original replay output shape differs from the plan");
		return Object.freeze({
			plan,
			expected: Object.freeze({
				orderParameter: Object.freeze([...orderParameter]),
				thetaFinal: Object.freeze([...thetaFinal])
			})
		});
	}
	throw new ExperimentRefusal("no disposed successful saved run for the selected immutable revision");
}
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/parameters/parameterDraft.ts
init_contracts();
init_canonical();
function take(result) {
	if (!result.ok) throw new Error(`${result.path}: ${result.message}`);
	return result.value;
}
function freezeSnapshot(source, parameters, units, masks) {
	canonicalBytes("studio_parameter_input.v1", {
		parameters,
		units,
		masks: masks ?? null
	});
	const entries = dataEntries(parameters);
	const keys = entries.map(([key]) => key);
	if (Object.keys(units).length !== keys.length || keys.some((key) => !Object.hasOwn(units, key))) throw new Error("Parameter units must cover the values exactly");
	if (masks && (Object.keys(masks).length !== keys.length || keys.some((key) => !Object.hasOwn(masks, key)))) throw new Error("Trainable masks must cover the values exactly");
	const specs = new Map(source.specs.map((spec) => [spec.body["key"], spec]));
	let total = 0;
	const values = [];
	const selectedMasks = [];
	for (const [key, payload] of entries) {
		const spec = specs.get(key);
		if (!spec) throw new Error(`Missing ParameterSpec for ${key}`);
		const unit = units[key];
		take(validateParameterBinding(spec, payload, unit));
		const typed = payload;
		const shape = spec.body["shape"];
		if (shape.length > 2) throw new Error("Parameter editor supports scalar, vector and matrix shapes only");
		total += typed.values.length;
		if (total > 4096) throw new Error("Parameter editor exceeds the 4096-element bound");
		values.push([key, Object.freeze({
			dtype: spec.body["dtype"],
			shape,
			values: Object.freeze([...typed.values])
		})]);
		const mask = masks?.[key] ?? typed.values.map(() => spec.body["trainable"] === true);
		if (mask.length !== typed.values.length || mask.some((value) => typeof value !== "boolean" || value && spec.body["trainable"] !== true)) throw new Error("Trainable mask must match shape and source eligibility");
		selectedMasks.push([key, Object.freeze([...mask])]);
	}
	return Object.freeze({
		parameters: Object.freeze(Object.fromEntries(values)),
		units: Object.freeze({ ...units }),
		trainableMasks: Object.freeze(Object.fromEntries(selectedMasks))
	});
}
/** Validate original parsers/bindings and capture an independent immutable draft. */
function createParameterDraft(input) {
	canonicalBytes("studio_parameter_source.v1", input);
	const revision = take(parseExperimentRevision(input.revision));
	const specs = Object.freeze(input.specs.map((spec) => take(parseParameterSpec(spec))));
	if (new Set(specs.map((spec) => spec.body["key"])).size !== specs.length) throw new Error("Duplicate ParameterSpec key");
	const source = Object.freeze({
		revision,
		specs,
		units: Object.freeze({ ...input.units })
	});
	const editor = revision.extensions["parameter_editor"];
	let masks;
	if (editor !== void 0) {
		if (typeof editor !== "object" || editor === null || Array.isArray(editor)) throw new Error("Parameter editor metadata must be an object");
		const fields = Object.fromEntries(dataEntries(editor));
		if (Object.keys(fields).length !== 2 || fields["version"] !== 1n || typeof fields["trainable_masks"] !== "object" || fields["trainable_masks"] === null || Array.isArray(fields["trainable_masks"])) throw new Error("Unsupported parameter editor version or trainable masks");
		const maskFields = dataEntries(fields["trainable_masks"]);
		if (maskFields.some(([, mask]) => !Array.isArray(mask))) throw new Error("Trainable masks must be arrays");
		masks = Object.fromEntries(maskFields);
	}
	const snapshot = freezeSnapshot(source, revision.body["parameters"], source.units, masks);
	const first = Object.entries(snapshot.parameters).find(([, value]) => value.values.length > 0);
	return Object.freeze({
		source,
		snapshot,
		selection: first ? Object.freeze({
			key: first[0],
			index: 0
		}) : null,
		policy: "directed",
		past: Object.freeze([]),
		future: Object.freeze([]),
		refusal: null
	});
}
function encode(text, dtype) {
	if (dtype !== "float64") {
		if (!/^(?:0|-?[1-9][0-9]{0,19})$/.test(text) || /[\r\n]$/.test(text)) throw new Error("Canonical int64/uint64 decimal required");
		return text;
	}
	if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(text) || /[\r\n]$/.test(text) || !Number.isFinite(Number(text))) throw new Error("A finite float64 decimal value is required");
	const bytes = /* @__PURE__ */ new Uint8Array(8);
	new DataView(bytes.buffer).setFloat64(0, Number(text), false);
	return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}
var inputUnits = Object.freeze({
	rad: Object.freeze({
		rad: 1,
		mrad: .001,
		urad: 1e-6
	}),
	s: Object.freeze({
		s: 1,
		ms: .001,
		us: 1e-6
	}),
	"rad/s": Object.freeze({
		"rad/s": 1,
		"mrad/s": .001,
		"rad/ms": 1e3
	})
});
function convertedText(text, from, to, dtype) {
	encode(text, dtype);
	if (dtype !== "float64") throw new Error("Unit conversion requires float64; integer input retains its exact source unit");
	const units = Object.hasOwn(inputUnits, to) ? inputUnits[to] : void 0;
	const factor = units && Object.hasOwn(units, from) ? units[from] : void 0;
	if (factor === void 0) throw new Error("Unsupported unit conversion; source unit and values retained");
	const value = Number(text) * factor;
	return Object.is(value, -0) ? "-0" : String(value);
}
function element(state, key, index) {
	const value = Object.hasOwn(state.snapshot.parameters, key) ? state.snapshot.parameters[key] : void 0;
	if (!value || !Number.isSafeInteger(index) || index < 0 || index >= value.values.length) throw new Error("Parameter selection out of range");
	return value;
}
/** Apply one validated edit atomically; invalid values never modify history or saved evidence. */
function parameterDraftReducer(state, action) {
	try {
		if (action.type === "undo" || action.type === "redo") {
			const undo = action.type === "undo";
			const target = (undo ? state.past : state.future).at(-1);
			if (!target) return state;
			return Object.freeze({
				...state,
				snapshot: target,
				past: undo ? Object.freeze(state.past.slice(0, -1)) : Object.freeze([...state.past, state.snapshot]),
				future: undo ? Object.freeze([...state.future, state.snapshot]) : Object.freeze(state.future.slice(0, -1)),
				refusal: null
			});
		}
		if (action.type === "policy") return Object.freeze({
			...state,
			policy: action.policy,
			refusal: null
		});
		if (action.type === "select") {
			element(state, action.key, action.index);
			return Object.freeze({
				...state,
				selection: Object.freeze({
					key: action.key,
					index: action.index
				}),
				refusal: null
			});
		}
		let snapshot;
		if (action.type === "replace") snapshot = freezeSnapshot(state.source, action.parameters, action.units, state.snapshot.trainableMasks);
		else {
			const value = element(state, action.key, action.index);
			const indices = [action.index];
			if (state.policy === "symmetric" && value.shape.length === 2) {
				if (value.shape[0] !== value.shape[1]) throw new Error("Symmetric edits require a square matrix");
				const n = Number(value.shape[0]);
				const reverse = action.index % n * n + Math.floor(action.index / n);
				if (reverse !== action.index) indices.push(reverse);
			}
			if (action.type === "value") {
				const targetUnit = state.snapshot.units[action.key];
				if (targetUnit === void 0) throw new Error("Parameter source unit unavailable");
				if (action.unit !== targetUnit && !action.convert) throw new Error("Input unit mismatch; choose an explicit supported conversion");
				const encoded = encode(action.convert ? convertedText(action.text, action.unit, targetUnit, value.dtype) : action.text, value.dtype);
				const values = [...value.values];
				for (const index of indices) values[index] = encoded;
				snapshot = freezeSnapshot(state.source, {
					...state.snapshot.parameters,
					[action.key]: {
						...value,
						values
					}
				}, state.snapshot.units, state.snapshot.trainableMasks);
			} else {
				const originalMask = state.snapshot.trainableMasks[action.key];
				if (originalMask === void 0) throw new Error("Parameter trainable mask unavailable");
				const mask = [...originalMask];
				for (const index of indices) mask[index] = action.enabled;
				snapshot = freezeSnapshot(state.source, state.snapshot.parameters, state.snapshot.units, {
					...state.snapshot.trainableMasks,
					[action.key]: mask
				});
			}
		}
		if (new TextDecoder().decode(canonicalBytes("studio_parameter_snapshot.v1", snapshot)) === new TextDecoder().decode(canonicalBytes("studio_parameter_snapshot.v1", state.snapshot))) return Object.freeze({
			...state,
			refusal: null
		});
		return Object.freeze({
			...state,
			snapshot,
			past: Object.freeze([...state.past, state.snapshot].slice(-64)),
			future: Object.freeze([]),
			refusal: null
		});
	} catch (cause) {
		return Object.freeze({
			...state,
			refusal: cause instanceof Error ? cause.message : "Parameter edit refused"
		});
	}
}
/** Revalidate an external save candidate through original specifications before revision creation. */
function validateParameterSnapshot(source, snapshot) {
	const admitted = createParameterDraft(source);
	if (typeof snapshot.trainableMasks !== "object" || snapshot.trainableMasks === null || Array.isArray(snapshot.trainableMasks)) throw new Error("Trainable masks must be a complete object map");
	if (dataEntries(snapshot.trainableMasks).some(([, mask]) => !Array.isArray(mask))) throw new Error("Trainable masks must be arrays");
	if (Object.keys(snapshot.parameters).length !== Object.keys(admitted.snapshot.parameters).length || Object.keys(admitted.snapshot.parameters).some((key) => !Object.hasOwn(snapshot.parameters, key))) throw new Error("Revision parameter keys must remain unchanged");
	return freezeSnapshot(admitted.source, snapshot.parameters, snapshot.units, snapshot.trainableMasks);
}
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/parameters/parameterRevision.ts
init_contracts();
/** Create a v1 child revision; source records, settings, results and prior references remain untouched. */
async function createParameterRevision(source, candidate) {
	const captured = createParameterDraft(source).source;
	const snapshot = validateParameterSnapshot(captured, candidate);
	const original = captured.revision;
	const parent = await documentDigest(original);
	const revision = Object.freeze({
		schema: original.schema,
		body: Object.freeze({
			...original.body,
			parent_revision_hashes: Object.freeze([parent]),
			parameters: snapshot.parameters
		}),
		extensions: Object.freeze({
			...original.extensions,
			parameter_editor: Object.freeze({
				version: 1n,
				trainable_masks: snapshot.trainableMasks
			})
		})
	});
	await documentDigest(revision);
	return revision;
}
function sourceFromAdmission(archive, revisionHash) {
	const refs = archive.manifest.body["revision_refs"];
	const draft = archive.manifest.body["draft_ref"];
	const hash = revisionHash ?? draft?.sha256 ?? refs.at(-1)?.sha256;
	if (hash === void 0) return null;
	if (!refs.some((ref) => ref.sha256 === hash) && draft?.sha256 !== hash) throw new Error("Parameter revision must be listed in the admitted project");
	const document = archive.revisions[hash];
	const keys = Object.keys(document.body["parameters"]);
	const specs = Object.values(archive.parameterSpecs).filter((spec) => keys.includes(spec.body["key"]));
	const units = Object.fromEntries(keys.map((key) => [key, archive.parameterUnits[key]]));
	return Object.freeze({
		revision: document,
		specs: Object.freeze(specs),
		units: Object.freeze(units)
	});
}
/** Admit the original archive before resolving its selected parameter source. */
async function parameterSourceFromArchive(json, rawCodecs, revisionHash) {
	return sourceFromAdmission(await admitWorkspaceArchive(json, rawCodecs), revisionHash);
}
/** Append a fully admitted child to the exact original archive, retaining every prior member byte. */
async function appendParameterRevision(json, source, snapshot, rawCodecs, updatedAt) {
	const original = await admitWorkspaceArchive(json, rawCodecs);
	const revision = await createParameterRevision(sourceFromAdmission(original, await documentDigest(source.revision)), snapshot);
	const revisionHash = await documentDigest(revision);
	if (original.members.some((member) => member.sha256 === revisionHash)) throw new Error("That immutable parameter revision already exists; make another edit before saving");
	const ref = {
		schema: revision.schema,
		sha256: revisionHash,
		media_type: "application/json"
	};
	const manifest = parseWorkspaceManifest({
		...original.manifest,
		body: {
			...original.manifest.body,
			revision_refs: [...original.manifest.body["revision_refs"], ref],
			draft_ref: ref,
			updated_at: updatedAt
		}
	});
	if (!manifest.ok) throw new Error(`${manifest.path}: ${manifest.message}`);
	const member = {
		name: `documents/${revisionHash}.json`,
		kind: "document",
		schema: revision.schema,
		sha256: revisionHash,
		content: writeJson(revision)
	};
	const archive = await previewWorkspaceArchive(writeJson({
		schema: original.preview.schema,
		manifest: manifest.value,
		members: [...original.members, member],
		parameter_units: original.parameterUnits
	}), rawCodecs);
	return Object.freeze({
		archive,
		revisionHash
	});
}
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/workflows/workflowModel.ts
init_contracts();
init_canonical();
/** Exact additive graph version; no existing workspace format is replaced. */
var workflowSchema = "experiment_workflow.v1";
/** Total stage-evaluation ceiling; not a compute-duration promise. */
var maxWorkflowEvaluations = 4096;
/** Typed graph or original runtime value refused without executing or saving. */
var WorkflowRefusal = class extends Error {};
var verbs = /* @__PURE__ */ new Set([
	"compile",
	"simulate",
	"analyse",
	"validate",
	"benchmark",
	"replay",
	"differentiate",
	"mitigate",
	"execute"
]);
var dtypes = [
	"json",
	"float64",
	"int64",
	"uint64",
	"bool",
	"utf8"
];
function object$5(value, name, keys) {
	if (typeof value !== "object" || value === null || Array.isArray(value)) throw new WorkflowRefusal(`${name}: object required`);
	const result = Object.fromEntries(dataEntries(value));
	if (keys !== void 0 && (Object.keys(result).length !== keys.length || keys.some((k) => !Object.hasOwn(result, k)))) throw new WorkflowRefusal(`${name}: incomplete or unsupported fields`);
	return result;
}
function array(value, name, maximum) {
	if (!Array.isArray(value) || value.length > maximum) throw new WorkflowRefusal(`${name}: bounded array required`);
	return value;
}
function text$1(value, name, maximum = 128) {
	if (typeof value !== "string" || value.length === 0 || Array.from(value).length > maximum || Array.from(value).some((c) => c.charCodeAt(0) < 32)) throw new WorkflowRefusal(`${name}: bounded nonempty text required`);
	return value;
}
function identifier(value, name) {
	const result = text$1(value, name, 64);
	if (!/^[a-z][a-z0-9_-]{0,63}$/.test(result)) throw new WorkflowRefusal(`${name}: lowercase ASCII identifier required`);
	return result;
}
function parameterKey(value, name) {
	const result = text$1(value, name, 64);
	if (!/^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(result)) throw new WorkflowRefusal(`${name}: original ASCII parameter key required`);
	return result;
}
function integer(value, name, low, high) {
	if (typeof value !== "bigint" || value < low || value > high) throw new WorkflowRefusal(`${name}: integer between ${low} and ${high} required`);
	return value;
}
function capture(value) {
	if (Array.isArray(value)) return Object.freeze(value.map(capture));
	if (typeof value === "object" && value !== null) return Object.freeze(Object.fromEntries(dataEntries(value).map(([key, item]) => [key, capture(item)])));
	return value;
}
function captureRecord(value) {
	return Object.freeze(Object.fromEntries(Object.entries(value).map(([key, item]) => [key, capture(item)])));
}
function port(value) {
	const p = object$5(value, "port type", [
		"schema",
		"dtype",
		"shape",
		"unit"
	]);
	const schema = text$1(p["schema"], "port schema"), dtype = text$1(p["dtype"], "port dtype");
	if (!/^[a-z][a-z0-9_.-]{0,127}$/.test(schema) || !dtypes.includes(dtype)) throw new WorkflowRefusal("unsupported port schema or dtype");
	const shape = array(p["shape"], "port shape", 4).map((d) => integer(d, "port dimension", 0n, 4096n));
	if (shape.reduce((size, d) => size * d, 1n) > 4096n || dtype === "json" && shape.length !== 0) throw new WorkflowRefusal("unsupported port shape");
	return Object.freeze({
		schema,
		dtype,
		shape: Object.freeze(shape),
		unit: text$1(p["unit"], "port unit")
	});
}
function stage(value) {
	const p = object$5(value, "stage", [
		"id",
		"adapter",
		"verb",
		"backend",
		"parameters",
		"inputs",
		"outputs",
		"depends_on"
	]);
	const adapter = text$1(p["adapter"], "adapter"), verb = text$1(p["verb"], "verb");
	if (adapter !== "executive" && adapter !== "local-kuramoto" || !verbs.has(verb)) throw new WorkflowRefusal("unsupported workflow adapter or verb");
	if (adapter === "local-kuramoto" && ![
		"validate",
		"simulate",
		"analyse"
	].includes(verb)) throw new WorkflowRefusal("local classical adapter does not implement this verb");
	const parameters = object$5(p["parameters"], "parameters");
	const inputs = array(p["inputs"], "inputs", 16).map((item) => {
		const q = object$5(item, "input", [
			"parameter",
			"source_stage",
			"source_port",
			"type"
		]);
		return Object.freeze({
			parameter: parameterKey(q["parameter"], "input parameter"),
			source_stage: identifier(q["source_stage"], "source stage"),
			source_port: identifier(q["source_port"], "source port"),
			type: port(q["type"])
		});
	});
	const outputs = array(p["outputs"], "outputs", 16).map((item) => {
		const q = object$5(item, "output", [
			"name",
			"path",
			"type"
		]);
		const path = array(q["path"], "output path", 16).map((key) => text$1(key, "output path key", 256));
		if (path.length === 0) throw new WorkflowRefusal("nonempty original output path required");
		return Object.freeze({
			name: identifier(q["name"], "output name"),
			path: Object.freeze(path),
			type: port(q["type"])
		});
	});
	const incoming = inputs.map((i) => i.parameter), outgoing = outputs.map((o) => o.name);
	const depends_on = array(p["depends_on"], "dependencies", 64).map((id) => identifier(id, "dependency"));
	if (new Set(incoming).size !== incoming.length || new Set(outgoing).size !== outgoing.length || new Set(depends_on).size !== depends_on.length) throw new WorkflowRefusal("duplicate stage port or dependency");
	if (incoming.some((key) => Object.hasOwn(parameters, key))) throw new WorkflowRefusal("a bound input cannot also have a constant parameter");
	return Object.freeze({
		id: identifier(p["id"], "stage id"),
		adapter,
		verb,
		backend: text$1(p["backend"], "backend"),
		parameters: captureRecord(parameters),
		inputs: Object.freeze(inputs),
		outputs: Object.freeze(outputs),
		depends_on: Object.freeze(depends_on)
	});
}
function sweep(value, stages) {
	const p = object$5(value, "sweep", [
		"axes",
		"seeds",
		"seed_binding",
		"evaluation_budget"
	]);
	const byId = new Map(stages.map((s) => [s.id, s]));
	const targets = /* @__PURE__ */ new Set();
	let cells = 1;
	const axes = array(p["axes"], "axes", 4).map((item) => {
		const q = object$5(item, "axis", [
			"stage_id",
			"parameter",
			"values"
		]);
		const stage_id = identifier(q["stage_id"], "axis stage"), parameter = parameterKey(q["parameter"], "axis parameter");
		const target = JSON.stringify([stage_id, parameter]);
		const values = array(q["values"], "axis values", 64);
		const identities = values.map((v) => Array.from(canonicalBytes("studio.workflow-axis-value.v1", v)).join(","));
		if (values.length === 0 || new Set(identities).size !== values.length || targets.has(target)) throw new WorkflowRefusal("nonempty unique sweep coordinates required");
		const targetStage = byId.get(stage_id);
		if (targetStage === void 0 || targetStage.inputs.some((i) => i.parameter === parameter)) throw new WorkflowRefusal("sweep target is absent or already bound to a source");
		targets.add(target);
		cells *= values.length;
		return Object.freeze({
			stage_id,
			parameter,
			values: Object.freeze(values.map(capture))
		});
	});
	const seeds = array(p["seeds"], "seeds", 64).map((v) => text$1(v, "seed", 20));
	if (seeds.length === 0 || new Set(seeds).size !== seeds.length || seeds.some((s) => !/^(?:0|[1-9][0-9]{0,19})$/.test(s) || BigInt(s) > 2n ** 64n - 1n)) throw new WorkflowRefusal("unique canonical uint64 seeds required");
	let seed_binding = null;
	if (p["seed_binding"] !== null) {
		const q = object$5(p["seed_binding"], "seed binding", ["stage_id", "parameter"]);
		const stage_id = identifier(q["stage_id"], "seed stage"), parameter = parameterKey(q["parameter"], "seed parameter");
		const targetStage = byId.get(stage_id);
		if (targets.has(JSON.stringify([stage_id, parameter])) || targetStage === void 0 || Object.hasOwn(targetStage.parameters, parameter) || targetStage.inputs.some((i) => i.parameter === parameter)) throw new WorkflowRefusal("seed target is absent or already has a value");
		seed_binding = Object.freeze({
			stage_id,
			parameter
		});
	}
	const evaluation_budget = integer(p["evaluation_budget"], "evaluation budget", 1n, BigInt(maxWorkflowEvaluations));
	cells *= seeds.length;
	if (cells > 256 || BigInt(cells * stages.length) > evaluation_budget) throw new WorkflowRefusal("sweep exceeds cell or stage evaluation budget");
	return Object.freeze({
		axes: Object.freeze(axes),
		seeds: Object.freeze(seeds),
		seed_binding,
		evaluation_budget
	});
}
/** Return every stage once in stable ASCII lexical order, respecting data and control edges.
* Throws for missing dependencies or any graph cycle. No operation executes.
*/
function topologicalOrder(definition) {
	const pending = new Map(definition.stages.map((s) => [s.id, /* @__PURE__ */ new Set([...s.depends_on, ...s.inputs.map((i) => i.source_stage)])]));
	if (pending.size === 0 || pending.size !== definition.stages.length) throw new WorkflowRefusal("nonempty unique workflow stages required");
	if (Array.from(pending.values()).some((deps) => Array.from(deps).some((d) => !pending.has(d)))) throw new WorkflowRefusal("workflow dependency is absent");
	const order = [];
	while (pending.size !== 0) {
		const id = Array.from(pending).filter(([, deps]) => deps.size === 0).map(([id]) => id).sort()[0];
		if (id === void 0) throw new WorkflowRefusal("workflow graph contains a cycle");
		order.push(id);
		pending.delete(id);
		for (const dependencies of pending.values()) dependencies.delete(id);
	}
	return Object.freeze(order);
}
/** Admit a bounded immutable v1 graph before any request, worker or save exists.
* Uses the original lossless codec; rejects versions, unknown fields, duplicate
* identities, dangling edges, incompatible exact port declarations and budgets.
*/
function parseWorkflow(payload) {
	const json = writeJson(payload);
	if (new TextEncoder().encode(json).length > 16777216) throw new WorkflowRefusal("workflow definition exceeds byte bound");
	const document = object$5(readJson(json), "workflow", [
		"schema",
		"body",
		"extensions"
	]);
	if (document["schema"] !== "experiment_workflow.v1") throw new WorkflowRefusal("unsupported workflow schema");
	const body = object$5(document["body"], "workflow body", [
		"workflow_id",
		"stages",
		"sweep"
	]);
	const stages = array(body["stages"], "stages", 64).map(stage);
	if (stages.length === 0 || new Set(stages.map((s) => s.id)).size !== stages.length) throw new WorkflowRefusal("nonempty unique workflow stages required");
	const byId = new Map(stages.map((s) => [s.id, s]));
	if (stages.reduce((count, s) => count + s.inputs.length + s.depends_on.length, 0) > 128) throw new WorkflowRefusal("workflow edge bound exceeded");
	for (const s of stages) for (const input of s.inputs) {
		const producer = byId.get(input.source_stage);
		if (producer === void 0) throw new WorkflowRefusal("input producer stage is absent");
		const output = producer.outputs.find((o) => o.name === input.source_port);
		if (output === void 0 || writeJson(output.type) !== writeJson(input.type)) throw new WorkflowRefusal("input and original output port types are incompatible");
	}
	const definition = Object.freeze({
		workflow_id: identifier(body["workflow_id"], "workflow id"),
		stages: Object.freeze(stages),
		sweep: sweep(body["sweep"], stages),
		extensions: captureRecord(object$5(document["extensions"], "extensions"))
	});
	topologicalOrder(definition);
	return definition;
}
/** Produce an independent lossless wire document with original stage order and values. */
function workflowDocument(definition) {
	return Object.freeze({
		schema: workflowSchema,
		body: Object.freeze({
			workflow_id: definition.workflow_id,
			stages: definition.stages,
			sweep: definition.sweep
		}),
		extensions: definition.extensions
	});
}
/** Check original producer values against exact declared dtype/shape, without converting.
* Original handlers still validate domain and meaning. Returns an independent
* immutable datum; format, finiteness, scalar type or shape mismatch refuses.
*/
function validatePortValue(type, value) {
	canonicalBytes("studio.workflow-port-value.v1", value);
	port(type);
	const check = (item, shape) => {
		const [dimension, ...rest] = shape;
		if (dimension !== void 0) {
			if (!Array.isArray(item) || BigInt(item.length) !== dimension) throw new WorkflowRefusal("original port shape differs");
			for (const element of item) check(element, rest);
			return;
		}
		if (!(type.dtype === "json" || type.dtype === "utf8" && typeof item === "string" || type.dtype === "bool" && typeof item === "boolean" || type.dtype === "float64" && typeof item === "number" && Number.isFinite(item) || type.dtype === "int64" && typeof item === "bigint" && item >= -(2n ** 63n) && item < 2n ** 63n || type.dtype === "uint64" && typeof item === "bigint" && item >= 0n && item < 2n ** 64n)) throw new WorkflowRefusal("original port dtype differs");
	};
	check(value, type.shape);
	return capture(readJson(writeJson(value)));
}
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/workflows/workflowSweep.ts
init_contracts();
/** Readmit the graph and construct its bounded seed-ordered Cartesian coordinates.
* All source/unique-value/evaluation gates precede allocation. No backend,
* random generator, worker or storage operation is invoked. Returns stable
* exact cell identities; malformed or over-budget definitions refuse.
*/
async function buildWorkflowCells(definition) {
	const original = parseWorkflow(workflowDocument(definition));
	const workflow_digest = await canonicalDigest("studio.workflow-definition.v1", workflowDocument(original));
	const axes = original.sweep.axes;
	let coordinates = [[]];
	for (const axis of axes) coordinates = coordinates.flatMap((prefix) => axis.values.map((_, index) => [...prefix, index]));
	const cells = [];
	for (const seed of original.sweep.seeds) for (const coordinate of coordinates) {
		const overrides = Object.create(null);
		for (const [index, axis] of axes.entries()) {
			const position = coordinate[index];
			overrides[axis.stage_id] ??= Object.create(null);
			const values = overrides[axis.stage_id];
			values[axis.parameter] = axis.values[position];
		}
		const binding = original.sweep.seed_binding;
		if (binding !== null) {
			overrides[binding.stage_id] ??= Object.create(null);
			const values = overrides[binding.stage_id];
			values[binding.parameter] = BigInt(seed);
		}
		const id = await canonicalDigest("studio.workflow-cell.v1", {
			workflow_digest,
			coordinate: coordinate.map(BigInt),
			seed
		});
		cells.push(Object.freeze({
			id,
			workflow_digest,
			index: cells.length,
			coordinate: Object.freeze(coordinate),
			seed,
			overrides: Object.freeze(Object.fromEntries(Object.entries(overrides).map(([stage, values]) => [stage, Object.freeze(values)])))
		}));
	}
	return Object.freeze(cells);
}
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/workflows/workflowJournal.ts
init_contracts();
init_canonical();
/** Portable original attempt history; importing it never grants execution authority. */
var workflowJournalSchema = "experiment_workflow_journal.v1";
var statuses = [
	"running",
	"complete",
	"failed",
	"gated",
	"blocked",
	"cancelled",
	"interrupted"
];
function digest(value, name) {
	if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) throw new WorkflowRefusal(`${name}: exact SHA-256 identity required`);
	return value;
}
function object$4(value, name, fields) {
	if (typeof value !== "object" || value === null || Array.isArray(value)) throw new WorkflowRefusal(`${name}: complete supported object required`);
	const result = Object.fromEntries(dataEntries(value));
	if (fields !== void 0 && (Object.keys(result).length !== fields.length || fields.some((key) => !Object.hasOwn(result, key)))) throw new WorkflowRefusal(`${name}: complete supported object required`);
	return result;
}
function freeze(value) {
	if (Array.isArray(value)) return Object.freeze(value.map(freeze));
	if (typeof value === "object" && value !== null) return Object.freeze(Object.fromEntries(dataEntries(value).map(([key, item]) => [key, freeze(item)])));
	return value;
}
/** Return independent lossless wire data without dropping original attempts or opaque extensions. */
function workflowJournalDocument(journal) {
	return object$4(readJson(writeJson({
		schema: workflowJournalSchema,
		body: {
			workflow_digest: journal.workflow_digest,
			source_fingerprint: journal.source_fingerprint,
			runtime_fingerprint: journal.runtime_fingerprint,
			entries: journal.entries,
			state: journal.state,
			evaluations: journal.evaluations
		},
		extensions: journal.extensions
	})), "journal");
}
/** Admit graph, output, dependency and evaluation identities; structural admission is not producer attestation. */
async function parseWorkflowJournal(payload, definition) {
	const text = writeJson(payload);
	if (new TextEncoder().encode(text).length > 67108864) throw new WorkflowRefusal("workflow journal exceeds byte bound");
	const document = object$4(readJson(text), "journal", [
		"schema",
		"body",
		"extensions"
	]);
	if (document["schema"] !== "experiment_workflow_journal.v1") throw new WorkflowRefusal("unsupported workflow journal schema");
	const body = object$4(document["body"], "journal body", [
		"workflow_digest",
		"source_fingerprint",
		"runtime_fingerprint",
		"entries",
		"state",
		"evaluations"
	]);
	const cells = await buildWorkflowCells(definition);
	const workflow_digest = await canonicalDigest("studio.workflow-definition.v1", workflowDocument(definition));
	if (digest(body["workflow_digest"], "workflow") !== workflow_digest) throw new WorkflowRefusal("journal belongs to another workflow definition");
	const source_fingerprint = digest(body["source_fingerprint"], "original source");
	const runtime_fingerprint = digest(body["runtime_fingerprint"], "actual runtime");
	const stages = new Map(definition.stages.map((stage) => [stage.id, stage]));
	const cellIds = new Set(cells.map((cell) => cell.id));
	const rawEntries = body["entries"];
	if (!Array.isArray(rawEntries) || rawEntries.length > 8192) throw new WorkflowRefusal("bounded original journal entries required");
	const entries = [];
	const latest = /* @__PURE__ */ new Map();
	let evaluations = 0n;
	for (const raw of rawEntries) {
		const row = object$4(raw, "attempt", [
			"cell_id",
			"stage_id",
			"fingerprint",
			"dependencies",
			"status",
			"output",
			"output_digest",
			"reason",
			"evaluated"
		]);
		const cid = row["cell_id"], sid = row["stage_id"];
		if (typeof cid !== "string" || !cellIds.has(cid) || typeof sid !== "string" || !stages.has(sid)) throw new WorkflowRefusal("attempt coordinate or stage is absent");
		const stage = stages.get(sid);
		const parents = /* @__PURE__ */ new Set([...stage.depends_on, ...stage.inputs.map((input) => input.source_stage)]);
		const rawDependencies = object$4(row["dependencies"], "attempt dependencies");
		if (Object.keys(rawDependencies).length !== parents.size || Object.keys(rawDependencies).some((parent) => !parents.has(parent))) throw new WorkflowRefusal("attempt dependencies differ from original graph");
		const dependencies = Object.fromEntries(Object.entries(rawDependencies).map(([parent, value]) => [parent, value === null ? null : digest(value, "parent output")]));
		const status = row["status"], reason = row["reason"];
		if (typeof status !== "string" || !statuses.includes(status)) throw new WorkflowRefusal("unsupported original attempt status");
		if (status === "complete") {
			if (reason !== null || row["output"] === null) throw new WorkflowRefusal("completed attempt requires original output without failure reason");
			for (const [parent, parentDigest] of Object.entries(dependencies)) {
				const original = latest.get(`${cid}:${parent}`);
				if (original === void 0 || original.status !== "complete" || parentDigest !== original.output_digest) throw new WorkflowRefusal("completed attempt lacks its original completed dependency");
			}
		} else if (typeof reason !== "string" || Array.from(reason).length < 1 || Array.from(reason).length > 2048) throw new WorkflowRefusal("incomplete attempt requires a bounded original reason");
		const output = row["output"], outputDigest = row["output_digest"];
		if (output === null) {
			if (outputDigest !== null) throw new WorkflowRefusal("absent output cannot carry an output digest");
		} else if (digest(outputDigest, "original output") !== await canonicalDigest("studio.workflow-output.v1", output)) throw new WorkflowRefusal("original attempt output digest differs");
		const evaluated = row["evaluated"];
		if (typeof evaluated !== "boolean" || status === "complete" && !evaluated || status === "blocked" && evaluated) throw new WorkflowRefusal("attempt evaluation accounting differs");
		const previous = latest.get(`${cid}:${sid}`);
		if (previous?.status === "complete" && previous.fingerprint === row["fingerprint"]) throw new WorkflowRefusal("completed matching stage cannot be duplicated");
		if (evaluated) evaluations += 1n;
		const attempt = Object.freeze({
			cell_id: cid,
			stage_id: sid,
			fingerprint: digest(row["fingerprint"], "stage input"),
			dependencies: Object.freeze(dependencies),
			status,
			output: freeze(output),
			output_digest: outputDigest === null ? null : digest(outputDigest, "original output"),
			reason: reason === null ? null : reason,
			evaluated
		});
		entries.push(attempt);
		latest.set(`${cid}:${sid}`, attempt);
	}
	if (typeof body["evaluations"] !== "bigint" || body["evaluations"] !== evaluations || evaluations > definition.sweep.evaluation_budget) throw new WorkflowRefusal("journal stage evaluation budget or accounting differs");
	const state = body["state"];
	if (state !== "partial" && state !== "cancelled" && state !== "complete") throw new WorkflowRefusal("unsupported journal completion state");
	if (state === "complete") for (const cell of cells) for (const sid of topologicalOrder(definition)) {
		const entry = latest.get(`${cell.id}:${sid}`);
		if (entry === void 0 || entry.status !== "complete") throw new WorkflowRefusal("journal has incomplete original coordinates");
		for (const [parent, parentDigest] of Object.entries(entry.dependencies)) {
			const original = latest.get(`${cell.id}:${parent}`);
			if (original.status !== "complete" || parentDigest !== original.output_digest) throw new WorkflowRefusal("completed journal dependency has changed");
		}
	}
	const extensions = object$4(document["extensions"], "journal extensions");
	return Object.freeze({
		workflow_digest,
		source_fingerprint,
		runtime_fingerprint,
		entries: Object.freeze(entries),
		state,
		evaluations,
		extensions: Object.freeze(Object.fromEntries(Object.entries(extensions).map(([key, value]) => [key, freeze(value)])))
	});
}
/** Create an empty partial checkpoint bound to actual source/build identities supplied by the owning host. */
async function createWorkflowJournal(definition, source_fingerprint, runtime_fingerprint) {
	return parseWorkflowJournal({
		schema: workflowJournalSchema,
		body: {
			workflow_digest: await canonicalDigest("studio.workflow-definition.v1", workflowDocument(definition)),
			source_fingerprint,
			runtime_fingerprint,
			entries: [],
			state: "partial",
			evaluations: 0n
		},
		extensions: {}
	}, definition);
}
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/workflows/workflowArchive.ts
init_contracts();
init_canonical();
function object$3(value, fields) {
	if (typeof value !== "object" || value === null || Array.isArray(value)) throw new WorkflowRefusal("workflow archive metadata object required");
	const result = Object.fromEntries(dataEntries(value));
	if (Object.keys(result).length !== fields.length || fields.some((key) => !Object.hasOwn(result, key))) throw new WorkflowRefusal("workflow archive fields are incomplete or unsupported");
	return result;
}
function identity(value) {
	if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) throw new WorkflowRefusal("exact original workflow/revision identity required");
	return value;
}
/** Admit the complete original archive and all retained graph/checkpoint metadata without executing or saving. */
async function readWorkflowArchive(json, rawCodecs) {
	const source = await admitWorkspaceArchive(json, rawCodecs);
	const extension = source.manifest.extensions["experiment_workflows"];
	if (extension === void 0) return Object.freeze({
		source,
		selected: null,
		workflows: Object.freeze([])
	});
	const metadata = object$3(extension, [
		"version",
		"selected",
		"items"
	]);
	if (metadata["version"] !== 1n || !Array.isArray(metadata["items"]) || metadata["items"].length < 1 || metadata["items"].length > 32) throw new WorkflowRefusal("unsupported workflow archive version or history bound");
	const workflows = [];
	const seen = /* @__PURE__ */ new Set();
	for (const item of metadata["items"]) {
		const row = object$3(item, [
			"hash",
			"base_revision_hash",
			"definition",
			"journal"
		]);
		const hash = identity(row["hash"]), base_revision_hash = identity(row["base_revision_hash"]);
		const definition = parseWorkflow(row["definition"]);
		if (seen.has(hash) || hash !== await canonicalDigest("studio.workflow-definition.v1", workflowDocument(definition))) throw new WorkflowRefusal("duplicate or changed original archived workflow");
		if (source.revisions[base_revision_hash] === void 0) throw new WorkflowRefusal("original workflow baseline revision is absent");
		const journal = row["journal"] === null ? null : await parseWorkflowJournal(row["journal"], definition);
		seen.add(hash);
		workflows.push(Object.freeze({
			hash,
			base_revision_hash,
			definition,
			journal
		}));
	}
	const selected = identity(metadata["selected"]);
	if (!seen.has(selected)) throw new WorkflowRefusal("selected original workflow is absent");
	return Object.freeze({
		source,
		selected,
		workflows: Object.freeze(workflows)
	});
}
async function retainHistory(previous, next) {
	if (previous.source_fingerprint !== next.source_fingerprint || previous.runtime_fingerprint !== next.runtime_fingerprint || next.entries.length < previous.entries.length || next.evaluations < previous.evaluations) throw new WorkflowRefusal("original checkpoint source, runtime or attempt history changed");
	for (const [index, before] of previous.entries.entries()) {
		const after = next.entries[index];
		if (await canonicalDigest("studio.workflow-attempt.v1", before) === await canonicalDigest("studio.workflow-attempt.v1", after)) continue;
		if (before.status !== "running" || after.status === "running") throw new WorkflowRefusal("original completed or partial attempt cannot be rewritten");
		if (await canonicalDigest("studio.workflow-attempt.v1", {
			...before,
			status: after.status,
			output: after.output,
			output_digest: after.output_digest,
			reason: after.reason
		}) !== await canonicalDigest("studio.workflow-attempt.v1", after)) throw new WorkflowRefusal("original reserved attempt identity changed");
	}
}
/** Preview an additive original workspace save; original transactions still guard the exact prior source text. */
async function archiveWorkflow(json, baseRevision, definition, journal, rawCodecs) {
	const current = await readWorkflowArchive(json, rawCodecs);
	const base_revision_hash = identity(baseRevision);
	if (current.source.revisions[base_revision_hash] === void 0) throw new WorkflowRefusal("original workflow baseline revision is absent");
	const admitted = parseWorkflow(workflowDocument(definition));
	const hash = await canonicalDigest("studio.workflow-definition.v1", workflowDocument(admitted));
	const next = journal === null ? null : await parseWorkflowJournal(workflowJournalDocument(journal), admitted);
	const prior = current.workflows.find((item) => item.hash === hash);
	if (prior !== void 0) {
		if (prior.base_revision_hash !== base_revision_hash) throw new WorkflowRefusal("original graph cannot be rebound to another baseline revision");
		if (prior.journal !== null) {
			if (next === null) throw new WorkflowRefusal("original partial or completed journal cannot be discarded");
			await retainHistory(prior.journal, next);
		}
	} else if (current.workflows.length >= 32) throw new WorkflowRefusal("workflow history bound reached; export original evidence before composing another graph");
	const item = {
		hash,
		base_revision_hash,
		definition: admitted,
		journal: next
	};
	const retained = prior === void 0 ? [...current.workflows, item] : current.workflows.map((entry) => entry.hash === hash ? item : entry);
	const original = current.source;
	const manifest = {
		...original.manifest,
		extensions: {
			...original.manifest.extensions,
			experiment_workflows: {
				version: 1n,
				selected: hash,
				items: retained.map((entry) => ({
					hash: entry.hash,
					base_revision_hash: entry.base_revision_hash,
					definition: workflowDocument(entry.definition),
					journal: entry.journal === null ? null : workflowJournalDocument(entry.journal)
				}))
			}
		}
	};
	return previewWorkspaceArchive(writeJson({
		schema: original.preview.schema,
		manifest,
		members: original.members,
		parameter_units: original.parameterUnits
	}), rawCodecs);
}
/** Select an admitted immutable revision through the original manifest without dropping evidence or workflow history. */
async function selectWorkflowRevision(json, revisionHash, rawCodecs, recordHash) {
	const original = await admitWorkspaceArchive(json, rawCodecs);
	const revision = original.revisions[identity(revisionHash)];
	if (revision === void 0) throw new WorkflowRefusal("original workflow revision is absent");
	let references = original.manifest.body["artefact_refs"];
	if (recordHash !== void 0) {
		const hash = identity(recordHash);
		if (!references.some((ref) => ref.sha256 === hash) || original.documents[hash]?.schema !== "local_run_record.v1") throw new WorkflowRefusal("original indexed workflow run is absent");
		references = [...references.filter((ref) => ref.sha256 !== hash), ...references.filter((ref) => ref.sha256 === hash)];
	}
	const manifest = {
		...original.manifest,
		body: {
			...original.manifest.body,
			draft_ref: {
				schema: revision.schema,
				sha256: revisionHash,
				media_type: "application/json"
			},
			artefact_refs: references
		}
	};
	return previewWorkspaceArchive(writeJson({
		schema: original.preview.schema,
		manifest,
		members: original.members,
		parameter_units: original.parameterUnits
	}), rawCodecs);
}
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/results/resultModel.ts
function object$2(value, keys) {
	if (typeof value !== "object" || value === null || Array.isArray(value)) throw new ResultRefusal("Result object required");
	const entries = dataEntries(value);
	if (entries.length !== keys.length || entries.some(([key]) => !keys.includes(key))) throw new ResultRefusal("Missing or unsupported result metadata fields");
	return Object.fromEntries(entries);
}
function forbiddenResultControl(value) {
	for (const character of value) {
		const code = character.charCodeAt(0);
		if (code <= 8 || code === 11 || code === 12 || code >= 14 && code <= 31) return true;
	}
	return false;
}
function text(value, empty = false) {
	if (typeof value !== "string" || value.length > 2048 || forbiddenResultControl(value) || !empty && value.length === 0 || /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(value)) throw new ResultRefusal("Bounded Unicode result text required");
	return value;
}
function finite(value) {
	if (typeof value !== "number" || !Number.isFinite(value)) throw new ResultRefusal("Finite original result coordinate or bound required");
	return value;
}
function interval(value, scalar) {
	if (value === null) return null;
	const raw = object$2(value, [
		"lower",
		"upper",
		"method",
		"level"
	]);
	const lower = finite(raw["lower"]), upper = finite(raw["upper"]);
	const level = raw["level"] === null ? null : finite(raw["level"]);
	if (scalar === null || lower > scalar || scalar > upper || level !== null && (level <= 0 || level >= 1)) throw new ResultRefusal("Source interval must bracket a finite value and declare a valid level");
	return Object.freeze({
		lower,
		upper,
		method: text(raw["method"]),
		level
	});
}
function sample(value, kind) {
	const raw = object$2(value, [
		"coordinate",
		"objectId",
		"columnId",
		"x",
		"y",
		"value",
		"status",
		"interval"
	]);
	const status = raw["status"];
	if (status !== "finite" && status !== "missing" && status !== "nan" && status !== "refused") throw new ResultRefusal("Explicit result sample state required");
	const scalar = status === "finite" ? finite(raw["value"]) : null;
	if (status !== "finite" && raw["value"] !== null) throw new ResultRefusal("Unavailable samples must carry null and their explicit state");
	const y = raw["y"] === null ? null : finite(raw["y"]);
	const columnId = raw["columnId"] === null ? null : text(raw["columnId"]);
	if (kind === "matrix" && (y === null || columnId === null) || kind !== "matrix" && (y !== null || columnId !== null)) throw new ResultRefusal("Matrix coordinates and source column identities required only for matrices");
	return Object.freeze({
		coordinate: finite(raw["coordinate"]),
		objectId: text(raw["objectId"]),
		columnId,
		x: finite(raw["x"]),
		y,
		value: scalar,
		status,
		interval: interval(raw["interval"], scalar)
	});
}
function panel(value, remaining) {
	const raw = object$2(value, [
		"id",
		"title",
		"kind",
		"coordinateLabel",
		"coordinateUnit",
		"xLabel",
		"xUnit",
		"yLabel",
		"yUnit",
		"valueLabel",
		"valueUnit",
		"valueDtype",
		"samples"
	]);
	const kind = raw["kind"];
	if (kind !== "series" && kind !== "histogram" && kind !== "spectrum" && kind !== "matrix") throw new ResultRefusal("Supported source result display form required");
	if (!Array.isArray(raw["samples"]) || raw["samples"].length < 1 || raw["samples"].length > remaining) throw new ResultRefusal("Bounded nonempty original result samples required");
	const dtype = raw["valueDtype"];
	if (dtype !== "float64" && dtype !== "int64") throw new ResultRefusal("Unsupported source result scalar dtype");
	const samples = raw["samples"].map((value) => sample(value, kind));
	if (dtype === "int64" && samples.some((item) => item.value !== null && (!Number.isSafeInteger(item.value) || Object.is(item.value, -0)))) throw new ResultRefusal("Original int64 values must remain exact safe integers");
	const identities = /* @__PURE__ */ new Set(), order = /* @__PURE__ */ new Map();
	for (const item of samples) {
		const identity = JSON.stringify([
			item.coordinate,
			item.objectId,
			item.columnId
		]);
		if (identities.has(identity)) throw new ResultRefusal("Duplicate source sample identity");
		identities.add(identity);
		const prior = order.get(item.objectId);
		if (kind === "series" && prior !== void 0 && item.x <= prior) throw new ResultRefusal("Series coordinates must retain strictly increasing source order per object");
		order.set(item.objectId, item.x);
	}
	return Object.freeze({
		id: text(raw["id"]),
		title: text(raw["title"]),
		kind,
		coordinateLabel: text(raw["coordinateLabel"]),
		coordinateUnit: text(raw["coordinateUnit"]),
		xLabel: text(raw["xLabel"]),
		xUnit: text(raw["xUnit"]),
		yLabel: text(raw["yLabel"], true),
		yUnit: text(raw["yUnit"], true),
		valueLabel: text(raw["valueLabel"]),
		valueUnit: text(raw["valueUnit"]),
		valueDtype: dtype,
		samples: Object.freeze(samples)
	});
}
/** Admit a bounded source projection before replacing visible result state. */
function admitResultSnapshot(value) {
	const raw = object$2(value, [
		"version",
		"title",
		"caption",
		"claimBoundary",
		"sourceSha256",
		"partial",
		"panels"
	]);
	if (raw["version"] !== 1) throw new ResultRefusal("Unsupported result metadata major version");
	if (typeof raw["sourceSha256"] !== "string" || !/^[0-9a-f]{64}$/.test(raw["sourceSha256"])) throw new ResultRefusal("Exact lowercase raw source SHA256 required");
	if (typeof raw["partial"] !== "boolean") throw new ResultRefusal("Producer completeness declaration required");
	if (!Array.isArray(raw["panels"]) || raw["panels"].length < 1 || raw["panels"].length > resultLimits.panels) throw new ResultRefusal("Bounded nonempty result panels required");
	let remaining = resultLimits.samples;
	const ids = /* @__PURE__ */ new Set();
	const panels = raw["panels"].map((value) => {
		const result = panel(value, remaining);
		if (ids.has(result.id)) throw new ResultRefusal("Unique result panel identities required");
		ids.add(result.id);
		remaining -= result.samples.length;
		return result;
	});
	const first = panels[0];
	if (first === void 0) throw new ResultRefusal("Bounded nonempty result panels required");
	if (panels.some((item) => item.coordinateLabel !== first.coordinateLabel || item.coordinateUnit !== first.coordinateUnit)) throw new ResultRefusal("Linked panels must declare identical coordinate meaning and units");
	return Object.freeze({
		version: 1,
		title: text(raw["title"]),
		caption: text(raw["caption"]),
		claimBoundary: text(raw["claimBoundary"]),
		sourceSha256: raw["sourceSha256"],
		partial: raw["partial"],
		panels: Object.freeze(panels)
	});
}
var ResultRefusal, resultLimits;
var init_resultModel = __esmMin((() => {
	init_canonical();
	ResultRefusal = class extends Error {};
	resultLimits = Object.freeze({
		/** Maximum linked source panels. */ panels: 16,
		/** Maximum original samples across all panels. */ samples: 65536,
		/** Maximum displayed original markers per panel. */ displayPoints: 1e3,
		/** Maximum raw table rows per page. */ tableRows: 20,
		/** Maximum original UTF-8 producer import bytes. */ importBytes: 2 * 1024 * 1024
	});
}));
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/results/resultSources.ts
var resultSources_exports = /* @__PURE__ */ __exportAll({
	inspectAnalyseProducer: () => inspectAnalyseProducer,
	inspectKuramotoResult: () => inspectKuramotoResult
});
function object$1(value) {
	if (typeof value !== "object" || value === null || Array.isArray(value)) throw new ResultRefusal("Original producer record object required");
	return Object.fromEntries(dataEntries(value));
}
function displayNumbers(value) {
	if (typeof value === "bigint") {
		if (value < BigInt(Number.MIN_SAFE_INTEGER) || value > BigInt(Number.MAX_SAFE_INTEGER)) throw new ResultRefusal("Producer integer cannot be represented exactly in this display");
		return Number(value);
	}
	if (Array.isArray(value)) return value.map(displayNumbers);
	if (typeof value === "object" && value !== null) return Object.fromEntries(dataEntries(value).map(([key, entry]) => [key, displayNumbers(entry)]));
	return value;
}
/** Read a bounded original executive analyse JSON export, preserving its exact raw-byte identity. */
async function inspectAnalyseProducer(json) {
	const bytes = new TextEncoder().encode(json);
	if (bytes.length < 1 || bytes.length > resultLimits.importBytes) throw new ResultRefusal("Bounded original producer JSON required");
	const source = object$1(readJson(json));
	const request = object$1(source["request"]), plan = object$1(source["plan"]), result = object$1(source["result"]);
	if (request["verb"] !== "analyse" || plan["verb"] !== "analyse" || result["status"] !== "succeeded") throw new ResultRefusal("Successful original analyse record required; no execution is requested");
	const outputs = object$1(result["outputs"]);
	if (outputs["analysis_schema"] !== "studio.sync-analysis.v1") throw new ResultRefusal("Unsupported original analysis schema major version");
	const inspection = object$1(displayNumbers(outputs["inspection"]));
	if (inspection["claimBoundary"] !== plan["claim_boundary"]) throw new ResultRefusal("Inspection claim differs from its original analysis plan");
	return admitResultSnapshot({
		...inspection,
		sourceSha256: await artifactBytesDigest(bytes)
	});
}
/** Project a disposed original WASM run without creating phases, histograms or uncertainty estimates. */
async function inspectKuramotoResult(plan, outcome) {
	if (!outcome.ok || !outcome.disposed || outcome.revisionHash !== plan.revisionHash || outcome.planHash !== plan.planHash || outcome.buildFingerprint !== plan.buildFingerprint) throw new ResultRefusal("Disposed current-source successful original run required");
	if (outcome.run.orderParameter.length !== plan.request.steps + 1 || outcome.run.thetaFinal.length !== plan.request.omega.length) throw new ResultRefusal("Original run shape differs from its source plan");
	const output = await makeArtifact("output", {
		revision_hash: outcome.revisionHash,
		plan_hash: outcome.planHash,
		kernel_sha256: outcome.buildFingerprint,
		order_parameter: Array.from(outcome.run.orderParameter, encodeFloat64),
		theta_final: Array.from(outcome.run.thetaFinal, encodeFloat64)
	});
	const grid = (index) => index * plan.request.dt;
	const finalCoordinate = grid(plan.request.steps);
	const axes = {
		kind: "series",
		coordinateLabel: "Requested integration grid",
		coordinateUnit: "model-time",
		yLabel: "",
		yUnit: "",
		valueDtype: "float64"
	};
	return admitResultSnapshot({
		version: 1,
		title: "Original classical Kuramoto run",
		caption: "Source revision " + plan.revisionHash + " · plan " + plan.planHash + " · Rust fixed-step RK4 float64. Coordinates are the requested i×dt grid; the kernel does not report measured timestamps. Only final phases are available. No uncertainty estimated.",
		claimBoundary: plan.admission.claimBoundary,
		sourceSha256: output.sha256,
		partial: false,
		panels: [{
			...axes,
			id: "order",
			title: "Original order parameter",
			xLabel: "Requested integration grid",
			xUnit: "model-time",
			valueLabel: "R",
			valueUnit: "1",
			samples: Array.from(outcome.run.orderParameter, (value, index) => ({
				coordinate: grid(index),
				objectId: "order-parameter",
				columnId: null,
				x: grid(index),
				y: null,
				value,
				status: "finite",
				interval: null
			}))
		}, {
			...axes,
			id: "final-phases",
			title: "Original final phases",
			xLabel: "Oscillator index",
			xUnit: "index",
			valueLabel: "Phase",
			valueUnit: "rad",
			samples: Array.from(outcome.run.thetaFinal, (value, index) => ({
				coordinate: finalCoordinate,
				objectId: `oscillator/${index}`,
				columnId: null,
				x: index,
				y: null,
				value,
				status: "finite",
				interval: null
			}))
		}]
	});
}
var init_resultSources = __esmMin((() => {
	init_contracts();
	init_canonical();
	init_kuramotoArtifacts();
	init_resultModel();
}));
//#endregion
//#region ../../../../.coordination/evidence/SCPN-QUANTUM-COMPUTING-SYSTEMS/F09_BB2DF1_CI_2026-10-15_codex-01a1344d/WF01/relocated-f09/source/studio-web/src/shared/workflows/workflowExecution.ts
init_kuramoto();
init_contracts();
init_kuramotoResources();
init_kuramotoArtifacts();
function object(value) {
	if (typeof value !== "object" || value === null || Array.isArray(value)) throw new WorkflowRefusal("original workflow object required");
	return value;
}
function select(value, path) {
	for (const key of path) {
		const current = object(value);
		if (!Object.hasOwn(current, key)) throw new WorkflowRefusal("original workflow output port is absent");
		value = current[key];
	}
	return value;
}
function scalarValues(value, shape, dtype) {
	if (shape.length > 0) {
		if (!Array.isArray(value) || BigInt(value.length) !== shape[0]) throw new WorkflowRefusal("workflow override differs from original parameter shape");
		return value.flatMap((item) => scalarValues(item, shape.slice(1), dtype));
	}
	if (dtype === "float64" && typeof value === "number") return [encodeFloat64(value)];
	if ((dtype === "int64" || dtype === "uint64") && typeof value === "bigint") return [value.toString()];
	throw new WorkflowRefusal("workflow override differs from original parameter dtype");
}
function parameterCandidate(source, values) {
	let draft = createParameterDraft(source);
	const updated = { ...values["parameters"] === void 0 ? draft.snapshot.parameters : object(values["parameters"]) };
	for (const [key, value] of Object.entries(values)) {
		if (key === "parameters") continue;
		const original = draft.snapshot.parameters[key];
		if (original === void 0) throw new WorkflowRefusal("workflow parameter is absent from the original specification");
		updated[key] = {
			dtype: original.dtype,
			shape: original.shape,
			values: scalarValues(value, original.shape, original.dtype)
		};
	}
	draft = parameterDraftReducer(draft, {
		type: "replace",
		parameters: updated,
		units: source.units
	});
	if (draft.refusal !== null) throw new WorkflowRefusal("original parameter specification refused the workflow coordinate");
	return validateParameterSnapshot(source, draft.snapshot);
}
function parameters(stage, cell, completed, stages) {
	const result = {
		...stage.parameters,
		...cell.overrides[stage.id]
	};
	for (const input of stage.inputs) {
		const parent = completed.get(input.source_stage);
		const port = stages.get(input.source_stage).outputs.find((output) => output.name === input.source_port);
		result[input.parameter] = validatePortValue(port.type, select(parent.output, port.path));
	}
	return result;
}
async function previewOutput(plan, source) {
	return {
		producer_identity: "studio-web/src/shared/experiments/experimentPlan.ts#prepareExperimentPlan",
		preview_only: true,
		revision_hash: plan.revisionHash,
		plan_hash: plan.planHash,
		build_fingerprint: plan.buildFingerprint,
		parameters: source.revision.body["parameters"],
		original_plan: await readExperimentArtifact("plan", artifactContent(plan.planMember))
	};
}
async function savedRun(json, reference, options) {
	const ref = object(reference);
	if (Object.keys(ref).length !== 4 || [
		"record_ref",
		"revision_hash",
		"plan_hash",
		"build_fingerprint"
	].some((key) => typeof ref[key] !== "string")) throw new WorkflowRefusal("complete original workflow run reference required");
	const archive = await selectWorkflowRevision(json, ref["revision_hash"], options.rawCodecs, ref["record_ref"]);
	const record = (await readWorkflowArchive(archive.json, options.rawCodecs)).source.documents[ref["record_ref"]];
	if (record === void 0 || record.body["revision_hash"] !== ref["revision_hash"] || record.body["plan_hash"] !== ref["plan_hash"]) throw new WorkflowRefusal("original indexed run belongs to another workflow source");
	const replay = await prepareSavedReplay(archive.json, options.kernel, options.rawCodecs);
	if (replay.plan.planHash !== ref["plan_hash"] || replay.plan.buildFingerprint !== ref["build_fingerprint"]) throw new WorkflowRefusal("original saved run fingerprint differs from current WASM and plan");
	const outcome = {
		ok: true,
		disposed: true,
		runId: record.body["run_id"],
		revisionHash: replay.plan.revisionHash,
		planHash: replay.plan.planHash,
		buildFingerprint: replay.plan.buildFingerprint,
		run: {
			orderParameter: Float64Array.from(replay.expected.orderParameter, decodeFloat64),
			thetaFinal: Float64Array.from(replay.expected.thetaFinal, decodeFloat64)
		}
	};
	return {
		plan: replay.plan,
		outcome
	};
}
function archivedEvents(events) {
	return events.map((event) => ({
		...event,
		version: 1n,
		sequence: BigInt(event.sequence),
		payload: Object.fromEntries(Object.entries(event.payload).map(([key, value]) => [key, ArrayBuffer.isView(value) && Object.prototype.toString.call(value) === "[object Float64Array]" ? {
			dtype: "float64",
			shape: [BigInt(value.length)],
			values: Array.from(value, encodeFloat64)
		} : value]))
	}));
}
function originalCodecs(codecs, capacity) {
	const scoped = /* @__PURE__ */ new Map();
	for (const [schema, verifier] of codecs) {
		if (localExperimentCodecs.get(schema) !== verifier) {
			scoped.set(schema, verifier);
			continue;
		}
		const verified = /* @__PURE__ */ new Map();
		scoped.set(schema, async (content) => {
			const hash = await artifactBytesDigest(content);
			const known = verified.get(hash);
			if (known !== void 0 && known.bytes.length === content.length && known.bytes.every((byte, index) => byte === content[index])) return known.identity;
			const identity = await verifier(content);
			if (verified.size < capacity.limit) verified.set(hash, {
				bytes: content.slice(),
				identity
			});
			return identity;
		});
	}
	return scoped;
}
/** Run original local validate/simulate/analyse stages, checkpointing the actual original producer history. */
async function runLocalWorkflow(options) {
	const definition = parseWorkflow(workflowDocument(options.definition));
	const capacity = { limit: Math.min(1e3, Number(definition.sweep.evaluation_budget)) };
	options = {
		...options,
		rawCodecs: originalCodecs(options.rawCodecs, capacity)
	};
	if (definition.stages.some((stage) => stage.adapter !== "local-kuramoto" || stage.backend !== "shipped-kuramoto-wasm-float64")) throw new WorkflowRefusal("browser execution requires the original classical Kuramoto WASM adapter; executive graphs run through the local CLI");
	const current = await readWorkflowArchive(options.sourceJson, options.rawCodecs);
	capacity.limit = Math.min(1e3, capacity.limit + current.source.preview.rawHashes.length);
	const baseline = await readLocalExperiment(options.sourceJson, options.rawCodecs, options.baseRevision);
	const baseParameters = await parameterSourceFromArchive(options.sourceJson, options.rawCodecs, options.baseRevision);
	await prepareExperimentPlan(baseline, options.kernel, options.planOptions);
	const sourceFingerprint = await canonicalDigest("studio.workflow-local-source.v1", {
		revision_hash: baseline.revisionHash,
		units: baseline.archive.parameterUnits,
		specs: baseline.archive.parameterSpecs,
		environment: baseline.environmentMember.sha256,
		policy: baseline.policy
	});
	const runtimeFingerprint = await canonicalDigest("studio.workflow-local-runtime.v1", {
		kernel_sha256: baseline.kernelHash,
		bounds: options.kernel.bounds,
		product_policy: browserResourcePolicy(options.kernel.bounds),
		adapter: "original classical Kuramoto WASM"
	});
	const digest = await canonicalDigest("studio.workflow-definition.v1", workflowDocument(definition));
	const stored = current.workflows.find((item) => item.hash === digest);
	if (stored !== void 0 && stored.base_revision_hash !== options.baseRevision) throw new WorkflowRefusal("original workflow baseline changed");
	let journal = stored?.journal ?? await createWorkflowJournal(definition, sourceFingerprint, runtimeFingerprint);
	if (journal.source_fingerprint !== sourceFingerprint || journal.runtime_fingerprint !== runtimeFingerprint) throw new WorkflowRefusal("original workflow source or runtime changed; prior journal remains retained");
	let archive = current.source.preview;
	let persistedJson = options.sourceJson;
	let disposalConfirmed = true;
	let entries = [...journal.entries];
	const checkpoint = async (state) => {
		const document = workflowJournalDocument(journal);
		const body = object(document["body"]);
		body["entries"] = entries;
		body["state"] = state;
		body["evaluations"] = BigInt(entries.filter((entry) => entry.evaluated).length);
		const next = await parseWorkflowJournal(document, definition);
		const candidate = await archiveWorkflow(archive.json, options.baseRevision, definition, next, options.rawCodecs);
		await options.save(candidate, persistedJson);
		archive = candidate;
		persistedJson = candidate.json;
		journal = next;
		entries = [...next.entries];
		options.onCheckpoint?.({
			archive,
			journal,
			disposalConfirmed
		});
	};
	let recovered = false;
	entries = entries.map((entry) => {
		if (entry.status !== "running") return entry;
		recovered = true;
		return {
			...entry,
			status: "interrupted",
			reason: "Original worker attempt has no terminal checkpoint"
		};
	});
	if (recovered) await checkpoint("partial");
	const stages = new Map(definition.stages.map((stage) => [stage.id, stage]));
	const latest = new Map(entries.map((entry) => [`${entry.cell_id}:${entry.stage_id}`, entry]));
	let finished = true;
	const cells = await buildWorkflowCells(definition);
	let settledStages = 0;
	const progress = (cellId, stageId, reused) => {
		++settledStages;
		options.onProgress?.({
			workflowDigest: digest,
			cellId,
			stageId,
			settledStages,
			totalStages: cells.length * definition.stages.length,
			reused
		});
	};
	for (const cell of cells) {
		const completed = /* @__PURE__ */ new Map();
		for (const sid of topologicalOrder(definition)) {
			if (latest.has(`${cell.id}:${sid}`)) await new Promise((resolve) => setTimeout(resolve, 0));
			if (options.signal.aborted) {
				await checkpoint("cancelled");
				return {
					archive,
					journal,
					disposalConfirmed
				};
			}
			const stage = stages.get(sid);
			const parents = [.../* @__PURE__ */ new Set([...stage.depends_on, ...stage.inputs.map((input) => input.source_stage)])].sort();
			const dependencies = Object.fromEntries(parents.map((parent) => [parent, completed.get(parent)?.output_digest ?? null]));
			const base = {
				cell_id: cell.id,
				stage_id: sid,
				source: sourceFingerprint,
				runtime: runtimeFingerprint,
				dependencies
			};
			if (parents.some((parent) => !completed.has(parent))) {
				const fingerprint = await canonicalDigest("studio.workflow-stage.v1", base);
				const prior = latest.get(`${cell.id}:${sid}`);
				if (prior?.status !== "blocked" || prior.fingerprint !== fingerprint) {
					entries.push({
						cell_id: cell.id,
						stage_id: sid,
						fingerprint,
						dependencies,
						status: "blocked",
						output: null,
						output_digest: null,
						reason: "Original parent did not complete",
						evaluated: false
					});
					await checkpoint("partial");
				}
				finished = false;
				progress(cell.id, sid, false);
				continue;
			}
			const values = parameters(stage, cell, completed, stages);
			let plan = null, selectedParameters = null;
			let refused = false;
			try {
				if (stage.verb !== "analyse") {
					const candidate = parameterCandidate(baseParameters, values);
					let revisionHash = options.baseRevision;
					if (await canonicalDigest("studio.workflow-parameters.v1", candidate.parameters) !== await canonicalDigest("studio.workflow-parameters.v1", baseParameters.revision.body["parameters"])) {
						revisionHash = await documentDigest(await createParameterRevision(baseParameters, candidate));
						if ((await readWorkflowArchive(archive.json, options.rawCodecs)).source.revisions[revisionHash] === void 0) archive = (await appendParameterRevision(archive.json, baseParameters, candidate, options.rawCodecs, (/* @__PURE__ */ new Date()).toISOString())).archive;
					}
					archive = await selectWorkflowRevision(archive.json, revisionHash, options.rawCodecs);
					selectedParameters = await parameterSourceFromArchive(archive.json, options.rawCodecs, revisionHash);
					plan = await prepareExperimentPlan(await readLocalExperiment(archive.json, options.rawCodecs, revisionHash), options.kernel, options.planOptions);
				} else {
					if (Object.keys(values).length !== 1 || values["run_ref"] === void 0) throw new WorkflowRefusal("local analyse requires one original run_ref input");
					plan = (await savedRun(archive.json, values["run_ref"], options)).plan;
				}
			} catch (cause) {
				if (!(cause instanceof WorkflowRefusal || cause instanceof ExperimentRefusal || cause instanceof Error)) throw cause;
				refused = true;
			}
			const fingerprint = await canonicalDigest("studio.workflow-stage.v1", {
				...base,
				parameters: values,
				plan_hash: plan?.planHash ?? null
			});
			const prior = latest.get(`${cell.id}:${sid}`);
			if (prior?.status === "complete" && prior.fingerprint === fingerprint) {
				if (refused || plan === null) throw new WorkflowRefusal("original completed stage no longer has a current plan");
				if (stage.verb === "validate") {
					if (await canonicalDigest("studio.workflow-output.v1", await previewOutput(plan, selectedParameters)) !== prior.output_digest) throw new WorkflowRefusal("original cached preview differs from current source and plan");
				} else {
					const ref = stage.verb === "simulate" ? object(prior.output)["run_ref"] : values["run_ref"];
					const original = await savedRun(archive.json, ref, options);
					if (original.plan.planHash !== plan.planHash) throw new WorkflowRefusal("original cached run plan differs");
					if (stage.verb === "analyse") {
						const { inspectKuramotoResult } = await Promise.resolve().then(() => (init_resultSources(), resultSources_exports));
						if (await canonicalDigest("studio.workflow-output.v1", {
							producer_identity: "studio-web/src/shared/results/resultSources.ts#inspectKuramotoResult",
							run_ref: ref,
							projection: await inspectKuramotoResult(original.plan, original.outcome),
							recorded_source: true
						}) !== prior.output_digest) throw new WorkflowRefusal("original cached projection differs");
					}
				}
				completed.set(sid, prior);
				progress(cell.id, sid, true);
				continue;
			}
			if (journal.evaluations >= definition.sweep.evaluation_budget) {
				await checkpoint("partial");
				return {
					archive,
					journal,
					disposalConfirmed
				};
			}
			const row = {
				cell_id: cell.id,
				stage_id: sid,
				fingerprint,
				dependencies,
				status: "running",
				output: null,
				output_digest: null,
				reason: "Original stage reserved; terminal checkpoint pending",
				evaluated: true
			};
			entries.push(row);
			await checkpoint("partial");
			let output = null, status = "failed";
			let reason = "Original workflow input or plan refused";
			if (options.signal.aborted) {
				status = "cancelled";
				reason = "Cancelled before original stage execution";
			} else if (!refused && plan !== null) try {
				if (!plan.admission.allowed) {
					output = {
						preview_only: true,
						original_plan: await readExperimentArtifact("plan", artifactContent(plan.planMember))
					};
					throw new WorkflowRefusal("original resource plan refused");
				}
				if (stage.verb === "validate") output = await previewOutput(plan, selectedParameters);
				else if (stage.verb === "analyse") {
					const { inspectKuramotoResult } = await Promise.resolve().then(() => (init_resultSources(), resultSources_exports));
					const original = await savedRun(archive.json, values["run_ref"], options);
					output = {
						producer_identity: "studio-web/src/shared/results/resultSources.ts#inspectKuramotoResult",
						run_ref: values["run_ref"],
						projection: await inspectKuramotoResult(original.plan, original.outcome),
						recorded_source: true
					};
				} else {
					const runId = crypto.randomUUID(), attemptId = crypto.randomUUID(), events = [];
					const owned = createOwnedKuramotoRun({
						runId,
						revisionHash: plan.revisionHash,
						planHash: plan.planHash,
						buildFingerprint: plan.buildFingerprint,
						request: plan.request,
						wasmBytes: plan.wasmBytes,
						bounds: plan.bounds,
						resourcePolicy: plan.policy,
						deadlineMs: plan.deadlineMs,
						onEvent: (event) => events.push(event),
						...options.workerFactory === void 0 ? {} : { workerFactory: options.workerFactory }
					});
					const cancel = () => {
						owned.cancel();
					};
					options.signal.addEventListener("abort", cancel, { once: true });
					let outcome;
					let observerFailed = false;
					try {
						try {
							options.onWorker?.(owned);
						} catch {
							observerFailed = true;
							await owned.cancel();
						}
						if (options.signal.aborted) cancel();
						outcome = await owned.result;
					} finally {
						options.signal.removeEventListener("abort", cancel);
						await owned.dispose();
						disposalConfirmed = (await owned.result).disposed;
						try {
							options.onWorker?.(null);
						} catch {
							observerFailed = true;
						}
					}
					const terminalKind = outcome.ok ? "result" : outcome.code === "cancelled" ? "cancelled" : "failed";
					if (!outcome.ok && events.at(-1)?.kind !== terminalKind) events.push({
						version: 1,
						run_id: runId,
						sequence: (events.at(-1)?.sequence ?? 0) + 1,
						kind: terminalKind,
						payload: {
							revision_hash: plan.revisionHash,
							plan_hash: plan.planHash,
							build_fingerprint: plan.buildFingerprint,
							disposed: outcome.disposed,
							reason: outcome.reason,
							code: outcome.code,
							origin: "original client outcome"
						}
					});
					if (outcome.disposed) {
						output = {
							producer_identity: "studio-web/src/workers/kernelClient.ts#createOwnedKuramotoRun",
							disposed: true,
							archive_admitted: false,
							events: archivedEvents(events)
						};
						const saved = await appendExperimentAttempt({
							...plan,
							sourceJson: archive.json
						}, runId, attemptId, events, outcome, options.rawCodecs);
						archive = saved;
						output = {
							producer_identity: "studio-web/src/workers/kernelClient.ts#createOwnedKuramotoRun",
							run_ref: {
								record_ref: saved.runRecordHash,
								revision_hash: plan.revisionHash,
								plan_hash: plan.planHash,
								build_fingerprint: plan.buildFingerprint
							},
							disposed: true,
							outcome: outcome.ok ? "result" : outcome.code
						};
					} else output = {
						producer_identity: "studio-web/src/workers/kernelClient.ts#createOwnedKuramotoRun",
						disposed: false,
						outcome: outcome.code,
						events: archivedEvents(events)
					};
					if (observerFailed) {
						status = "failed";
						reason = "Original worker ownership observer refused; native evidence retained";
					} else if (!outcome.ok) {
						status = outcome.code === "cancelled" ? "cancelled" : "failed";
						reason = outcome.reason;
					}
				}
				if (reason === "Original workflow input or plan refused") {
					status = "complete";
					reason = null;
				}
				if (status === "complete") for (const port of stage.outputs) validatePortValue(port.type, select(output, port.path));
			} catch (cause) {
				status = "failed";
				reason = cause instanceof WorkflowRefusal || cause instanceof ExperimentRefusal ? cause.message : "Original stage output or archive refused; partial evidence retained";
			}
			const completedRow = {
				...row,
				status,
				reason,
				output,
				output_digest: output === null ? null : await canonicalDigest("studio.workflow-output.v1", output)
			};
			entries[entries.length - 1] = completedRow;
			await checkpoint(options.signal.aborted ? "cancelled" : "partial");
			latest.set(`${cell.id}:${sid}`, completedRow);
			if (status === "complete") completed.set(sid, completedRow);
			else finished = false;
			progress(cell.id, sid, false);
			if (!disposalConfirmed || options.signal.aborted) return {
				archive,
				journal,
				disposalConfirmed
			};
		}
	}
	await checkpoint(finished ? "complete" : "partial");
	return {
		archive,
		journal,
		disposalConfirmed
	};
}
//#endregion
export { runLocalWorkflow };
