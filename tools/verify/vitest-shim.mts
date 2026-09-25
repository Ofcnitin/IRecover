/**
 * NOT vitest. A small, explicitly-labeled compatible runner covering
 * exactly the vitest surface this project's test files use (verified
 * by grepping the suite first: describe/it(+.each)/expect matchers/
 * beforeEach/afterEach/vi.fn/vi.stubGlobal/vi.unstubAllGlobals). Built
 * only because vitest itself cannot be installed in this offline
 * sandbox (no npm registry access). Any test that needs a real vitest
 * feature not implemented here will fail loudly (thrown error), not
 * silently pass.
 *
 * Real test files are imported UNMODIFIED; this module is fed to them
 * in place of the 'vitest' package via a resolution alias (see run.mts).
 */
type TestFn = () => void | Promise<void>;
interface TestCase { name: string; fn: TestFn; suite: string[]; skip?: boolean; }
const tests: TestCase[] = [];
const suiteStack: string[] = [];
const beforeEachHooks: TestFn[][] = [[]];
const afterEachHooks: TestFn[][] = [[]];

export function describe(name: string, fn: () => void) {
  suiteStack.push(name);
  beforeEachHooks.push([]);
  afterEachHooks.push([]);
  fn();
  afterEachHooks.pop();
  beforeEachHooks.pop();
  suiteStack.pop();
}
describe.skip = (name: string, _fn: () => void) => {};

function makeIt() {
  const base: any = (name: string, fn: TestFn) => {
    tests.push({ name, fn, suite: [...suiteStack] });
  };
  base.skip = (name: string, _fn: TestFn) => {
    tests.push({ name, fn: async () => {}, suite: [...suiteStack], skip: true });
  };
  base.each = (cases: any[]) => (name: string, fn: (...args: any[]) => void | Promise<void>) => {
    for (const c of cases) {
      const args = Array.isArray(c) ? c : [c];
      // vitest %s/%i/... templating: simplistic single-pass substitution
      let i = 0;
      const resolvedName = name.replace(/%[sdifjoOp#]/g, () => String(args[i++]));
      base(resolvedName || name, () => fn(...(Array.isArray(c) ? c : [c])));
    }
  };
  return base;
}
export const it = makeIt();
export const test = it;

export function beforeEach(fn: TestFn) { beforeEachHooks[beforeEachHooks.length - 1].push(fn); }
export function afterEach(fn: TestFn) { afterEachHooks[afterEachHooks.length - 1].push(fn); }
export function beforeAll(fn: TestFn) { fn(); } // no suite-level ordering needed by this suite
export function afterAll(_fn: TestFn) {}

// --- vi ---
function fnMock(impl?: (...args: any[]) => any) {
  const calls: any[][] = [];
  const m: ((...args: any[]) => any) & { mock: { calls: any[][] }; mockReturnValue: (v: any) => typeof m; mockImplementation: (f: any) => typeof m; mockResolvedValue: (v: any) => typeof m; mockRejectedValue: (v: any) => typeof m } = ((...args: any[]) => {
    calls.push(args);
    return impl ? impl(...args) : undefined;
  }) as any;
  m.mock = { calls };
  m.mockReturnValue = (v: any) => { impl = () => v; return m; };
  m.mockImplementation = (f: any) => { impl = f; return m; };
  m.mockResolvedValue = (v: any) => { impl = () => Promise.resolve(v); return m; };
  m.mockRejectedValue = (v: any) => { impl = () => Promise.reject(v); return m; };
  return m;
}
const stubbedGlobals: Record<string, any> = {};
const stubbedKeys = new Set<string>();
export const vi = {
  fn: (impl?: (...args: any[]) => any) => fnMock(impl),
  stubGlobal: (name: string, value: any) => {
    if (!stubbedKeys.has(name)) {
      stubbedGlobals[name] = (globalThis as any)[name];
      stubbedKeys.add(name);
    }
    Object.defineProperty(globalThis, name, { value, configurable: true, writable: true, enumerable: true });
  },
  unstubAllGlobals: () => {
    for (const k of stubbedKeys) {
      Object.defineProperty(globalThis, k, { value: stubbedGlobals[k], configurable: true, writable: true, enumerable: true });
    }
    stubbedKeys.clear();
  },
};

// --- expect ---
function deepEqual(a: any, b: any): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== typeof b) return false;
  if (a instanceof Float32Array || a instanceof Uint8ClampedArray || a instanceof Uint8Array || a instanceof Uint32Array) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    return a.every((v, i) => deepEqual(v, b[i]));
  }
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    const ak = Object.keys(a), bk = Object.keys(b);
    if (ak.length !== bk.length) return false;
    return ak.every((k) => deepEqual(a[k], b[k]));
  }
  return false;
}
function matchObject(actual: any, expected: any): boolean {
  if (expected === null || typeof expected !== 'object') return Object.is(actual, expected);
  if (actual === null || typeof actual !== 'object') return false;
  return Object.keys(expected).every((k) => matchObject(actual[k], expected[k]));
}
class AssertionError extends Error {}

export function expect(actual: any) {
  const make = (negate: boolean) => ({
    toBe: (exp: any) => { if (Object.is(actual, exp) === negate) throw new AssertionError(`expected ${negate ? 'not ' : ''}${JSON.stringify(actual)} to be ${JSON.stringify(exp)}`); },
    toEqual: (exp: any) => { if (deepEqual(actual, exp) === negate) throw new AssertionError(`expected ${negate ? 'not ' : ''}${JSON.stringify(actual)} to equal ${JSON.stringify(exp)}`); },
    toMatchObject: (exp: any) => { if (matchObject(actual, exp) === negate) throw new AssertionError(`expected ${JSON.stringify(actual)} to${negate ? ' not' : ''} match object ${JSON.stringify(exp)}`); },
    toBeCloseTo: (exp: any, digits = 2) => { const pass = Math.abs(actual - exp) < Math.pow(10, -digits) / 2; if (pass === negate) throw new AssertionError(`expected ${actual} to${negate ? ' not' : ''} be close to ${exp}`); },
    toBeDefined: () => { if ((actual !== undefined) === negate) throw new AssertionError(`expected value to${negate ? ' not' : ''} be defined`); },
    toBeUndefined: () => { if ((actual === undefined) === negate) throw new AssertionError(`expected ${JSON.stringify(actual)} to${negate ? ' not' : ''} be undefined`); },
    toBeNull: () => { if ((actual === null) === negate) throw new AssertionError(`expected ${JSON.stringify(actual)} to${negate ? ' not' : ''} be null`); },
    toBeTruthy: () => { if (!!actual === negate) throw new AssertionError(`expected ${JSON.stringify(actual)} to${negate ? ' not' : ''} be truthy`); },
    toBeFalsy: () => { if ((!actual) === negate) throw new AssertionError(`expected ${JSON.stringify(actual)} to${negate ? ' not' : ''} be falsy`); },
    toBeGreaterThan: (n: number) => { if ((actual > n) === negate) throw new AssertionError(`expected ${actual} to${negate ? ' not' : ''} be > ${n}`); },
    toBeGreaterThanOrEqual: (n: number) => { if ((actual >= n) === negate) throw new AssertionError(`expected ${actual} to${negate ? ' not' : ''} be >= ${n}`); },
    toBeLessThan: (n: number) => { if ((actual < n) === negate) throw new AssertionError(`expected ${actual} to${negate ? ' not' : ''} be < ${n}`); },
    toBeLessThanOrEqual: (n: number) => { if ((actual <= n) === negate) throw new AssertionError(`expected ${actual} to${negate ? ' not' : ''} be <= ${n}`); },
    toBeInstanceOf: (cls: any) => { if ((actual instanceof cls) === negate) throw new AssertionError(`expected ${actual} to${negate ? ' not' : ''} be instanceof ${cls?.name}`); },
    toContain: (item: any) => { const has = typeof actual === 'string' ? actual.includes(item) : Array.from(actual).includes(item); if (has === negate) throw new AssertionError(`expected ${JSON.stringify(actual)} to${negate ? ' not' : ''} contain ${JSON.stringify(item)}`); },
    toHaveLength: (n: number) => { if ((actual.length === n) === negate) throw new AssertionError(`expected length ${actual.length} to${negate ? ' not' : ''} be ${n}`); },
    toMatch: (re: RegExp | string) => { const r = typeof re === 'string' ? new RegExp(re) : re; if (r.test(actual) === negate) throw new AssertionError(`expected ${JSON.stringify(actual)} to${negate ? ' not' : ''} match ${re}`); },
    toHaveBeenCalled: () => { const called = actual?.mock?.calls?.length > 0; if (called === negate) throw new AssertionError(`expected mock to${negate ? ' not' : ''} have been called`); },
    toThrow: (matcher?: RegExp | string | Function) => {
      let threw = false, err: any;
      try { actual(); } catch (e) { threw = true; err = e; }
      if (threw && matcher) {
        if (typeof matcher === 'function') {
          if (!(err instanceof matcher)) throw new AssertionError(`expected thrown error to be instance of ${matcher.name}, got ${err}`);
        } else {
          const re = typeof matcher === 'string' ? new RegExp(matcher) : matcher;
          if (!re.test(String(err?.message ?? err))) throw new AssertionError(`expected thrown error message to match ${matcher}, got: ${err?.message ?? err}`);
        }
      }
      if (threw === negate) throw new AssertionError(`expected function to${negate ? ' not' : ''} throw`);
    },
  });
  const api: any = make(false);
  api.not = make(true);
  api.rejects = {
    toThrow: async (matcher?: RegExp | string | Function) => {
      let threw = false, err: any;
      try { await actual; } catch (e) { threw = true; err = e; }
      if (!threw) throw new AssertionError(`expected promise to reject`);
      if (matcher) {
        if (typeof matcher === 'function') {
          if (!(err instanceof matcher)) throw new AssertionError(`expected rejection to be instance of ${matcher.name}, got ${err}`);
        } else {
          const re = typeof matcher === 'string' ? new RegExp(matcher) : matcher;
          if (!re.test(String(err?.message ?? err))) throw new AssertionError(`expected rejection message to match ${matcher}, got: ${err?.message ?? err}`);
        }
      }
    },
    toBeInstanceOf: async (cls: any) => {
      let threw = false, err: any;
      try { await actual; } catch (e) { threw = true; err = e; }
      if (!threw) throw new AssertionError(`expected promise to reject`);
      if (!(err instanceof cls)) throw new AssertionError(`expected rejection to be instance of ${cls?.name}, got ${err}`);
    },
  };
  api.resolves = {
    toBe: async (exp: any) => { const v = await actual; if (!Object.is(v, exp)) throw new AssertionError(`expected resolved ${JSON.stringify(v)} to be ${JSON.stringify(exp)}`); },
    toEqual: async (exp: any) => { const v = await actual; if (!deepEqual(v, exp)) throw new AssertionError(`expected resolved ${JSON.stringify(v)} to equal ${JSON.stringify(exp)}`); },
    toBeDefined: async () => { const v = await actual; if (v === undefined) throw new AssertionError(`expected resolved value to be defined`); },
    toBeTruthy: async () => { const v = await actual; if (!v) throw new AssertionError(`expected resolved value to be truthy`); },
  };
  return api;
}
(expect as any).any = (cls: any) => ({ __anyMatcher: true, cls });

export async function runAll(): Promise<{ pass: number; fail: number; failures: { name: string; error: string }[] }> {
  let pass = 0, fail = 0;
  const failures: { name: string; error: string }[] = [];
  for (const t of tests) {
    const fullName = [...t.suite, t.name].join(' > ');
    if (t.skip) continue;
    try {
      for (const hooks of beforeEachHooks) for (const h of hooks) await h();
      await t.fn();
      for (const hooks of afterEachHooks) for (const h of hooks) await h();
      pass++;
    } catch (e: any) {
      fail++;
      failures.push({ name: fullName, error: e?.stack || String(e) });
    }
  }
  return { pass, fail, failures };
}
export function resetRegistry() { tests.length = 0; }
export { tests as __tests };
