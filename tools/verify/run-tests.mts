import '../../tests/setup';
import fs from 'node:fs';
import path from 'node:path';
import { runAll, resetRegistry } from './vitest-shim.mts';

process.on('unhandledRejection', (reason) => {
  console.log('\n[unhandledRejection during import/run]', reason);
});

const testsDir = path.resolve(import.meta.dirname, '../../tests');
const files = fs.readdirSync(testsDir).filter((f) => f.endsWith('.test.ts')).sort();

let totalPass = 0, totalFail = 0;
const fileResults: { file: string; pass: number; fail: number; failures: { name: string; error: string }[]; loadError?: string }[] = [];

for (const file of files) {
  resetRegistry();
  const full = path.join(testsDir, file);
  console.error(`[loading] ${file}`);
  try {
    await import(full + `?t=${Date.now()}`); // cache-bust per file (harmless; each file is a distinct path anyway)
  } catch (e: any) {
    fileResults.push({ file, pass: 0, fail: 0, failures: [], loadError: e?.stack || String(e) });
    continue;
  }
  const { pass, fail, failures } = await runAll();
  totalPass += pass; totalFail += fail;
  fileResults.push({ file, pass, fail, failures });
}

console.log('\n=== Test results (real test files, executed via a labeled non-vitest compatible shim -- see vitest-shim.mts) ===\n');
for (const r of fileResults) {
  if (r.loadError) {
    console.log(`✗ ${r.file}: FAILED TO LOAD\n${r.loadError.split('\n').slice(0, 6).join('\n')}\n`);
    continue;
  }
  const status = r.fail === 0 ? '✓' : '✗';
  console.log(`${status} ${r.file}: ${r.pass} pass, ${r.fail} fail`);
  for (const f of r.failures) {
    console.log(`   FAIL: ${f.name}`);
    console.log('   ' + f.error.split('\n').slice(0, 4).join('\n   '));
  }
}
console.log(`\nTOTAL: ${totalPass} pass, ${totalFail} fail across ${files.length} files`);
process.exit(totalFail > 0 ? 1 : 0);
