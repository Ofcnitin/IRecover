/**
 * Minimal, no-bundler "browser transpile" step used ONLY because esbuild
 * is not installable in this offline sandbox (no npm registry access).
 * This performs NO code transformation beyond what tsc's own
 * transpileModule (type-stripping) does per-file -- it is not a
 * reimplementation of any pipeline logic. Each .ts file is transpiled
 * 1:1 to a .js file at the same relative path, then relative import
 * specifiers are rewritten to add the .js extension so native browser
 * ES module resolution (no bundling, no node_modules resolution needed
 * since this subgraph has zero npm dependencies) can load the real
 * graph of files directly over HTTP.
 */
import * as ts from 'typescript';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(import.meta.dirname, '../../');
const OUT = path.resolve(ROOT, '.browser-dist');

function resolveImport(fromFile: string, spec: string): string {
  const base = path.resolve(path.dirname(fromFile), spec);
  const candidates = [base + '.ts', base + '.tsx', path.join(base, 'index.ts')];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  throw new Error(`Cannot resolve import "${spec}" from ${fromFile}`);
}

const visited = new Map<string, string>(); // abs src path -> abs out path

function outPathFor(absSrc: string): string {
  const rel = path.relative(ROOT, absSrc).replace(/\.tsx?$/, '.js');
  return path.join(OUT, rel);
}

function processFile(absSrc: string) {
  if (visited.has(absSrc)) return;
  const absOut = outPathFor(absSrc);
  visited.set(absSrc, absOut);

  const source = fs.readFileSync(absSrc, 'utf8');
  const sf = ts.createSourceFile(absSrc, source, ts.ScriptTarget.ES2020, true, absSrc.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS);

  // Collect relative import/export specifiers to recurse into, and to rewrite.
  const specifiers: string[] = [];
  const visit = (node: ts.Node) => {
    if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) && node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier)) {
      const spec = node.moduleSpecifier.text;
      if (spec.startsWith('.')) specifiers.push(spec);
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);

  // Transpile (type-stripping only, matches project's isolatedModules contract).
  const result = ts.transpileModule(source, {
    compilerOptions: {
      target: ts.ScriptTarget.ES2020,
      module: ts.ModuleKind.ESNext,
      jsx: ts.JsxEmit.ReactJSX,
      isolatedModules: true,
      verbatimModuleSyntax: false,
    },
    fileName: absSrc,
    reportDiagnostics: true,
  });

  if (result.diagnostics && result.diagnostics.length) {
    for (const d of result.diagnostics) {
      const msg = ts.flattenDiagnosticMessageText(d.messageText, '\n');
      console.error(`[transpile diagnostic] ${path.relative(ROOT, absSrc)}: ${msg}`);
    }
  }

  let js = result.outputText;
  // Rewrite relative specifiers to point at emitted .js files with correct
  // relative path (output tree mirrors source tree, so specifier text
  // itself doesn't change -- just needs the .js extension appended).
  for (const spec of new Set(specifiers)) {
    const resolvedAbsSrc = resolveImport(absSrc, spec);
    processFile(resolvedAbsSrc); // recurse BEFORE rewriting so it's visited
    const escaped = spec.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(`(['"\`])${escaped}\\1`, 'g');
    js = js.replace(re, (_m, q) => `${q}${spec}.js${q}`);
  }

  fs.mkdirSync(path.dirname(absOut), { recursive: true });
  fs.writeFileSync(absOut, js, 'utf8');
}

const entry = process.argv[2];
if (!entry) {
  console.error('usage: tsx bundle-for-browser.mts <entry-ts-file>');
  process.exit(1);
}
processFile(path.resolve(ROOT, entry));
console.log(`Transpiled ${visited.size} files -> ${OUT}`);
for (const [src, out] of visited) {
  console.log(`  ${path.relative(ROOT, src)} -> ${path.relative(ROOT, out)}`);
}
