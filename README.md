# IRecover

**From Infrared to a Clearer World**

IRecover converts infrared and near-infrared imagery into a natural-looking
visible-light RGB **approximation**. The core conversion is deterministic,
classical image processing -- color-space math, tone curves, local
contrast, noise reduction, and sharpening -- **not** a machine-learning
model. An optional **Photo Finish** stage can send the already-converted
image to Cloudinary for additional, restrained photographic finishing.

> Visible colors are approximated from infrared data and may differ from
> the original scene. IRecover does not, and cannot, recover the true
> original visible-light colors -- that information was never captured by
> an infrared sensor.

---

## 1. What IRecover does

1. You upload (or generate) an infrared / near-infrared image.
2. IRecover runs a deterministic pipeline in your browser (in a Web Worker,
   off the main thread) to produce a visible-light-style RGB approximation.
3. You can compare original vs. converted (side by side, before/after
   slider, or a fixed split view), tune parameters live, inspect the
   histogram, and export the result.
4. Optionally, you can turn on **Photo Finish** to send the *already
   converted* image to Cloudinary for a final, restrained photographic
   finishing pass (auto color/contrast, warm/cool tone, film-neutral, or
   custom brightness/contrast/saturation/hue/sharpen). This is off by
   default and always opt-in.

## 2. Scientific limitation

Infrared sensors do not capture the same information a visible-light
sensor does:

- **Near-infrared (NIR)** measures reflected light just beyond red
  (~700-1100 nm). It correlates with scene structure but not with true
  visible color.
- **Thermal (far-infrared / LWIR)** measures *emitted heat*, which has no
  fixed relationship to an object's visible color at all.

IRecover's conversion is an estimate built from tonal structure, not a
measurement of true color. This is stated in the app header, the About
section, and the FAQ, and it never uses language like "restore true
colors" or "perfect color recovery."

## 3. Architecture

```
Browser
  |
  | IR/NIR image (PNG / JPEG / WebP)
  v
Local Processing Worker (deterministic, non-ML)
  |
  | RGB approximation (ImageData)
  v
IRecover Workspace (compare, tune, histogram)
  |
  +---- Download locally (PNG / JPEG / WebP)
  |
  +---- Optional Photo Finish
              |
              | already-converted RGB image only
              v
        Cloudflare Pages Function  (/api/photo-finish)
              |
              | signed, server-side request
              v
          Cloudinary
              |
              | transformed image bytes
              v
        Final RGB image  -->  back to the browser
```

Your **original** infrared image never leaves the browser under any
configuration. Only the **already-converted local result** is ever sent
externally, and only when you explicitly enable Photo Finish or trigger
Geographic Analysis.

### Project layout

```
src/
  components/        UI: header, upload, workspace/comparison, controls,
                      histogram, presets, export, Photo Finish panel,
                      Geographic Analysis panel,
                      marketing sections (Examples / How It Works / About / FAQ)
  processing/         Pure, synchronous, deterministic pixel-processing
                      functions (color space, histogram, normalize, tone
                      mapping, contrast, noise reduction, sharpening,
                      color mapping, presets, image analysis, pipeline,
                      geographicAnalysis.ts -- local land-cover heuristics)
  workers/            imageProcessor.worker.ts -- runs the pipeline off
                      the main thread
  hooks/              useImageProcessor (debounced worker calls),
                      usePhotoFinish (Photo Finish state machine),
                      useGeographicAnalysis (Geographic Analysis state
                      machine -- local analysis + optional Gemini/location)
  services/
    cloudinary.ts               client-safe fetch wrapper -- calls OUR
                                 backend endpoint only, never Cloudinary
                                 directly, never touches credentials
    photoFinishTransforms.ts    pure preset -> Cloudinary transformation
                                 mapping, shared by the client (for preview
                                 text) and the server function (for the
                                 real request)
    geographicAnalysisApi.ts    client-safe fetch wrapper for
                                 /api/geographic-analysis -- never touches
                                 the Gemini key
  utils/              image I/O, file validation, sample test patterns,
                      debounce, exifMetadata.ts (EXIF/GPS reader),
                      geographicValidation.ts (Gemini response validation,
                      shared with the server function)
  types/              image.ts, processing.ts, photoFinish.ts, geographic.ts
functions/
  api/photo-finish.ts          Cloudflare Pages Function -- the ONLY place
                                Cloudinary credentials are used
  api/geographic-analysis.ts   Cloudflare Pages Function -- the ONLY place
                                the Gemini API key is used
  _lib/cloudinaryServer.ts     SHA-1 signing, base64, data-URL validation
                                helpers (server-only)
  _lib/geminiServer.ts         Gemini prompt construction + API call
                                (server-only)
  _lib/nominatimServer.ts      OpenStreetMap reverse geocoding, rate-
                                limited and cached (server-only)
tests/                Vitest unit + integration tests, including full
                      mocked round trips through the Photo Finish and
                      Geographic Analysis APIs
```

## 4. Local (core) processing pipeline

```
Decode -> Channel extraction / intensity recovery -> Percentile auto
levels -> Noise reduction -> Local contrast (CLAHE-like) -> Tone curve
(exposure/contrast/gamma/shadow/highlight) -> Detail-preservation blend
-> Optional deterministic scene heuristics (sky/vegetation tint, NOT
object recognition) -> OKLab color-ramp reconstruction -> Saturation /
temperature / hue -> Precision pipeline (below) -> Unsharp-mask
sharpening -> RGB output
```

Every stage is a pure function over typed arrays; the pipeline as a whole
is deterministic (same input + same settings -> byte-identical output),
which is covered by `tests/pipeline.test.ts`.

### 4a. Precision pipeline (Azusa / AutoTone / ColorCorrectionPipeline / gamut protection)

After the deterministic IR->RGB color mapping produces its 8-bit RGBA
output, and only if "Precision pipeline" is enabled in Advanced
Settings, four more refinement stages run on a **linear-light Float32
working buffer** (converted from/to sRGB once, at the edges, to avoid
repeated float<->8-bit quantization):

```
RGB (8-bit) -> sRGB->linear (Float32) -> Azusa (white balance)
-> AutoTone (tonal correction) -> ColorCorrectionPipeline
(precision color correction) -> Gamut Protection (OKLCH chroma
compression) -> linear->sRGB -> RGB (8-bit)
```

**Naming note:** the task that produced this stage named the three
correction steps "Azusa", "AutoTone" and "ColorCorrectionPipeline" as if
they were external libraries. No browser-compatible npm/JS package by
any of those names implements this functionality (verified by search).
Per that task's own fallback instructions, each is implemented here as a
deterministic, dependency-free TypeScript module that fulfills the
*named role*:

- `src/processing/whiteBalance.ts` (**Azusa**) -- confidence-gated,
  outlier-robust gray-world white balance. Deliberately conservative:
  IR-recovered images don't have a true photographic white reference, so
  correction strength is scaled down automatically for already-colorful
  images (low confidence) and hard-capped regardless of user strength.
- `src/processing/autoTone.ts` (**AutoTone**) -- reuses the existing
  histogram module for percentile-based black/white point refinement,
  with a soft-knee highlight roll-off instead of hard clipping.
- `src/processing/colorCorrectionPipeline.ts` (**ColorCorrectionPipeline**)
  -- tiny channel-balance nudge plus OKLCH-based "vibrance" (chroma
  boosted in midtones, tapered in shadows/highlights and for
  already-saturated pixels), capped well short of oversaturation.
- `src/processing/gamutProtection.ts` -- final safety net: compresses
  out-of-gamut chroma (in OKLCH, preserving lightness and hue) instead of
  naively clamping each channel, which avoids hue-shifting/banding at the
  sRGB gamut boundary.

Each stage is unit-tested independently (`tests/whiteBalance.test.ts`,
`tests/autoTone.test.ts`, `tests/colorCorrectionPipeline.test.ts`,
`tests/gamutProtection.test.ts`) as well as exercised end-to-end through
`tests/pipeline.test.ts`. Diagnostics (gains applied, confidence,
black/white points, clipping %, out-of-gamut %) are returned from
`runPipeline()` as `result.precision` and forwarded through the worker
and `useImageProcessor` hook, for future debug-mode UI.

### 4b. Processing backend (`src/processing/backend.ts`)

`detectCapabilities()` and `resolveBackend()` implement real WebGPU /
WebGL2 / OffscreenCanvas feature detection and Auto/CPU/WebGL2/WebGPU
selection (image-size-aware heuristic, safe fallback with a
user-facing reason when a requested backend isn't available). This is
wired into Advanced Settings (`Processing Engine`) and into
`runPipeline()`'s diagnostics.

**Honest scoping (updated):** the precision-pipeline stages above --
Azusa white balance, AutoTone, ColorCorrectionPipeline's chroma shaping,
and gamut protection -- now have a **real WebGL2 GPU implementation**
(`src/processing/gpu/webgl2Backend.ts` + `shaders.ts`): a single fused
GLSL ES 3.00 fragment shader, run once over an `RGBA32F` float texture
(one upload, one draw call, one readback -- no unnecessary GPU
round-trips), with scalar parameters (gains, black/white points)
estimated on the CPU from the exact same statistics helpers the CPU
path uses (`computeWhiteBalanceGains`, `computeAutoToneParams`,
`computeChannelBalanceGains` -- extracted as the single source of truth
so GPU and CPU can't drift). The shader's numeric constants (chroma-boost
ceiling, gamut bisection step count/epsilon, AutoTone knee) are
interpolated from those same TS modules' exported constants rather than
re-typed by hand. If WebGL2 or `EXT_color_buffer_float` isn't available,
or the shader fails to compile/link/run for any reason, it throws and
`runPipeline()` transparently falls back to the CPU implementation --
`result.precision.gpuAccelerated` and `result.precision.backend` report
which one actually ran, honestly, including the fallback reason.

**Update -- the full pipeline is now GPU-accelerated on WebGL2, not just
the precision sub-stage.** `src/processing/gpu/webgl2FullPipeline.ts` +
`fullPipelineShaders.ts` chain real GLSL ES 3.00 render passes (ping-pong
`R32F`/`RGBA32F` textures) covering noise reduction (`'gaussian'` and
`'bilateral'` methods), local contrast, the global tone curve, detail
preservation, optional scene heuristics, IR->RGB color-ramp
reconstruction (OKLab-interpolated, same preset stops as the CPU ramp),
and sharpening -- built on top of, not replacing, the original
precision-only shader above. `runPipeline()` tries this full-GPU path
first; on success it skips the CPU implementation of steps 4-10
entirely. Every formula is a direct line-for-line port of its CPU
counterpart (same constants, same operation order) specifically so the
two stay visually/numerically equivalent -- see the header comment in
`fullPipelineShaders.ts` for the per-stage mapping and the row-order
(texture-flip) reasoning.

**Large images (GPU tiling).** Images whose width or height exceeds the
GPU's `MAX_TEXTURE_SIZE` are no longer a CPU-fallback case for the full
pipeline: `runFullPipelineWebGL2` hands them to
`gpu/webgl2TiledPipeline.ts`, which runs the same shaders over
overlapping tiles (default <= 2048 px per tile, halo included). Each
tile carries a halo equal to the *sum* of the reaches of the spatial
stages chained before the next full-image sync point (noise reduction +
local contrast + scene variance, then separately sharpening), so every
tile's core is computed from exactly the same inputs as in the untiled
run. Tile edges on the true image border get no halo, so edge clamping
behaves as before. The precision stage's white-balance / AutoTone /
color-correction parameters are whole-image statistics: they are
estimated once from the complete image (same CPU estimators) and only
the pointwise apply runs per tile. Scene heuristics use a shader
*derived* from the original by an asserted substitution so the row
fraction is global, not per-tile (`tiledShaders.ts`); the original
shaders are unmodified. Verified tiled-vs-untiled bit-identical (scene
heuristics: within a 2-level / 0.01% analytic bound, observed 0) --
see `tools/gpu-consistency/README.md`. The tiler throws
`TilingUnsupportedError` (=> CPU fallback) only if the required overlap
leaves no useful tile.

Honestly scoped exclusions, not attempted here: the `'median'`
noise-reduction method (needs a GPU sorting network -- deliberately
deferred rather than rushed; the full-GPU attempt throws its own
explicit error and falls back to CPU whenever it's selected), and
tiling of the *precision-only* GPU path (only reached after a `'median'`
fallback; an oversize image there still gets its precision stage on
CPU).

## WebGPU (WGSL compute)

WebGPU is a real backend, not a label: `src/processing/gpu/webgpu/`
holds 13 WGSL compute kernels (IR->RGB color ramp, gaussian / bilateral /
**median** noise reduction, local contrast, tone curve, scene heuristics,
white balance + AutoTone + color correction + gamut protection, unsharp
mask), a device/pipeline manager, and a tiled orchestrator. Each kernel is
a line-for-line port of the GLSL pass and the CPU reference; the numeric
constants are read from the same CPU modules. Median is implemented here
(exact selection, 9- or 25-element window) although WebGL2 never ported it.

* **Entry point:** WebGPU readback is asynchronous (`mapAsync`), so it runs
  through `runPipelineAsync()` (`processing/pipelineAsync.ts`), which the
  worker now calls. The synchronous `runPipeline()` is unchanged in
  structure and is the fallback tier: **WebGPU -> WebGL2 -> CPU**.
* **Honest reporting:** every result carries `execution`
  (`executionReport.ts`): `executed` names the backend that ACTUALLY
  produced the pixels, `attempts` is the audit trail with failure reasons,
  and `'webgpu'` appears only if WebGPU itself produced the returned
  image (never merely because `navigator.gpu` exists). When WebGPU runs,
  the adapter is reported, including a `software` flag for software
  rasterizers (e.g. SwiftShader). Fixed along the way: the synchronous
  pipeline used to skip WebGL2 and run CPU while reporting
  `resolved: 'webgpu', fellBack: false` whenever `navigator.gpu` existed
  (including `auto` on Chrome for images >= 512x512); it now re-maps to
  what it can really execute and reports the fall back.
* **Tiling:** the same halo planner as WebGL2 (`gpu/tilePlanner.ts`,
  reused unmodified). There is one code path: an image that fits is a
  one-tile plan, so tiled and untiled are the same code with different
  tile sizes, and are bit-identical (compute kernels use exact integer
  texel addressing and an exact global row index, so even scene
  heuristics tile bit-exactly). Tiles are clamped to the device's
  `maxTextureDimension2D`.
* **Resource management:** pipelines cached per device and compiled
  lazily (WGSL diagnostics are read and any error throws); a texture pool
  ping-pongs intermediates and reuses them across tiles; one uniform arena
  and one submit + one `mapAsync` per tile per phase; per-tile
  validation / out-of-memory error scopes; device loss evicts the cached
  device so the next run re-acquires.
* **Verification:** `npm run gpu-consistency:webgpu` runs a real headless
  Chromium against a real WebGPU adapter (see
  `tools/gpu-consistency/README.md`). Chrome's WebGPU needs a secure
  context (`http://localhost`) and, in headless CI without a GPU, the
  SwiftShader Vulkan software adapter -- so those results verify the
  *math and the plumbing* on a software adapter, not hardware speed.

The fallback is layered, not all-or-nothing: if the full-GPU attempt
throws (any reason), `runPipeline()` falls back to the CPU
implementation of steps 4-10 -- which itself still makes its own,
independent attempt at the narrower, original precision-only GPU path
(unchanged, see above) before falling back further to full CPU. So a
GPU failure specific to, say, the color-ramp pass doesn't necessarily
lose GPU acceleration for white balance/AutoTone/color correction/gamut
too. `result.precision.gpuAccelerated` / `.backend` report the truth for
whichever tier actually ran.

**Update -- real, in-browser CPU/GPU numerical consistency verification now
exists and has been run.** `tools/gpu-consistency/` (see its own README
for full detail) launches a real headless Chromium with a real WebGL2
context (via Playwright) and runs the actual shipped
`runFullPipelineWebGL2` / `runPrecisionPipelineWebGL2` functions there --
no reimplementation, no mocked GPU -- comparing every result pixel-for-
pixel against the CPU reference, stage by stage and end-to-end, with
MAE/max-error/percent-over-tolerance metrics. Run it with
`npm i -D playwright esbuild tsx && npx playwright install chromium && npm run gpu-consistency`.

**43/43 checks passed** as of the last run, including every stage at
maximum-stress settings (100%/200% strengths, `quality: 'maximum'`, all 8
presets at extreme saturation/temperature/hue, solid and non-square
images) -- most with **zero measured error**. Tolerances were fixed
before running, based on 8-bit visual equivalence, and were never loosened
to force a pass; no CPU/GPU formula discrepancy was found at any setting.
The one check with any pixels over tolerance is deliberately the single
worst case possible (every stage simultaneously maxed, chaining ~8
nonlinear GPU passes) and shows a `10/255` (~4%) difference on 9 of 2352
pixels -- consistent with ordinary float32-vs-float64 rounding through
that many chained `pow`/`exp`/`cbrt` operations, not a logic bug. See
`tools/gpu-consistency/README.md` for the full results table.

`tests/webgl2PrecisionPipeline.test.ts` and `tests/webgl2FullPipeline.test.ts`
remain as the Vitest-run (no-GPU-environment) fallback-contract and
CPU-regression coverage described below; `tools/gpu-consistency/` is what
now actually closes the real in-browser verification gap those tests
can't reach on their own.

CPU/GPU numeric parity within Vitest's own no-GPU environment:
`tests/webgl2PrecisionPipeline.test.ts` and `tests/webgl2FullPipeline.test.ts`
verify the stats-only helpers stay in exact lockstep with the mutating
CPU functions they're factored out of, verify both GPU modules' fallback
contracts (throw cleanly, never partially mutate buffers, never silently
no-op, and reject `'median'`/oversized-preset/size-mismatch inputs with
their own explicit errors before touching the GPU), and run `runPipeline()`
end-to-end across several settings combinations to guard the
`pipeline.ts` control-flow rewrite against regressing the CPU reference
path.

## 5. Photo Finish architecture (new)

Photo Finish is an **optional, final-stage, opt-in** enhancement. It never
replaces or runs instead of the local conversion -- it only runs after it,
and only on demand (an explicit "Apply Photo Finish" action, never on
every slider movement).

- **Client** (`src/services/cloudinary.ts`): builds a request containing
  the already-converted image (as a PNG data URL, capped at 1600px on the
  long side to keep uploads fast) and the selected preset, and POSTs it to
  our own same-origin `/api/photo-finish` endpoint. The browser never
  talks to Cloudinary directly and never sees a Cloudinary credential.
- **Server** (`functions/api/photo-finish.ts`, a Cloudflare Pages
  Function): validates the request, checks for configured credentials,
  performs a *signed* upload to Cloudinary, builds the requested
  transformation URL, fetches the transformed bytes, returns them to the
  browser as a data URL, and then **deletes the temporary Cloudinary
  asset** (`image/destroy`) so nothing is retained long-term.

### Photo Finish presets -> Cloudinary transformations

| Preset | Transformation approach |
| --- | --- |
| Off | No Cloudinary call is ever made |
| Natural | `e_improve` (restrained auto brightness/contrast/color) |
| Natural + Auto Color | `e_auto_color` |
| Natural + Auto Contrast | `e_auto_contrast` |
| Natural + Auto Color + Contrast | `e_auto_color` then `e_auto_contrast` |
| Warm Photograph | auto color + warm tint + restrained contrast/saturation |
| Cool Photograph | auto color + cool tint + restrained contrast |
| Film Neutral | controlled contrast/saturation + subtle sharpen |
| Custom | user-controlled brightness/contrast/saturation/hue/sharpen |

Deliberately **excluded**: Cloudinary's generative/AI-powered
transformations (`e_gen_restore`, `e_gen_recolor`, generative background
replacement, the AI mode of `e_enhance`, etc). Photo Finish is a
controlled photographic finishing layer, not an image generator --
`tests/photoFinish.test.ts` asserts none of these codes ever appear in a
built transformation string.

See `src/services/photoFinishTransforms.ts` for the exact mapping and
https://cloudinary.com/documentation/transformation_reference for the
underlying Cloudinary transformations.

## 6. Geographic Analysis architecture (new)

Geographic Analysis is a **separate, optional** capability from the core
IR -> RGB conversion. It never feeds back into the conversion pipeline and
never slows down the primary Upload -> Convert -> Compare -> Export flow.
It is triggered only by an explicit "Analyze Geography" click -- never
automatically, never on every slider movement.

```
                    IRecover
                       |
          +------------+------------+
          |                         |
      Conversion                Analysis
          |                         |
     IR -> RGB              Geographic Analysis
          |                         |
    Photo Finish          Scene / Land Analysis
          |                         |
          +------------+------------+
                       |
                  Final Report
```

**Local, in-browser measurements** (never leave the browser, run whenever
the panel is opened):

- `src/utils/exifMetadata.ts` -- a minimal, dependency-free EXIF/GPS
  reader (JPEG APP1 and PNG `eXIf` chunks). Only reports what is actually
  present in the file; never guesses.
- `src/processing/geographicAnalysis.ts` -- deterministic HSV-threshold
  land-cover classification (vegetation / water / bare soil / rock /
  snow-ice / urban / agriculture / open terrain), a terrain-texture
  heuristic, and a built-up-surface heuristic. This is classical image
  processing, **not** machine learning or object detection, and every
  value it produces carries an explicit confidence level.
- NDVI (`computeNdvi`) is calculated **only** when real, distinct NIR and
  Red band samples are supplied. IRecover's normal inputs (a single IR
  channel, or the deterministic RGB approximation) are never passed in as
  if they were real spectral bands -- in practice this means the UI
  reports "Spectral vegetation index unavailable for this image" for
  ordinary uploads, which is the scientifically honest answer.

**Optional server-side interpretation layer** (only runs when the user
clicks "Analyze Geography", and only sends the already-downscaled,
already-converted image -- never the raw original at full resolution):

- **Client** (`src/services/geographicAnalysisApi.ts`,
  `src/hooks/useGeographicAnalysis.ts`): downscales the converted image to
  at most 1024px on the long side, summarizes the local measurements as
  plain text, and POSTs both (plus GPS coordinates already found in the
  file's own metadata, only if the user opts in) to our own same-origin
  `/api/geographic-analysis` endpoint.
- **Server** (`functions/api/geographic-analysis.ts`, a Cloudflare Pages
  Function): validates the request and size limits, optionally reverse-
  geocodes GPS coordinates via OpenStreetMap Nominatim
  (`functions/_lib/nominatimServer.ts`), and optionally calls Gemini
  (`functions/_lib/geminiServer.ts`) with a strict prompt instructing it
  to describe only what's visible/measured, never invent place names or
  landmarks, and return structured JSON.
- **Response validation** (`src/utils/geographicValidation.ts`): Gemini's
  JSON is never trusted or displayed as-is -- it is parsed, every field is
  type/range-checked, unrecognized feature types are coerced to `"other"`,
  and malformed entries are dropped rather than causing the whole response
  to be discarded.

If `GEMINI_API_KEY` is not configured, or Gemini is unreachable/slow/
errors, the endpoint returns a plain "interpretation unavailable" status
(never a fabricated result) and the local measurements remain fully
displayed -- Geographic Analysis degrades gracefully, it never breaks the
rest of the app.

### Reverse geocoding (OpenStreetMap Nominatim)

- Only runs when GPS metadata actually exists in the uploaded file **and**
  the user explicitly opts in via the "look up location" checkbox.
- Called server-side only (`functions/_lib/nominatimServer.ts`), because
  Nominatim's usage policy requires a real identifying User-Agent, which
  browsers do not let client-side JS set on `fetch()`.
- Respects Nominatim's public-service policy: throttled to roughly one
  request/second and results are cached in-memory per Cloudflare isolate
  to avoid repeat lookups for the same coordinates within a session.
- Attribution ("&copy; OpenStreetMap contributors") is included with every
  location result shown in the UI.
- If IRecover ever needs high-volume geocoding, only
  `functions/_lib/nominatimServer.ts` needs to change (e.g. to a
  self-hosted Nominatim instance) -- callers are unaffected.

### The three location cases (Part 24 of the brief)

1. **GPS in EXIF** -- shown directly, and (if the user opts in) reverse-
   geocoded to country/region/district/locality via Nominatim.
2. **No GPS** -- the report says plainly: "Geographic location could not
   be determined from available metadata." Gemini is explicitly told not
   to guess a location from visual appearance alone.
3. **Manual location** -- not yet implemented (see Known limitations); the
   type (`ManualLocationContext`) exists for a future manual-context field.

## 7. Cloudinary setup

You need a free or paid Cloudinary account with:

- `CLOUDINARY_CLOUD_NAME`
- `CLOUDINARY_API_KEY`
- `CLOUDINARY_API_SECRET`

**The API secret must only ever live on the server.** It is read via
`env.CLOUDINARY_API_SECRET` inside `functions/api/photo-finish.ts`
(a Cloudflare Pages Function, which runs on Cloudflare's servers, not in
the browser bundle). It is never:

- imported into anything under `src/`
- prefixed with `VITE_` (which Vite would inline into the client bundle)
- written to `public/`, `localStorage`, or any client-visible response
- committed to the repository (`.env`, `.env.*`, and `.dev.vars` are
  gitignored; only `.env.example`, containing empty placeholders, is
  committed)

## 8. Cloudflare setup

`functions/api/photo-finish.ts` is a [Cloudflare Pages
Function](https://developers.cloudflare.com/pages/functions/) using the
file-based `/functions` routing convention -- `POST /api/photo-finish`
maps directly to this file. `wrangler.toml` configures the Pages project
(`pages_build_output_dir = "dist"`, matching the Vite build output).

Configure the three Cloudinary variables as **encrypted environment
variables** in the Cloudflare Pages dashboard (Settings -> Environment
variables) for both Production and Preview, or in a local, gitignored
`.dev.vars` file for `wrangler pages dev`.

If Cloudinary credentials are not configured, the endpoint returns a
clean `503 { ok: false, code: "not-configured" }` response rather than
crashing or silently pretending to succeed -- the UI shows this plainly
and the local result remains fully usable.

## 9. Environment variables

See `.env.example`:

```
CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=

GEMINI_API_KEY=
```

The frontend (`src/`) does not read any environment variables at all --
it only ever calls the same-origin `/api/photo-finish` and
`/api/geographic-analysis` endpoints. `GEMINI_API_KEY` is optional: if
unset, Geographic Analysis still runs its local, in-browser measurements
and simply reports the Gemini interpretation layer as unavailable.

## 10. Local development

Requires Node.js 18+.

```bash
npm install
```

- `npm run dev` -- Vite dev server only (Photo Finish's `/api/*` route
  will not exist under plain `vite dev`).
- `npm run pages:dev` -- runs the app **through Wrangler** (`wrangler
  pages dev`), which serves both the Vite dev server output and the
  `/functions` API routes together, so Photo Finish and Geographic
  Analysis can be exercised locally. Put real credentials in a local
  `.dev.vars` file for this.
- `npm run test` -- Vitest (`npm run test:watch` for watch mode).
- `npm run build` -- type-check (`tsc -b`) then `vite build`.
- `npm run preview` -- serve the production build locally (static only,
  no `/api` routes -- use `pages:dev` or a real deployment to exercise
  Photo Finish against a production-like build).

## 11. Deployment (Cloudflare Pages)

```bash
npm run build
npm run pages:deploy   # wrangler pages deploy dist
```

Or connect the repository to Cloudflare Pages directly (build command
`npm run build`, output directory `dist`) and set the `CLOUDINARY_*`
variables (for Photo Finish) and `GEMINI_API_KEY` (for Geographic
Analysis' interpretation layer) in the project's environment-variable
settings before your first deploy that uses either feature. Both features
degrade gracefully if their credentials are left unset.

## 12. Privacy model

| Mode | What happens | Header indicator |
| --- | --- | --- |
| Local (default) | Original and converted images never leave the browser | "100% Private -- Processed locally in your browser" |
| Local + Photo Finish | Original stays local; the *already-converted* RGB image is sent to our Cloudflare endpoint, then to Cloudinary, for finishing, and the temporary Cloudinary asset is deleted immediately after | "Photo Finish Active -- Sent to Cloudinary for finishing" |
| Local + Geographic Analysis | Local land-cover/vegetation/water/terrain measurements run entirely in the browser and are shown regardless. Only if the user explicitly clicks "Analyze Geography" is the downscaled converted image (and, only if separately opted in, this file's own GPS coordinates) sent to our Cloudflare endpoint for Gemini interpretation and/or Nominatim reverse geocoding | Geographic Analysis panel states exactly what was sent, per request |

IRecover never displays "100% Private" while Photo Finish is on, and
never claims zero upload when an upload is genuinely happening. The same
honesty applies to Geographic Analysis: it is never triggered
automatically, and the panel always states plainly whether anything left
the browser for the report currently shown.

## 13. Error handling & fallback

- Missing/invalid file, corrupted image, oversized image, and worker
  failures are all caught and shown as plain-language messages (never
  stack traces).
- If Photo Finish fails for any reason (network error, Cloudinary
  authentication/transformation failure, timeout, missing server
  configuration), the app never breaks -- it shows the error and a "Use
  Local Result Instead" action, and the local IRecover RGB result and its
  download remain fully available throughout.
- The Cloudflare function never echoes Cloudinary error bodies, stack
  traces, or credential values back to the browser (`tests/photoFinish.test.ts`
  specifically checks that a configured secret never appears in any
  response body).
- If Geographic Analysis' Gemini call fails, times out, or returns a
  malformed response, the endpoint reports "unavailable" rather than
  erroring the whole feature -- the local measurements the browser already
  computed stay visible either way, and the core IR -> RGB conversion is
  completely unaffected (`tests/geographicApi.test.ts` covers missing key,
  upstream HTTP failure, timeout, and malformed-JSON cases).

## 14. Supported formats

- Upload: PNG, JPEG, WebP
- Export: PNG, JPEG, WebP (JPEG/WebP quality is adjustable)
- TIFF is not supported (browsers can't decode it natively; would need an
  added library such as `UTIF.js`)

## 15. Tests

```bash
npm run test
```

Covers (see `tests/`): color-space round-trips (sRGB/linear/HSL/OKLab),
histogram edge cases (black/white/uniform/noisy images), tone-curve
behavior, preset consistency and determinism, full-pipeline regression
(dimensions, determinism, NaN/range safety, transparent/degenerate
images), Photo Finish preset -> transformation mapping (including an
assertion that no generative/AI transformation codes are ever produced),
Cloudinary request signing (SHA-1, against the FIPS 180 "abc" test
vector), data-URL validation, and the Photo Finish API's request
validation, missing-credential fallback, secret-leak prevention, and a
full successful round trip against a mocked Cloudinary.

Geographic Analysis is covered by:

- `tests/geographicAnalysis.test.ts` -- local land-cover classification
  against synthetic solid-color images (green -> vegetation, dark blue ->
  water, near-white/low-saturation -> snow/ice), confidence levels always
  present, terrain confidence never claims certainty, and NDVI gating
  (unavailable without real bands; correct formula when real bands are
  supplied).
- `tests/geographicValidation.test.ts` -- Gemini JSON-fence stripping and
  strict response validation (malformed payloads rejected, invalid
  feature entries dropped rather than failing the whole response,
  confidence clamped to [0,1], unrecognized feature types coerced to
  `"other"`).
- `tests/exifMetadata.test.ts` -- EXIF/GPS extraction from a synthetic
  JPEG byte stream (including correct N/S/E/W sign handling), graceful
  "no metadata" and "corrupt file" handling.
- `tests/geographicApi.test.ts` -- the `/api/geographic-analysis` endpoint:
  request validation, size limits, missing-`GEMINI_API_KEY` fallback,
  upstream Gemini failure/timeout handled as "unavailable" (not a hard
  error), malformed-Gemini-JSON handling, a full successful round trip
  against a mocked Gemini response, and a check that raw upstream error
  bodies are never forwarded to the client.

## 16. Security notes

- Cloudinary's API secret is read only inside `functions/api/photo-finish.ts`
  (server-side Cloudflare runtime) and is never referenced anywhere under
  `src/`.
- All Cloudinary requests from our server are signed per Cloudinary's
  documented scheme (SHA-1 over sorted parameters + secret) --
  https://cloudinary.com/documentation/authentication_signatures
- The endpoint validates: HTTP method, JSON body shape, preset enum
  membership, image data-URL format/MIME, and a request size ceiling
  (15 MB) before ever contacting Cloudinary.
- Temporary Cloudinary assets are uploaded to a dedicated folder and
  explicitly deleted (`image/destroy`) after the transformed bytes are
  fetched, so user images are not retained long-term.
- No image contents or credentials are logged.
- `GEMINI_API_KEY` is read only inside `functions/_lib/geminiServer.ts`
  (server-side Cloudflare runtime), sent via the `x-goog-api-key` request
  header (not a query string, so it never ends up in logs/URLs), and is
  never referenced anywhere under `src/`.
- The Geographic Analysis endpoint validates JSON body shape, image
  data-URL format/MIME, an image size ceiling, a request body size
  ceiling, and GPS coordinate ranges before contacting Gemini or
  Nominatim -- and it never calls either automatically (always requires
  the explicit "Analyze Geography" action), which keeps request volume
  and cost bounded.
- Gemini's response is treated as untrusted input: parsed, schema-
  validated, and range/type-checked (`src/utils/geographicValidation.ts`)
  before anything from it reaches the UI. Raw upstream error bodies from
  either Gemini or Cloudinary are never forwarded to the browser.
- Nominatim reverse geocoding only ever sends the coordinates already
  present in the uploaded file's own metadata (never the user's live
  browser location, which IRecover does not request), and only when the
  user explicitly opts in.

## 17. Known limitations

- TIFF input is not supported.
- No PWA/offline shell yet.
- Scene heuristics (sky/vegetation tinting) are simple, deterministic
  pattern rules -- not object recognition -- by design.
- Photo Finish input is capped at 1600px on the long side before upload
  to keep requests fast; the local IRecover RGB download is still
  available at full resolution regardless of this cap.
- **The Cloudinary/Cloudflare integration was implemented and covered by
  tests against a mocked Cloudinary API, but could not be live-tested
  against a real Cloudinary account or a real Cloudflare Pages deployment
  in this environment** (no network access, no credentials available).
  Please verify against a real account before relying on it in
  production.
- **Geographic Analysis' local heuristics are classical HSV-threshold
  rules, not a trained land-cover classifier** -- treat every percentage
  and label as a rough estimate, exactly as the UI labels it. They will
  misclassify atypical scenes (e.g. blue rooftops, red rock that reads as
  "bare soil", etc).
- **NDVI is effectively always "unavailable" for normal IRecover uploads**
  by design -- the app's inputs are a single IR channel or a false-color
  capture, not calibrated multi-band imagery with a real, separate Red
  channel, so fabricating an index from the RGB approximation would be
  scientifically dishonest. `computeNdvi` is fully implemented and tested
  for the case where real NIR+Red band data is available.
- **Case 2 from the brief (a manually-supplied location used as context)
  is not yet implemented** -- `ManualLocationContext` exists as a type but
  there is no UI field for it yet.
- **The analysis overlay visualization (Part 27 of the original brief) is
  not implemented in this pass** -- the Comparison workspace currently
  offers Original/Recovered/Split as before; a labelled, approximate
  overlay mode for land-cover regions is a reasonable follow-up.
- **The Gemini and Nominatim integrations were implemented and covered by
  tests against mocked responses, but could not be live-tested against
  the real Gemini API or the real Nominatim service in this environment**
  (no network access, no API key available). In particular, double-check
  the pinned Gemini model name (`GEMINI_MODEL` in
  `functions/_lib/geminiServer.ts`) against
  https://ai.google.dev/gemini-api/docs/models before relying on it, in
  case it has been renamed or retired since this was written.
- The "IRecover UI reference image" mentioned in the product brief was not
  actually supplied alongside these instructions, so no visual-matching
  pass against it was performed in this update -- the existing light,
  minimal, three-column layout was preserved and extended rather than
  redesigned.

## 18. License

MIT for the project's own code. See in-code comments for third-party
service terms (Cloudinary, Google Gemini, OpenStreetMap/Nominatim) that
apply only when Photo Finish and/or Geographic Analysis are used.

## 19. WebGPU backend (production-hardening pass)

A WGSL/WebGPU backend sits alongside WebGL2 (`src/processing/gpu/webgpu/`).
Its data flow keeps intermediate results resident in VRAM whenever a run
needs both the precision stage and sharpening: tile cores are copied
GPU-to-GPU across the whole-image statistics sync, sRGB quantisation and
8-bit packing happen in WGSL (`quantSrgb`/`pack8`), and the only mandatory
float readback is the one the CPU white-balance/AutoTone/colour-correction
estimators need (those estimators remain CPU-side by design). Falls back to
WebGL2, then CPU, on any adapter/device/shader/tiling failure, truthfully
reported in `PipelineResult.execution` (`executedOnGpu()` in
`executionReport.ts` reads that report; `isGpuCapableBackend()` in
`backend.ts` is the separate, static "does this backend kind have a GPU
implementation" question). See `HARDENING_NOTES.md` for what changed in the
production-hardening pass and `WEBGPU_AUDIT.md` for the backend's original
provenance/audit record.
