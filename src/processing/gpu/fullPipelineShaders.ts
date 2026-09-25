/**
 * WebGL2 shader sources for the "full pipeline" GPU extension.
 * --------------------------------------------------------------
 * Extends the existing, working precision-pipeline shader (shaders.ts /
 * webgl2Backend.ts -- NOT modified by this file) to cover the remaining
 * CPU-only stages: noise reduction, local contrast, the global tone
 * curve, optional scene heuristics, IR->RGB color-ramp reconstruction,
 * and sharpening.
 *
 * Unlike the precision pipeline (which is entirely pointwise and fuses
 * into one shader pass), these stages are spatial -- they read
 * neighboring pixels -- so they're implemented as a small chain of
 * discrete render passes (see webgl2FullPipeline.ts for the orchestrator
 * and ping-pong texture management), sharing ONE reusable separable
 * box-blur pair of programs across every stage that needs one
 * (noise reduction's 'gaussian' method, local contrast, scene-variance,
 * and sharpening's unsharp mask all reduce to the same primitive the CPU
 * reference implementation uses: contrast.ts's boxBlur()).
 *
 * Every formula below is a direct line-for-line port of the matching CPU
 * function -- same constants, same order of operations -- specifically
 * so CPU and GPU output stay visually/numerically equivalent. Where a
 * CPU function short-circuits for performance (e.g. skipping the HSL
 * round-trip when saturation/hue are both zero), the shader instead
 * always runs the full computation, because the skipped branch is
 * mathematically an identity in those cases -- so the *output* still
 * matches, while the shader stays branch-free and simpler to audit.
 *
 * NOT ported to GPU here (still CPU-only): the 'median' noise-reduction
 * method (needs a per-pixel sorting network -- correctness-critical and
 * deliberately deferred rather than rushed) and image tiling for
 * oversized textures (see webgl2FullPipeline.ts's MAX_TEXTURE_SIZE
 * check, which throws and lets the caller fall back to CPU honestly).
 */

const MAX_STOPS = 8;

/**
 * Reused, generic fullscreen-triangle vertex shader -- identical
 * technique (and identical UV convention, so the row-order reasoning
 * below holds) as the existing precision-pipeline vertex shader. Kept
 * as its own copy (rather than importing from shaders.ts) so this file
 * has zero coupling to, and zero risk of disturbing, that already-
 * verified module.
 */
export const FULL_PIPELINE_VERTEX_SRC = `#version 300 es
out vec2 v_uv;
void main() {
  vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  v_uv = pos;
  gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
`;

/**
 * NOTE ON ROW ORDER: v_uv.y == 0 corresponds to row 0 of the originally
 * uploaded source array (top of the image, same as the CPU pipeline's
 * `y=0` convention), because texImage2D/readPixels both preserve the
 * source array's row order in texture-coordinate space, and this
 * vertex shader's v_uv happens to align with that same convention (see
 * the module header of webgl2FullPipeline.ts for the full argument).
 * Any pass below that needs "distance from the top of the frame" (scene
 * heuristics) uses v_uv.y directly -- never gl_FragCoord.y, which is in
 * window/rasterization space and has the opposite vertical sense.
 */

// ---- Separable box blur (mirrors contrast.ts boxBlur(), run as two
//      passes: u_dir=(1,0) then u_dir=(0,1)). Single channel. ----
export const BOX_BLUR_R_FRAG_SRC = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform vec2 u_dir;
uniform int u_radius;
void main() {
  float acc = 0.0;
  int r = u_radius;
  for (int i = -r; i <= r; i++) {
    vec2 uv = clamp(v_uv + u_dir * u_texel * float(i), vec2(0.0), vec2(1.0));
    acc += texture(u_src, uv).r;
  }
  outColor = vec4(acc / float(2 * r + 1), 0.0, 0.0, 1.0);
}
`;

// ---- Same box blur, 3-channel (RGB), used by sharpening's unsharp mask. ----
export const BOX_BLUR_RGB_FRAG_SRC = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform vec2 u_dir;
uniform int u_radius;
void main() {
  vec3 acc = vec3(0.0);
  int r = u_radius;
  for (int i = -r; i <= r; i++) {
    vec2 uv = clamp(v_uv + u_dir * u_texel * float(i), vec2(0.0), vec2(1.0));
    acc += texture(u_src, uv).rgb;
  }
  outColor = vec4(acc / float(2 * r + 1), 1.0);
}
`;

// ---- Generic 2-input lerp (mirrors the `blend()` helper in
//      noiseReduction.ts, and doubles as pipeline.ts's preserveDetail()
//      since `toneMapped + (original-toneMapped)*k` == `mix(toneMapped,
//      original, k)`). Single channel. ----
export const BLEND_R_FRAG_SRC = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_a;
uniform sampler2D u_b;
uniform float u_t;
void main() {
  float a = texture(u_a, v_uv).r;
  float b = texture(u_b, v_uv).r;
  outColor = vec4(mix(a, b, u_t), 0.0, 0.0, 1.0);
}
`;

// ---- Bilateral-lite filter (mirrors noiseReduction.ts bilateralLite()
//      exactly -- a single direct windowed pass, not separable). ----
export const BILATERAL_R_FRAG_SRC = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform int u_radius;
uniform float u_twoSpatialSigma2;
uniform float u_twoRangeSigma2;
void main() {
  float centerVal = texture(u_src, v_uv).r;
  float sum = 0.0;
  float weightSum = 0.0;
  int r = u_radius;
  for (int dy = -r; dy <= r; dy++) {
    for (int dx = -r; dx <= r; dx++) {
      vec2 uv = clamp(v_uv + vec2(float(dx), float(dy)) * u_texel, vec2(0.0), vec2(1.0));
      float val = texture(u_src, uv).r;
      float spatialDist2 = float(dx * dx + dy * dy);
      float rangeDist2 = (val - centerVal) * (val - centerVal);
      float weight = exp(-spatialDist2 / u_twoSpatialSigma2 - rangeDist2 / u_twoRangeSigma2);
      sum += val * weight;
      weightSum += weight;
    }
  }
  outColor = vec4(weightSum > 0.0 ? clamp(sum / weightSum, 0.0, 1.0) : centerVal, 0.0, 0.0, 1.0);
}
`;

// ---- out = in*in (feeds the "mean of squares" half of variance). ----
export const SQUARE_R_FRAG_SRC = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_src;
void main() {
  float v = texture(u_src, v_uv).r;
  outColor = vec4(v * v, 0.0, 0.0, 1.0);
}
`;

// ---- variance = max(0, meanSq - mean^2). Mirrors contrast.ts localVariance(). ----
export const VARIANCE_COMBINE_FRAG_SRC = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_mean;
uniform sampler2D u_meanSq;
void main() {
  float mean = texture(u_mean, v_uv).r;
  float meanSq = texture(u_meanSq, v_uv).r;
  outColor = vec4(max(0.0, meanSq - mean * mean), 0.0, 0.0, 1.0);
}
`;

// ---- Local contrast: mirrors contrast.ts applyLocalContrast() exactly. ----
export const LOCAL_CONTRAST_FRAG_SRC = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_src;
uniform sampler2D u_localMean;
uniform float u_amt;
void main() {
  float I = texture(u_src, v_uv).r;
  float localMean = texture(u_localMean, v_uv).r;
  float detail = I - localMean;
  outColor = vec4(clamp(I + detail * u_amt * 1.5, 0.0, 1.0), 0.0, 0.0, 1.0);
}
`;

// ---- Global tone curve: mirrors toneMapping.ts buildToneCurve() exactly,
//      including its exposure/brightness -> shadow lift -> highlight
//      recovery -> gamma -> contrast S-curve order. At neutral settings
//      (contrastAmt=0.5) the S-curve's k == 1, which is an identity, so
//      this can run unconditionally with no branch for "contrast off". ----
export const TONE_CURVE_FRAG_SRC = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_src;
uniform float u_exposureMul;
uniform float u_brightnessAdd;
uniform float u_gamma;
uniform float u_shadowLift;
uniform float u_highlightRecovery;
uniform float u_sCurveK;

float sCurveF(float x, float k) {
  if (x <= 0.5) return 0.5 * pow(x / 0.5, k);
  return 1.0 - 0.5 * pow((1.0 - x) / 0.5, k);
}

void main() {
  float i = texture(u_src, v_uv).r * u_exposureMul + u_brightnessAdd;
  i = clamp(i, 0.0, 1.0);

  if (u_shadowLift > 0.0) {
    i = i + u_shadowLift * 0.35 * (1.0 - i) * exp(-i * 6.0);
  }

  if (u_highlightRecovery > 0.0) {
    float knee = 1.0 - u_highlightRecovery * 0.5;
    if (i > knee) {
      float over = (i - knee) / max(0.0001, 1.0 - knee);
      i = knee + (1.0 - knee) * (1.0 - exp(-over * 2.0));
    }
  }

  i = pow(clamp(i, 0.0, 1.0), 1.0 / u_gamma);
  i = sCurveF(i, u_sCurveK);

  outColor = vec4(clamp(i, 0.0, 1.0), 0.0, 0.0, 1.0);
}
`;

// ---- Scene heuristics (sky/vegetation confidence maps). Mirrors
//      pipeline.ts computeSceneMaps() exactly. Two render-target outputs
//      (MRT) so this needs only one pass, not two. ----
export const SCENE_FRAG_SRC = `#version 300 es
precision highp float;
in vec2 v_uv;
layout(location = 0) out vec4 outSky;
layout(location = 1) out vec4 outVeg;
uniform sampler2D u_intensity;
uniform sampler2D u_variance;
void main() {
  float I = texture(u_intensity, v_uv).r;
  float v = texture(u_variance, v_uv).r;
  float rowFrac = v_uv.y; // 0 = top of original image, see module header

  float topBias = clamp(1.0 - rowFrac * 1.6, 0.0, 1.0);
  float brightness = clamp((I - 0.55) / 0.45, 0.0, 1.0);
  float flatness = clamp(1.0 - v * 40.0, 0.0, 1.0);
  float sky = topBias * brightness * flatness;

  float bottomBias = clamp(rowFrac * 1.2 + 0.2, 0.0, 1.0);
  float midtone = clamp(1.0 - abs(I - 0.42) / 0.35, 0.0, 1.0);
  float texture_ = clamp(v * 30.0, 0.0, 1.0);
  float veg = bottomBias * midtone * texture_;

  outSky = vec4(sky, 0.0, 0.0, 1.0);
  outVeg = vec4(veg, 0.0, 0.0, 1.0);
}
`;

// ---- IR/NIR -> RGB color-ramp reconstruction. Mirrors colorMapping.ts
//      mapIntensityToRgb() (ramp sampling in OKLab, colorStrength blend,
//      scene tint, temperature, saturation/hue via HSL) exactly, plus a
//      final srgbToLinear so the output texture is immediately usable as
//      input to the existing precision-pipeline shader (which expects
//      linear RGB) with no extra CPU-side conversion. ----
function buildColorMapFragSrc(): string {
  return `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_intensity;
uniform sampler2D u_sky;
uniform sampler2D u_veg;
uniform bool u_sceneEnabled;

uniform float u_stopT[${MAX_STOPS}];
uniform vec3 u_stopColor[${MAX_STOPS}];
uniform int u_stopCount;

uniform float u_colorStrength; // 0..1
uniform float u_satAdjust;     // -1..1
uniform float u_hueBiasFrac;   // 0..1 (wraps)
uniform float u_tempAmt;

// ---- OKLab, operating on LINEAR sRGB (same constants as colorSpace.ts). ----
float cbrtSigned(float x) { return sign(x) * pow(abs(x), 1.0 / 3.0); }

vec3 linearRgbToOklab(vec3 c) {
  float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
  float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
  float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
  float l_ = cbrtSigned(l);
  float m_ = cbrtSigned(m);
  float s_ = cbrtSigned(s);
  return vec3(
    0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
    1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
    0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_
  );
}
vec3 oklabToLinearRgb(vec3 lab) {
  float l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  float m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  float s_ = lab.x - 0.0894841775 * lab.y - 1.291485548 * lab.z;
  float l = l_ * l_ * l_;
  float m = m_ * m_ * m_;
  float s = s_ * s_ * s_;
  return vec3(
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s
  );
}
float srgbToLinearF(float c) { return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4); }
float linearToSrgbF(float c) {
  float v = c <= 0.0031308 ? c * 12.92 : 1.055 * pow(max(c, 0.0), 1.0 / 2.4) - 0.055;
  return clamp(v, 0.0, 1.0);
}
vec3 srgbToLinear3(vec3 c) { return vec3(srgbToLinearF(c.r), srgbToLinearF(c.g), srgbToLinearF(c.b)); }
vec3 srgbToOklab(vec3 c) { return linearRgbToOklab(srgbToLinear3(c)); }
vec3 oklabToSrgb(vec3 lab) {
  vec3 lin = oklabToLinearRgb(lab);
  return vec3(linearToSrgbF(lin.r), linearToSrgbF(lin.g), linearToSrgbF(lin.b));
}
vec3 lerpOklabF(vec3 c1, vec3 c2, float t) {
  vec3 lab1 = srgbToOklab(c1);
  vec3 lab2 = srgbToOklab(c2);
  return oklabToSrgb(mix(lab1, lab2, t));
}

vec3 sampleRampF(float t) {
  float tc = clamp(t, 0.0, 1.0);
  if (u_stopCount <= 0) return vec3(tc);
  if (tc <= u_stopT[0]) return u_stopColor[0];
  if (tc >= u_stopT[u_stopCount - 1]) return u_stopColor[u_stopCount - 1];
  for (int i = 0; i < ${MAX_STOPS - 1}; i++) {
    if (i >= u_stopCount - 1) break;
    float a = u_stopT[i];
    float b = u_stopT[i + 1];
    if (tc >= a && tc <= b) {
      float span = max(1e-6, b - a);
      float localT = (tc - a) / span;
      return lerpOklabF(u_stopColor[i], u_stopColor[i + 1], localT);
    }
  }
  return u_stopColor[u_stopCount - 1];
}

// ---- HSL (mirrors colorSpace.ts rgbToHsl/hslToRgb exactly; always run
//      -- with satAdjust=0 and hueBiasFrac=0 it is an identity, matching
//      the CPU short-circuit's output without needing a branch here). ----
vec3 rgbToHslF(vec3 c) {
  float maxc = max(c.r, max(c.g, c.b));
  float minc = min(c.r, min(c.g, c.b));
  float l = (maxc + minc) * 0.5;
  if (maxc == minc) return vec3(0.0, 0.0, l);
  float d = maxc - minc;
  float s = l > 0.5 ? d / (2.0 - maxc - minc) : d / (maxc + minc);
  float h;
  if (maxc == c.r) {
    h = (c.g - c.b) / d + (c.g < c.b ? 6.0 : 0.0);
  } else if (maxc == c.g) {
    h = (c.b - c.r) / d + 2.0;
  } else {
    h = (c.r - c.g) / d + 4.0;
  }
  h /= 6.0;
  return vec3(h, s, l);
}
float hue2rgbF(float p, float q, float t) {
  float tt = t;
  if (tt < 0.0) tt += 1.0;
  if (tt > 1.0) tt -= 1.0;
  if (tt < 1.0 / 6.0) return p + (q - p) * 6.0 * tt;
  if (tt < 0.5) return q;
  if (tt < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - tt) * 6.0;
  return p;
}
vec3 hslToRgbF(float h, float s, float l) {
  if (s == 0.0) return vec3(l);
  float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
  float p = 2.0 * l - q;
  return vec3(hue2rgbF(p, q, h + 1.0 / 3.0), hue2rgbF(p, q, h), hue2rgbF(p, q, h - 1.0 / 3.0));
}

void main() {
  float I = texture(u_intensity, v_uv).r;
  vec3 rgb = sampleRampF(I);

  if (u_colorStrength < 1.0) {
    rgb = rgb * u_colorStrength + vec3(I) * (1.0 - u_colorStrength);
  }

  if (u_sceneEnabled) {
    float sky = texture(u_sky, v_uv).r;
    float veg = texture(u_veg, v_uv).r;
    if (sky > 0.15) {
      float w = sky * 0.35;
      rgb.b = clamp(rgb.b + w * 0.10, 0.0, 1.0);
      rgb.r = clamp(rgb.r - w * 0.04, 0.0, 1.0);
    }
    if (veg > 0.15) {
      float w = veg * 0.35;
      rgb.g = clamp(rgb.g + w * 0.08, 0.0, 1.0);
      rgb.r = clamp(rgb.r - w * 0.03, 0.0, 1.0);
    }
  }

  rgb.r = clamp(rgb.r + u_tempAmt, 0.0, 1.0);
  rgb.b = clamp(rgb.b - u_tempAmt, 0.0, 1.0);

  vec3 hsl = rgbToHslF(rgb);
  float newH = hsl.x + u_hueBiasFrac;
  newH -= floor(newH);
  float newS = u_satAdjust >= 0.0
    ? clamp(hsl.y + (1.0 - hsl.y) * u_satAdjust, 0.0, 1.0)
    : clamp(hsl.y * (1.0 + u_satAdjust), 0.0, 1.0);
  rgb = hslToRgbF(newH, newS, hsl.z);

  outColor = vec4(srgbToLinear3(clamp(rgb, 0.0, 1.0)), 1.0);
}
`;
}

export const COLOR_MAP_FRAG_SRC = buildColorMapFragSrc();
export { MAX_STOPS };

// ---- Unsharp-mask combine. Mirrors sharpening.ts unsharpMask() exactly,
//      applied to all 3 channels at once (paired with BOX_BLUR_RGB). ----
export const SHARPEN_COMBINE_FRAG_SRC = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_src;
uniform sampler2D u_blurred;
uniform float u_amount;
uniform float u_threshold;
void main() {
  vec3 orig = texture(u_src, v_uv).rgb;
  vec3 blurred = texture(u_blurred, v_uv).rgb;
  vec3 detail = orig - blurred;
  vec3 applied = vec3(
    abs(detail.r) >= u_threshold ? detail.r * u_amount : 0.0,
    abs(detail.g) >= u_threshold ? detail.g * u_amount : 0.0,
    abs(detail.b) >= u_threshold ? detail.b * u_amount : 0.0
  );
  outColor = vec4(clamp(orig + applied, 0.0, 1.0), 1.0);
}
`;
