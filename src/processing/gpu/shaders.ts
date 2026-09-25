/**
 * WebGL2 shader source for the "precision pipeline" GPU fast path.
 * --------------------------------------------------------------
 * Covers the four RGB-domain refinement stages that run after IR->RGB
 * color mapping (see processing/pipeline.ts step 9b): Azusa white
 * balance -> AutoTone -> ColorCorrectionPipeline (chroma shaping half) ->
 * gamut protection. All four are pointwise operations (no spatial
 * neighbor reads), so they are fused into a SINGLE fragment shader pass
 * -- one texture upload, one draw call, one readback -- rather than
 * ping-ponging between framebuffers, per the "avoid unnecessary GPU
 * transfers" requirement.
 *
 * The numeric constants below (chroma-boost ceiling, gamut bisection
 * step count/epsilon, AutoTone knee start) are imported directly from
 * the CPU modules and interpolated into the shader source as literals,
 * rather than re-typed by hand, specifically so the GPU and CPU
 * implementations cannot silently drift apart.
 *
 * NOT yet GPU-accelerated here (remains CPU-only): IR->RGB color-ramp
 * reconstruction (colorMapping.ts), noise reduction, local contrast, the
 * global tone curve, and sharpening -- these are spatial-kernel or
 * ramp-sampling operations that need their own shader passes and are
 * explicitly out of scope for this change (see backend.ts header and the
 * README GPU section for the honest, current state).
 */

import { DEFAULT_MAX_CHROMA_BOOST } from '../colorCorrectionPipeline';
import { DEFAULT_STEPS, DEFAULT_EPS } from '../gamutProtection';
import { AUTOTONE_KNEE_START } from '../autoTone';

export const PRECISION_PIPELINE_VERTEX_SRC = `#version 300 es
// Fullscreen triangle -- no vertex buffer needed, positions derived from gl_VertexID.
out vec2 v_uv;
void main() {
  vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  v_uv = pos;
  gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
`;

/** Must stay numerically identical to gamutMapLinearRgb's inGamut() in gamutProtection.ts. */
function buildFragmentSrc(): string {
  const steps = DEFAULT_STEPS;
  const eps = DEFAULT_EPS;
  const maxChromaBoost = DEFAULT_MAX_CHROMA_BOOST;
  const kneeStart = AUTOTONE_KNEE_START;

  return `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_src; // linear-light RGB, alpha unused (always 1)

// ---- Azusa white balance (uniform gains, computed on CPU from the same
//      robust trimmed-mean/chroma-confidence estimate as the CPU path --
//      see whiteBalance.ts computeWhiteBalanceGains). ----
uniform vec3 u_wbGain;

// ---- AutoTone (uniform black point / range, computed on CPU from the
//      same percentile histogram as the CPU path -- see autoTone.ts
//      computeAutoToneParams). ----
uniform float u_atBlackPoint;
uniform float u_atRange; // max(whitePoint - blackPoint, 1e-4)
uniform bool u_atEnabled;

// ---- ColorCorrectionPipeline channel balance (uniform gains, computed
//      on CPU -- see colorCorrectionPipeline.ts computeChannelBalanceGains). ----
uniform vec3 u_ccGain;
uniform float u_ccStrengthFrac; // 0..1, drives the per-pixel chroma-shaping pass below
uniform bool u_ccEnabled;

uniform bool u_gamutEnabled;

const float KNEE_START = ${kneeStart};
const float MAX_CHROMA_BOOST = ${maxChromaBoost};
const int GAMUT_STEPS = ${steps};
const float GAMUT_EPS = ${eps};

// ---- OKLab (Bjorn Ottosson's formulation), operating on linear sRGB. ----
// Mirrors colorSpace.ts linearRgbToOklab / oklabToLinearRgb exactly.
float cbrtSigned(float x) {
  return sign(x) * pow(abs(x), 1.0 / 3.0);
}

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

// ---- AutoTone soft-knee tone map. Mirrors autoTone.ts toneMap(). ----
float autoToneMap(float v) {
  float x = (v - u_atBlackPoint) / u_atRange;
  if (x > KNEE_START) {
    float over = (x - KNEE_START) / (1.0 - KNEE_START);
    x = KNEE_START + (1.0 - KNEE_START) * (1.0 - exp(-over));
  }
  return clamp(x, 0.0, 1.0);
}

// ---- ColorCorrectionPipeline chroma shaping. Mirrors the per-pixel loop
//      in colorCorrectionPipeline.ts applyColorCorrectionPipeline(). ----
vec3 chromaShape(vec3 rgb) {
  vec3 lab = linearRgbToOklab(rgb);
  float C = length(lab.yz);
  if (C < 1e-5) return rgb;

  float midtoneWeight = 1.0 - pow(min(1.0, abs(lab.x - 0.5) / 0.5), 1.5);
  float saturationRolloff = 1.0 / (1.0 + C * 3.0);
  float boost = 1.0 + (MAX_CHROMA_BOOST - 1.0) * u_ccStrengthFrac * midtoneWeight * saturationRolloff;
  float newC = C * boost;
  float scale = newC / C;

  return oklabToLinearRgb(vec3(lab.x, lab.y * scale, lab.z * scale));
}

// ---- Gamut protection: OKLCH chroma reduction via bisection. Mirrors
//      gamutMapLinearRgb() in gamutProtection.ts exactly, including its
//      step count and epsilon. ----
bool inGamut(vec3 rgb) {
  return all(greaterThanEqual(rgb, vec3(-GAMUT_EPS))) && all(lessThanEqual(rgb, vec3(1.0 + GAMUT_EPS)));
}

vec3 gamutMap(vec3 rgb) {
  if (inGamut(rgb)) return clamp(rgb, 0.0, 1.0);

  vec3 lab = linearRgbToOklab(rgb);
  float C = length(lab.yz);
  if (C < 1e-6) return clamp(rgb, 0.0, 1.0);

  vec2 hue = lab.yz / C;
  float lo = 0.0;
  float hi = C;
  vec3 best = clamp(rgb, 0.0, 1.0);

  for (int i = 0; i < GAMUT_STEPS; i++) {
    float mid = (lo + hi) * 0.5;
    vec3 candidate = oklabToLinearRgb(vec3(lab.x, hue * mid));
    if (inGamut(candidate)) {
      best = candidate;
      lo = mid;
    } else {
      hi = mid;
    }
  }

  return clamp(best, 0.0, 1.0);
}

void main() {
  vec3 rgb = texture(u_src, v_uv).rgb;

  // 1. Azusa white balance.
  rgb *= u_wbGain;

  // 2. AutoTone.
  if (u_atEnabled) {
    rgb = vec3(autoToneMap(rgb.r), autoToneMap(rgb.g), autoToneMap(rgb.b));
  }

  // 3. ColorCorrectionPipeline (channel balance + chroma shaping).
  if (u_ccEnabled) {
    rgb *= u_ccGain;
    rgb = chromaShape(rgb);
  }

  // 4. Gamut protection.
  if (u_gamutEnabled) {
    rgb = gamutMap(rgb);
  } else {
    rgb = clamp(rgb, 0.0, 1.0);
  }

  outColor = vec4(rgb, 1.0);
}
`;
}

export const PRECISION_PIPELINE_FRAGMENT_SRC = buildFragmentSrc();
