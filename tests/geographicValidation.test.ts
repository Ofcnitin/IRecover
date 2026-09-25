import { describe, it, expect } from 'vitest';
import { validateGeminiResponse, stripJsonFences } from '../src/utils/geographicValidation';

describe('stripJsonFences', () => {
  it('removes ```json fences', () => {
    expect(stripJsonFences('```json\n{"a":1}\n```')).toBe('{"a":1}');
  });

  it('removes plain ``` fences', () => {
    expect(stripJsonFences('```\n{"a":1}\n```')).toBe('{"a":1}');
  });

  it('leaves unfenced text alone', () => {
    expect(stripJsonFences('{"a":1}')).toBe('{"a":1}');
  });
});

describe('validateGeminiResponse', () => {
  const valid = {
    scene: { description: 'A mixed landscape.', confidence: 0.8 },
    features: [
      { type: 'vegetation', description: 'Dense vegetation', estimatedCoverage: 42, confidence: 0.9 },
    ],
    location: null,
    limitations: ['Exact geographic identity cannot be determined from the image alone.'],
  };

  it('accepts a well-formed response', () => {
    const result = validateGeminiResponse(valid);
    expect(result.ok).toBe(true);
    expect(result.value?.scene.description).toBe('A mixed landscape.');
    expect(result.value?.features).toHaveLength(1);
    expect(result.value?.location).toBeNull();
  });

  it('rejects a non-object payload', () => {
    expect(validateGeminiResponse('not an object').ok).toBe(false);
    expect(validateGeminiResponse(null).ok).toBe(false);
    expect(validateGeminiResponse([1, 2, 3]).ok).toBe(false);
  });

  it('rejects a response missing scene.description', () => {
    const result = validateGeminiResponse({ scene: {}, features: [], location: null, limitations: [] });
    expect(result.ok).toBe(false);
  });

  it('drops malformed feature entries rather than failing entirely', () => {
    const result = validateGeminiResponse({
      ...valid,
      features: [
        { type: 'vegetation', description: 'ok', confidence: 0.5 },
        { description: 123 }, // invalid -- description must be a string
        'not-an-object',
      ],
    });
    expect(result.ok).toBe(true);
    expect(result.value?.features).toHaveLength(1);
  });

  it('coerces an unrecognized feature type to "other"', () => {
    const result = validateGeminiResponse({
      ...valid,
      features: [{ type: 'lava-flow', description: 'ok', confidence: 0.5 }],
    });
    expect(result.value?.features[0].type).toBe('other');
  });

  it('clamps confidence into [0,1]', () => {
    const result = validateGeminiResponse({ ...valid, scene: { description: 'x', confidence: 5 } });
    expect(result.value?.scene.confidence).toBe(1);
    const result2 = validateGeminiResponse({ ...valid, scene: { description: 'x', confidence: -5 } });
    expect(result2.value?.scene.confidence).toBe(0);
  });

  it('falls back to a default limitation when none are provided', () => {
    const result = validateGeminiResponse({ ...valid, limitations: [] });
    expect(result.value?.limitations.length).toBeGreaterThan(0);
  });

  it('rejects a malformed location object but keeps the rest valid', () => {
    const result = validateGeminiResponse({ ...valid, location: 'Paris, France' });
    expect(result.ok).toBe(true);
    expect(result.value?.location).toBeNull();
  });

  it('accepts a well-formed non-null location', () => {
    const result = validateGeminiResponse({
      ...valid,
      location: { country: 'France', region: null, note: 'Low confidence guess.' },
    });
    expect(result.value?.location?.country).toBe('France');
  });
});
