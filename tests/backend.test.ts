import { describe, it, expect } from 'vitest';
import { detectCapabilities, resolveBackend } from '../src/processing/backend';

describe('detectCapabilities', () => {
  it('never throws and returns a well-formed capabilities object in a non-browser test environment', () => {
    const caps = detectCapabilities();
    expect(typeof caps.webgpu).toBe('boolean');
    expect(typeof caps.webgl2).toBe('boolean');
    expect(typeof caps.offscreenCanvas).toBe('boolean');
  });
});

describe('resolveBackend', () => {
  it('resolves "cpu" to cpu unconditionally', () => {
    const result = resolveBackend('cpu', { webgpu: true, webgl2: true, offscreenCanvas: true }, 1_000_000);
    expect(result.resolved).toBe('cpu');
    expect(result.fellBack).toBe(false);
  });

  it('falls back from webgpu to cpu when webgpu is unavailable', () => {
    const result = resolveBackend('webgpu', { webgpu: false, webgl2: true, offscreenCanvas: true }, 1_000_000);
    expect(result.resolved).toBe('cpu');
    expect(result.fellBack).toBe(true);
    expect(result.reason).toBeTruthy();
  });

  it('falls back from webgl2 to cpu when webgl2 is unavailable', () => {
    const result = resolveBackend('webgl2', { webgpu: false, webgl2: false, offscreenCanvas: false }, 1_000_000);
    expect(result.resolved).toBe('cpu');
    expect(result.fellBack).toBe(true);
  });

  it('uses webgpu when requested and available', () => {
    const result = resolveBackend('webgpu', { webgpu: true, webgl2: true, offscreenCanvas: true }, 1_000_000);
    expect(result.resolved).toBe('webgpu');
    expect(result.fellBack).toBe(false);
  });

  it('"auto" prefers cpu for small images even when GPU is available', () => {
    const result = resolveBackend('auto', { webgpu: true, webgl2: true, offscreenCanvas: true }, 100);
    expect(result.resolved).toBe('cpu');
  });

  it('"auto" prefers webgpu for large images when available', () => {
    const result = resolveBackend('auto', { webgpu: true, webgl2: true, offscreenCanvas: true }, 4_000_000);
    expect(result.resolved).toBe('webgpu');
  });

  it('"auto" with no GPU available always resolves to cpu', () => {
    const result = resolveBackend('auto', { webgpu: false, webgl2: false, offscreenCanvas: false }, 4_000_000);
    expect(result.resolved).toBe('cpu');
    expect(result.fellBack).toBe(false);
  });
});
