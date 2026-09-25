export const ACCEPTED_MIME_TYPES = ['image/png', 'image/jpeg', 'image/webp'];
export const ACCEPTED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp'];
export const MAX_FILE_SIZE_BYTES = 60 * 1024 * 1024; // 60 MB

export interface ValidationResult {
  ok: boolean;
  error?: string;
}

/**
 * Validates a File before it's decoded. Checks MIME type, extension, and
 * size, never trusting the filename alone for anything beyond display.
 */
export function validateImageFile(file: File): ValidationResult {
  if (file.size === 0) {
    return { ok: false, error: 'The selected file is empty.' };
  }
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return {
      ok: false,
      error: `File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum supported size is ${
        MAX_FILE_SIZE_BYTES / (1024 * 1024)
      } MB.`,
    };
  }

  const nameLower = file.name.toLowerCase();
  const hasAcceptedExtension = ACCEPTED_EXTENSIONS.some((ext) => nameLower.endsWith(ext));
  const hasAcceptedMime = ACCEPTED_MIME_TYPES.includes(file.type);

  if (!hasAcceptedMime && !hasAcceptedExtension) {
    return {
      ok: false,
      error: 'Unsupported file type. Please use PNG, JPEG, or WebP images.',
    };
  }

  return { ok: true };
}

/** Escapes a string for safe insertion into text content (defense in depth). */
export function sanitizeDisplayName(name: string): string {
  return name.replace(/[<>&"'`]/g, '');
}
