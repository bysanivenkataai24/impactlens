import { MAX_PHOTOS, type Photo, type ServiceLimits } from './types';

export type MediaType = 'image/jpeg' | 'image/png' | 'image/webp';

/** Identify JPEG / PNG / WebP from the file's bytes; the name and declared type are ignored. */
export function sniffMediaType(bytes: Uint8Array): MediaType | null {
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'image/jpeg';
  const png = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (bytes.length >= 8 && png.every((b, i) => bytes[i] === b)) return 'image/png';
  const ascii = (start: number, end: number) => String.fromCharCode(...bytes.slice(start, end));
  if (bytes.length >= 12 && ascii(0, 4) === 'RIFF' && ascii(8, 12) === 'WEBP') return 'image/webp';
  return null;
}

export async function sha256Hex(data: ArrayBuffer | Uint8Array): Promise<string> {
  const view = data instanceof Uint8Array ? data : new Uint8Array(data);
  const digest = await crypto.subtle.digest('SHA-256', view as Uint8Array<ArrayBuffer>);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Decode with an <img> element: its natural size honours EXIF orientation
 * (CSS `image-orientation: from-image`), i.e. exactly what the viewer shows and
 * what the service reports after it applies the same orientation.
 */
export function decodeImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = 'async';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('decode failed'));
    image.src = url;
  });
}

export interface Rejection {
  fileName: string;
  reason: string;
}

export interface Accepted {
  photo: Photo;
  file: File;
}

let counter = 0;
export function newId(prefix: string): string {
  counter += 1;
  const random = crypto.getRandomValues(new Uint32Array(2));
  return `${prefix}-${Date.now().toString(36)}-${random[0].toString(36)}${random[1].toString(36)}-${counter}`;
}

/** Validate and decode user-selected files. Nothing is uploaded here. */
export async function acceptFiles(
  files: File[],
  { existing, limits }: { existing: Photo[]; limits: ServiceLimits },
): Promise<{ accepted: Accepted[]; rejected: Rejection[] }> {
  const accepted: Accepted[] = [];
  const rejected: Rejection[] = [];
  const hashes = new Set(existing.map((p) => p.sha256));
  for (const file of files) {
    const name = file.name || 'unnamed file';
    if (existing.length + accepted.length >= MAX_PHOTOS) {
      rejected.push({ fileName: name, reason: `A case holds at most ${MAX_PHOTOS} photos.` });
      continue;
    }
    if (file.size === 0) {
      rejected.push({ fileName: name, reason: 'The file is empty.' });
      continue;
    }
    if (file.size > limits.max_upload_bytes) {
      rejected.push({ fileName: name, reason: `${formatBytes(file.size)} is over the ${formatBytes(limits.max_upload_bytes)} limit.` });
      continue;
    }
    const bytes = new Uint8Array(await file.arrayBuffer());
    const mediaType = sniffMediaType(bytes);
    if (!mediaType) {
      rejected.push({ fileName: name, reason: 'Not a JPEG, PNG or WebP image (checked from the file content). HEIC photos must be converted first.' });
      continue;
    }
    const sha256 = await sha256Hex(bytes);
    if (hashes.has(sha256)) {
      rejected.push({ fileName: name, reason: 'This exact file is already in the case.' });
      continue;
    }
    const url = URL.createObjectURL(file);
    let width = 0;
    let height = 0;
    try {
      const image = await decodeImage(url);
      width = image.naturalWidth;
      height = image.naturalHeight;
    } catch {
      URL.revokeObjectURL(url);
      rejected.push({ fileName: name, reason: 'The image could not be decoded; it may be truncated or corrupt.' });
      continue;
    }
    if (width * height > limits.max_image_pixels) {
      URL.revokeObjectURL(url);
      rejected.push({ fileName: name, reason: `${width}×${height} px exceeds the ${limits.max_image_pixels.toLocaleString('en-IN')} pixel limit.` });
      continue;
    }
    if (Math.min(width, height) < limits.min_image_side) {
      URL.revokeObjectURL(url);
      rejected.push({ fileName: name, reason: `${width}×${height} px is too small (minimum ${limits.min_image_side} px per side).` });
      continue;
    }
    hashes.add(sha256);
    accepted.push({
      file,
      photo: {
        id: newId('photo'),
        file_name: name,
        media_type: mediaType,
        bytes: file.size,
        sha256,
        width,
        height,
        object_url: url,
        model_run: null,
      },
    });
  }
  return { accepted, rejected };
}
