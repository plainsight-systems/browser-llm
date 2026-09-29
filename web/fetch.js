// Axis L: browser network APIs, and how we download.
//
// Range fetches for preflight, and streaming fetch with progress for the whole
// file. The file's total size comes from the server's Content-Range and is the
// byte source's authoritative size. Failures are named and reported, never
// retried silently.

// Why a fetch failed. `reason` is one of the keys of REASONS.
export class FetchError extends Error {
  constructor(reason, detail) {
    super(`${REASONS[reason]}${detail ? ` (${detail})` : ''}`);
    this.name = 'FetchError';
    this.reason = reason;
  }
}

const REASONS = {
  network: 'the server could not be reached',
  'not-found': 'the file does not exist at this URL',
  'needs-sign-in': 'the file requires signing in; only public files can be downloaded',
  'no-range-support': 'the server does not support partial downloads',
  http: 'the server returned an error',
};

// Parses "bytes 0-15/382156480". Null for anything else.
export function parseContentRange(header) {
  const match = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(header ?? '');
  if (match === null) return null;
  const [start, end, total] = match.slice(1).map(Number);
  return { start, end, total };
}

// Fetches bytes [start, end) of the file at `url`. Resolves with the bytes
// received — fewer than asked for if the file ends first — and the file's
// total size.
export async function fetchRange(url, start, end, { signal } = {}) {
  const response = await request(url, { headers: { Range: `bytes=${start}-${end - 1}` }, signal });
  if (response.status !== 206) throw new FetchError('no-range-support', `status ${response.status}`);
  const range = parseContentRange(response.headers.get('Content-Range'));
  if (range === null || range.start !== start) {
    throw new FetchError('http', 'the server answered with a different range');
  }
  return { bytes: await response.arrayBuffer(), totalSize: range.total };
}

async function request(url, init) {
  let response;
  try {
    response = await fetch(url, init);
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new FetchError('network', new URL(url).host);
  }
  if (response.status === 401 || response.status === 403) throw new FetchError('needs-sign-in');
  if (response.status === 404) throw new FetchError('not-found');
  if (!response.ok) throw new FetchError('http', `status ${response.status}`);
  return response;
}
