// Axis L: browser network APIs, and how we download.
//
// Range fetches for preflight: the header prefix first, extended when the
// reader reports how far into the file it needs to read. Streaming fetch with
// progress for the whole file, delivered to the worker in order, in chunks.
//
// The file's total size comes from the server's Content-Range or
// Content-Length and is the byte source's authoritative size. Failures are
// named — HTTP status, network error, size mismatch — and reported, never
// retried silently.
