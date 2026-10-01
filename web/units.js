// Device memory: "376 MiB", "2.0 GiB". Binary units, as the memory budget is
// set in them.
export function formatMemory(bytes) {
  if (bytes >= 2 ** 30) return `${(bytes / 2 ** 30).toFixed(1)} GiB`;
  return `${Math.round(bytes / 2 ** 20)} MiB`;
}

// File sizes: "429 MB", "1.0 GB". Decimal units, matching how Hugging Face
// lists files.
export function formatBytes(bytes) {
  if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(1)} TB`;
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`;
  if (bytes >= 1e3) return `${Math.round(bytes / 1e3)} kB`;
  return `${bytes} B`;
}
