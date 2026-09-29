// DEVELOPMENT ONLY — NOT A RUNTIME.
//
// Stands in for the C++ runtime so the page's own paths — download, cache,
// load, template, chat — can be exercised before any model runs. It accepts
// every model, loads nothing, and runs nothing. Selected only by the
// ?fake-runtime query parameter, and the page shows a permanent banner while
// it is in use.
//
// Never deployed: tools/assemble_site.sh leaves web/dev/ out of the site, and
// tools/check_site.sh fails the build if it is there.

export async function createRuntime({ onDevice }) {
  onDevice({ ok: true, fake: true });
  return {
    preflight: async () => ({
      status: 'read',
      architecture: 'fake runtime',
      tensorCount: 0,
      accepted: true,
      rejections: [],
    }),
    loadChunk: async ({ offset, bytes, totalSize }) => ({ received: offset + bytes.byteLength, totalSize }),
  };
}
