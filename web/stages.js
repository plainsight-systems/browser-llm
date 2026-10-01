// Axis H. The stages preflight judges a model against, in order, as named in
// its answer. Mirrors bllm::preflight::Stage. Pure functions over a verdict.

export const STAGES = ['read', 'download', 'describe', 'fit', 'upload', 'run'];

export const reaches = (verdict, stage) =>
  STAGES.indexOf(verdict.reached) >= STAGES.indexOf(stage);

// The stage after the one reached (undefined once a model runs), the blockers
// that stop it, and every blocker further on.
export function nextStage(verdict) {
  const next = STAGES[STAGES.indexOf(verdict.reached) + 1];
  return {
    next,
    blocking: verdict.blockers.filter((b) => b.stage === next),
    later: verdict.blockers.filter((b) => b.stage !== next),
  };
}
