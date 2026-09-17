import assert from "node:assert/strict";
import test from "node:test";

import {
  fuseShotEvidence,
  type SourcedShotDecision,
} from "../src/tracking/shotEvidenceFusion";
import type { ShotKind, VideoShotDecision } from "../types/tracking";

function evidence(
  source: SourcedShotDecision["source"],
  atSeconds: number,
  kind: ShotKind,
  confidence: number,
): SourcedShotDecision {
  const decision: VideoShotDecision = {
    id: `${source}-${atSeconds}`,
    atSeconds,
    suggestedKind: kind,
    finalKind: confidence >= 0.86 ? kind : null,
    confidence,
    reason: source === "semantic" ? "semantic-basket" : "rim-crossing",
  };
  return { source, decision };
}

test("raises confidence when trajectory and semantic evidence agree", () => {
  const decisions = fuseShotEvidence([
    evidence("trajectory", 1, "make", 0.84),
    evidence("semantic", 1.12, "make", 0.88),
  ]);
  assert.equal(decisions.length, 1);
  assert.equal(decisions[0]?.finalKind, "make");
  assert.equal(decisions[0]?.reason, "evidence-fusion");
  assert.ok((decisions[0]?.confidence ?? 0) >= 0.91);
});

test("routes contradictory make and miss evidence to review", () => {
  const decisions = fuseShotEvidence([
    evidence("trajectory", 2, "miss", 0.92),
    evidence("semantic", 2.18, "make", 0.94),
  ]);
  assert.equal(decisions.length, 1);
  assert.equal(decisions[0]?.finalKind, null);
  assert.equal(decisions[0]?.reason, "evidence-conflict");
});

test("never downgrades a verified detector make because the recovery track follows a rebound", () => {
  const decisions = fuseShotEvidence([
    evidence("detector", 2, "make", 0.93),
    evidence("trajectory", 3.25, "miss", 0.96),
  ]);
  assert.equal(decisions.length, 1);
  assert.equal(decisions[0]?.finalKind, "make");
  assert.ok((decisions[0]?.confidence ?? 0) >= 0.93);
});

test("keeps rapid detector crossings as two independent attempts", () => {
  const decisions = fuseShotEvidence([
    evidence("detector", 2, "make", 0.93),
    evidence("detector", 2.9, "make", 0.94),
  ]);
  assert.deepEqual(decisions.map((decision) => decision.finalKind), ["make", "make"]);
});

test("attaches multi-source evidence to the nearest rapid detector attempt", () => {
  const decisions = fuseShotEvidence([
    evidence("detector", 2, "make", 0.93),
    evidence("semantic", 2.14, "make", 0.91),
    evidence("detector", 2.9, "make", 0.94),
    evidence("semantic", 3.04, "make", 0.92),
  ]);
  assert.equal(decisions.length, 2);
  assert.ok(decisions.every((decision) => decision.finalKind === "make"));
  assert.ok(decisions.every((decision) => decision.reason === "evidence-fusion"));
});

test("preserves a strong semantic-only make recovery", () => {
  const decisions = fuseShotEvidence([
    evidence("semantic", 3, "make", 0.93),
  ]);
  assert.equal(decisions.length, 1);
  assert.equal(decisions[0]?.finalKind, "make");
  assert.equal(decisions[0]?.reason, "semantic-basket");
});
