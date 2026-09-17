import type { ShotKind, VideoShotDecision } from "../../types/tracking";
import { MIN_AUTOMATIC_DECISION_CONFIDENCE } from "./engine";

export type ShotEvidenceSource = "detector" | "trajectory" | "semantic";

export interface SourcedShotDecision {
  source: ShotEvidenceSource;
  decision: VideoShotDecision;
}

// Independent sources normally observe the same rim crossing within a few
// decoded frames. Keep that agreement window narrow so rapid, distinct shots
// cannot collapse into one event. The longer window is used only to attach a
// rebound-shaped trajectory conflict to an earlier detector crossing.
const SOURCE_AGREEMENT_WINDOW_SECONDS = 0.7;
const REBOUND_RECOVERY_WINDOW_SECONDS = 1.7;

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function clusterEvidence(evidence: SourcedShotDecision[]): SourcedShotDecision[][] {
  const ordered = [...evidence].sort(
    (left, right) => left.decision.atSeconds - right.decision.atSeconds,
  );
  // Each detector crossing is an attempt anchor. Never merge two detector
  // events merely because a fast drill puts them close together.
  const detectorClusters = ordered
    .filter((item) => item.source === "detector")
    .map((item) => [item]);
  const secondaryClusters: SourcedShotDecision[][] = [];

  for (const item of ordered.filter((candidate) => candidate.source !== "detector")) {
    const sameKindAnchor = detectorClusters
      .map((cluster) => cluster[0])
      .filter((anchor): anchor is SourcedShotDecision =>
        anchor !== undefined &&
        anchor.decision.suggestedKind === item.decision.suggestedKind &&
        Math.abs(anchor.decision.atSeconds - item.decision.atSeconds) <=
          SOURCE_AGREEMENT_WINDOW_SECONDS
      )
      .sort((left, right) =>
        Math.abs(left.decision.atSeconds - item.decision.atSeconds) -
        Math.abs(right.decision.atSeconds - item.decision.atSeconds)
      )[0];
    if (sameKindAnchor) {
      detectorClusters.find((cluster) => cluster[0] === sameKindAnchor)?.push(item);
      continue;
    }

    // A trajectory tracker may follow a scored ball into its rebound and emit
    // an adjacent miss. Preserve the verified crossing, but apply this broader
    // suppression only to that rebound-prone source and only after the anchor.
    const reboundAnchor = item.source === "trajectory"
      ? detectorClusters
        .map((cluster) => cluster[0])
        .filter((anchor): anchor is SourcedShotDecision => {
          if (!anchor) return false;
          const delay = item.decision.atSeconds - anchor.decision.atSeconds;
          return (
            anchor.decision.finalKind !== null &&
            anchor.decision.suggestedKind !== item.decision.suggestedKind &&
            delay >= 0 &&
            delay <= REBOUND_RECOVERY_WINDOW_SECONDS
          );
        })
        .sort((left, right) => right.decision.atSeconds - left.decision.atSeconds)[0]
      : undefined;
    if (reboundAnchor) {
      detectorClusters.find((cluster) => cluster[0] === reboundAnchor)?.push(item);
      continue;
    }

    const current = secondaryClusters.at(-1);
    const first = current?.[0];
    if (
      current &&
      first &&
      item.decision.atSeconds - first.decision.atSeconds <= SOURCE_AGREEMENT_WINDOW_SECONDS
    ) {
      current.push(item);
    } else {
      secondaryClusters.push([item]);
    }
  }
  return [...detectorClusters, ...secondaryClusters].sort(
    (left, right) =>
      (left[0]?.decision.atSeconds ?? 0) - (right[0]?.decision.atSeconds ?? 0),
  );
}

function strongestForKind(
  cluster: SourcedShotDecision[],
  kind: ShotKind,
): SourcedShotDecision | null {
  return cluster
    .filter((item) => item.decision.suggestedKind === kind)
    .sort((left, right) => right.decision.confidence - left.decision.confidence)[0] ?? null;
}

/**
 * Combines independent detector crossing, cleaned trajectory, and semantic
 * basket-occupancy evidence. Agreement raises confidence; disagreement is
 * never guessed and remains reviewable.
 */
export function fuseShotEvidence(
  evidence: SourcedShotDecision[],
): VideoShotDecision[] {
  return clusterEvidence(evidence).map((cluster, index) => {
    const make = strongestForKind(cluster, "make");
    const miss = strongestForKind(cluster, "miss");
    const sources = new Set(cluster.map((item) => item.source));
    const authoritativeDetector = cluster
      .filter((item) =>
        item.source === "detector" && item.decision.finalKind !== null
      )
      .sort((left, right) => right.decision.confidence - left.decision.confidence)[0] ?? null;

    // Recovery evidence is additive. It must never turn a verified crossing
    // into a different result because a secondary tracker followed the rebound
    // or briefly attached to another ball after the event.
    if (authoritativeDetector) {
      const agreeingSources = new Set(cluster
        .filter((item) =>
          item.decision.suggestedKind === authoritativeDetector.decision.suggestedKind
        )
        .map((item) => item.source));
      const confidence = clamp(
        authoritativeDetector.decision.confidence +
        (agreeingSources.size >= 2 ? 0.03 : 0),
      );
      return {
        ...authoritativeDetector.decision,
        id: `${Math.round(authoritativeDetector.decision.atSeconds * 1_000)}-${index}-fused`,
        finalKind: authoritativeDetector.decision.suggestedKind,
        confidence,
        reason: agreeingSources.size >= 2
          ? "evidence-fusion" as const
          : authoritativeDetector.decision.reason,
      };
    }

    if (make && miss) {
      const strongest = make.decision.confidence >= miss.decision.confidence ? make : miss;
      return {
        ...strongest.decision,
        id: `${Math.round(strongest.decision.atSeconds * 1_000)}-${index}-conflict`,
        finalKind: null,
        confidence: clamp(Math.max(make.decision.confidence, miss.decision.confidence) * 0.88),
        reason: "evidence-conflict" as const,
      };
    }

    const strongest = make ?? miss;
    if (!strongest) {
      throw new Error("Shot evidence cluster contained no decision.");
    }
    const agreeing = cluster.filter(
      (item) => item.decision.suggestedKind === strongest.decision.suggestedKind,
    );
    const agreementBonus = sources.size >= 3 ? 0.08 : sources.size >= 2 ? 0.05 : 0;
    const averageConfidence = agreeing.reduce(
      (sum, item) => sum + item.decision.confidence,
      0,
    ) / Math.max(1, agreeing.length);
    const confidence = clamp(
      Math.max(strongest.decision.confidence, averageConfidence + agreementBonus),
    );
    const finalKind = confidence >= MIN_AUTOMATIC_DECISION_CONFIDENCE
      ? strongest.decision.suggestedKind
      : null;
    return {
      ...strongest.decision,
      id: `${Math.round(strongest.decision.atSeconds * 1_000)}-${index}-fused`,
      finalKind,
      confidence,
      reason: sources.size >= 2 ? "evidence-fusion" as const : strongest.decision.reason,
    };
  });
}
