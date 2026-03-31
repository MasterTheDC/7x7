(function () {
  "use strict";

  const snapshot = globalThis.FILL_GAME_POLICY_SNAPSHOT;
  if (!snapshot) {
    throw new Error("policy snapshot was not loaded");
  }

  const BOARD_SIZE = snapshot.boardSize;
  const RUNTIME_CONFIG = (snapshot.buildInfo && snapshot.buildInfo.runtimeConfig) || {
    searchDepth: 2,
    candidateK: 6,
    useSymmetry: true,
    runtimeMode: snapshot.runtimeMode || "table_plus_distilled_expectimax_v1",
    exactReachableMaxStates: 50000,
  };
  const PIECE_LABELS = {
    cross: { title: "十字 5マス", subtitle: "中心 + 上下左右" },
    xshape: { title: "バツ 5マス", subtitle: "中心 + 斜め4方向" },
    square3: { title: "3x3 9マス", subtitle: "中心 + 周囲8マス" },
    vert: { title: "縦一列 7マス", subtitle: "指定列を全埋め" },
    hori: { title: "横一列 7マス", subtitle: "指定行を全埋め" },
  };
  const PIECE_ORDER = { cross: 0, xshape: 1, square3: 2, vert: 3, hori: 4 };
  const INVERSE_TRANSFORM = { 0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7 };
  const SWAP_ORIENTATION_TRANSFORMS = new Set([1, 3, 5, 7]);

  const actionCatalog = {};
  const actionByMask = {};
  const policyTable = snapshot.policyTable || {};
  const distilledPolicy = snapshot.distilledPolicy || { models: {} };
  const distilledValue = snapshot.distilledValue || { weights: [] };
  const pieceProbabilities = snapshot.pieceProbabilities || {};

  Object.keys(snapshot.actionCatalog).forEach((pieceType) => {
    actionCatalog[pieceType] = snapshot.actionCatalog[pieceType].map((action) => {
      const enriched = {
        actionId: action.actionId,
        pieceType: action.pieceType,
        anchor: action.anchor,
        label: action.label,
        cells: action.cells,
        mask: BigInt(action.mask),
      };
      actionByMask[enriched.mask.toString()] = {
        pieceType: enriched.pieceType,
        actionId: enriched.actionId,
      };
      return enriched;
    });
  });

  const state = {
    boardState: 0n,
    selectedPieceType: null,
    recommendation: null,
    history: [],
    manualMode: false,
  };

  const elements = {
    board: document.getElementById("board"),
    pieceButtons: document.getElementById("piece-buttons"),
    confirmButton: document.getElementById("confirm-button"),
    undoButton: document.getElementById("undo-button"),
    resetButton: document.getElementById("reset-button"),
    manualMode: document.getElementById("manual-mode"),
    turnCount: document.getElementById("turn-count"),
    filledCount: document.getElementById("filled-count"),
    recommendationEmpty: document.getElementById("recommendation-empty"),
    recommendationCard: document.getElementById("recommendation-card"),
    recommendationLabel: document.getElementById("recommendation-label"),
    recommendationGain: document.getElementById("recommendation-gain"),
    recommendationPiece: document.getElementById("recommendation-piece"),
    historyList: document.getElementById("history-list"),
  };

  function bitAt(row, col) {
    return 1n << BigInt(row * BOARD_SIZE + col);
  }

  function popcount(value) {
    let count = 0;
    let cursor = value;
    while (cursor > 0n) {
      count += Number(cursor & 1n);
      cursor >>= 1n;
    }
    return count;
  }

  function filledCount(boardState) {
    return popcount(boardState);
  }

  function remainingCount(boardState) {
    return 49 - filledCount(boardState);
  }

  function gain(boardState, mask) {
    return popcount(mask & ~boardState);
  }

  function isTerminal(boardState) {
    return boardState === ((1n << 49n) - 1n);
  }

  function transformCoord(row, col, transformId) {
    const last = BOARD_SIZE - 1;
    if (transformId === 0) return [row, col];
    if (transformId === 1) return [col, last - row];
    if (transformId === 2) return [last - row, last - col];
    if (transformId === 3) return [last - col, row];
    if (transformId === 4) return [row, last - col];
    if (transformId === 5) return [col, row];
    if (transformId === 6) return [last - row, col];
    if (transformId === 7) return [last - col, last - row];
    throw new Error("unknown transform " + transformId);
  }

  function transformMask(mask, transformId) {
    let transformed = 0n;
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        if ((mask & bitAt(row, col)) !== 0n) {
          const next = transformCoord(row, col, transformId);
          transformed |= bitAt(next[0], next[1]);
        }
      }
    }
    return transformed;
  }

  function transformPieceType(pieceType, transformId) {
    if (pieceType === "cross" || pieceType === "xshape" || pieceType === "square3") {
      return pieceType;
    }
    if (SWAP_ORIENTATION_TRANSFORMS.has(transformId)) {
      return pieceType === "vert" ? "hori" : "vert";
    }
    return pieceType;
  }

  function transformAction(pieceType, actionId, transformId) {
    const action = actionCatalog[pieceType][actionId];
    const transformedMask = transformMask(action.mask, transformId);
    const transformed = actionByMask[transformedMask.toString()];
    if (!transformed) {
      throw new Error("transformed action not found");
    }
    return transformed;
  }

  function canonicalizeStatePiece(boardState, pieceType) {
    let bestState = null;
    let bestPieceType = null;
    let bestTransformId = 0;
    for (let transformId = 0; transformId < 8; transformId += 1) {
      const transformedState = transformMask(boardState, transformId);
      const transformedPieceType = transformPieceType(pieceType, transformId);
      const isBetter =
        bestState === null ||
        transformedState < bestState ||
        (transformedState === bestState &&
          PIECE_ORDER[transformedPieceType] < PIECE_ORDER[bestPieceType]);
      if (isBetter) {
        bestState = transformedState;
        bestPieceType = transformedPieceType;
        bestTransformId = transformId;
      }
    }
    return {
      canonicalState: bestState,
      canonicalPieceType: bestPieceType,
      transformId: bestTransformId,
    };
  }

  function canonicalizeState(boardState) {
    let best = boardState;
    for (let transformId = 1; transformId < 8; transformId += 1) {
      const transformed = transformMask(boardState, transformId);
      if (transformed < best) {
        best = transformed;
      }
    }
    return best;
  }

  function tableActionId(boardState, pieceType) {
    if (!snapshot.symmetryEnabled) {
      const directKey = pieceType + "|" + boardState.toString();
      if (policyTable[directKey] === undefined) {
        return null;
      }
      return policyTable[directKey];
    }
    const canonical = canonicalizeStatePiece(boardState, pieceType);
    const key = canonical.canonicalPieceType + "|" + canonical.canonicalState.toString();
    const canonicalActionId = policyTable[key];
    if (canonicalActionId === undefined) {
      return null;
    }
    const restored = transformAction(
      canonical.canonicalPieceType,
      canonicalActionId,
      INVERSE_TRANSFORM[canonical.transformId],
    );
    return restored.actionId;
  }

  function dotProduct(weights, features) {
    let total = 0;
    for (let index = 0; index < weights.length && index < features.length; index += 1) {
      total += weights[index] * features[index];
    }
    return total;
  }

  function pieceProbability(pieceType) {
    if (pieceProbabilities[pieceType] !== undefined) {
      return Number(pieceProbabilities[pieceType]);
    }
    return 1 / Math.max(1, snapshot.pieceTypes.length);
  }

  function weightedPieceExpectation(valuesByPiece) {
    let total = 0;
    snapshot.pieceTypes.forEach((pieceType) => {
      total += pieceProbability(pieceType) * valuesByPiece[pieceType];
    });
    return total;
  }

  function rowColCounts(boardState) {
    const rowCounts = [0, 0, 0, 0, 0, 0, 0];
    const colCounts = [0, 0, 0, 0, 0, 0, 0];
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        if ((boardState & bitAt(row, col)) !== 0n) {
          rowCounts[row] += 1;
          colCounts[col] += 1;
        }
      }
    }
    return { rowCounts, colCounts };
  }

  function gainStats(boardState, pieceType) {
    const gains = actionCatalog[pieceType].map((action) => gain(boardState, action.mask));
    return {
      maxGain: Math.max.apply(null, gains) / 7,
      meanGain: gains.reduce((sum, value) => sum + value, 0) / gains.length / 7,
    };
  }

  function emptyRegionStats(boardState) {
    const visited = new Set();
    let regionCount = 0;
    let largest = 0;

    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        const key = row + ":" + col;
        if ((boardState & bitAt(row, col)) !== 0n || visited.has(key)) {
          continue;
        }
        regionCount += 1;
        let regionSize = 0;
        const stack = [[row, col]];
        visited.add(key);
        while (stack.length > 0) {
          const current = stack.pop();
          regionSize += 1;
          const deltas = [
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
          ];
          for (let index = 0; index < deltas.length; index += 1) {
            const nextRow = current[0] + deltas[index][0];
            const nextCol = current[1] + deltas[index][1];
            const nextKey = nextRow + ":" + nextCol;
            if (nextRow < 0 || nextRow >= BOARD_SIZE || nextCol < 0 || nextCol >= BOARD_SIZE) {
              continue;
            }
            if ((boardState & bitAt(nextRow, nextCol)) !== 0n || visited.has(nextKey)) {
              continue;
            }
            visited.add(nextKey);
            stack.push([nextRow, nextCol]);
          }
        }
        if (regionSize > largest) {
          largest = regionSize;
        }
      }
    }

    return {
      regionCount: regionCount / 10,
      largestRegion: largest / 49,
    };
  }

  function edgeCornerCenterStats(boardState) {
    let edgeFilled = 0;
    let cornerFilled = 0;
    let mainDiag = 0;
    let antiDiag = 0;
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        if ((boardState & bitAt(row, col)) === 0n) {
          continue;
        }
        if (row === 0 || row === BOARD_SIZE - 1 || col === 0 || col === BOARD_SIZE - 1) {
          edgeFilled += 1;
        }
        if (row === col) {
          mainDiag += 1;
        }
        if (row + col === BOARD_SIZE - 1) {
          antiDiag += 1;
        }
        if ((row === 0 || row === BOARD_SIZE - 1) && (col === 0 || col === BOARD_SIZE - 1)) {
          cornerFilled += 1;
        }
      }
    }
    return {
      edge: edgeFilled / 24,
      corner: cornerFilled / 4,
      center: (boardState & bitAt(BOARD_SIZE / 2 | 0, BOARD_SIZE / 2 | 0)) !== 0n ? 1 : 0,
      mainDiag: mainDiag / 7,
      antiDiag: antiDiag / 7,
    };
  }

  function valueFeatureVector(boardState) {
    const counts = rowColCounts(boardState);
    const cross = gainStats(boardState, "cross");
    const xshape = gainStats(boardState, "xshape");
    const vert = gainStats(boardState, "vert");
    const hori = gainStats(boardState, "hori");
    const regions = emptyRegionStats(boardState);
    const structure = edgeCornerCenterStats(boardState);
    const features = [
      1,
      filledCount(boardState) / 49,
      remainingCount(boardState) / 49,
    ];
    counts.rowCounts.forEach((count) => features.push(count / 7));
    counts.colCounts.forEach((count) => features.push(count / 7));
    features.push(
      cross.maxGain,
      xshape.maxGain,
      vert.maxGain,
      hori.maxGain,
      cross.meanGain,
      xshape.meanGain,
      vert.meanGain,
      hori.meanGain,
      regions.regionCount,
      regions.largestRegion,
      structure.edge,
      structure.corner,
      structure.center,
      structure.mainDiag,
      structure.antiDiag,
    );
    return features;
  }

  function policyFeatureVector(boardState, action) {
    const before = rowColCounts(boardState);
    const nextState = boardState | action.mask;
    const after = rowColCounts(nextState);
    let actionEdgeCoverage = 0;
    let actionCornerCoverage = 0;
    let newEdgeFill = 0;
    let newCornerFill = 0;
    let touchesCenter = 0;
    let touchesMainDiag = 0;
    let touchesAntiDiag = 0;

    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        const bit = bitAt(row, col);
        if ((action.mask & bit) === 0n) {
          continue;
        }
        if (row === 0 || row === BOARD_SIZE - 1 || col === 0 || col === BOARD_SIZE - 1) {
          actionEdgeCoverage += 1;
          if ((boardState & bit) === 0n) {
            newEdgeFill += 1;
          }
        }
        if ((row === 0 || row === BOARD_SIZE - 1) && (col === 0 || col === BOARD_SIZE - 1)) {
          actionCornerCoverage += 1;
          if ((boardState & bit) === 0n) {
            newCornerFill += 1;
          }
        }
        if (row === 3 && col === 3) {
          touchesCenter = 1;
        }
        if (row === col) {
          touchesMainDiag = 1;
        }
        if (row + col === BOARD_SIZE - 1) {
          touchesAntiDiag = 1;
        }
      }
    }

    let rowCompletionDelta = 0;
    let colCompletionDelta = 0;
    for (let index = 0; index < BOARD_SIZE; index += 1) {
      if (before.rowCounts[index] < BOARD_SIZE && after.rowCounts[index] === BOARD_SIZE) {
        rowCompletionDelta += 1;
      }
      if (before.colCounts[index] < BOARD_SIZE && after.colCounts[index] === BOARD_SIZE) {
        colCompletionDelta += 1;
      }
    }

    return [
      1,
      filledCount(boardState) / 49,
      remainingCount(boardState) / 49,
      gain(boardState, action.mask) / 7,
      filledCount(action.mask) / 7,
      actionEdgeCoverage / 7,
      actionCornerCoverage / 4,
      newEdgeFill / 7,
      newCornerFill / 4,
      filledCount(nextState) / 49,
      rowCompletionDelta / 7,
      colCompletionDelta / 7,
      Math.max.apply(null, after.rowCounts) / 7,
      Math.max.apply(null, after.colCounts) / 7,
      touchesCenter,
      touchesMainDiag,
      touchesAntiDiag,
      action.anchor[0] / 6,
      action.anchor[1] / 6,
      action.actionId / Math.max(1, actionCatalog[action.pieceType].length - 1),
    ];
  }

  function heuristicValueProxy(boardState) {
    if (isTerminal(boardState)) {
      return 0;
    }
    const bestGains = {};
    snapshot.pieceTypes.forEach((pieceType) => {
      bestGains[pieceType] = Math.max.apply(
        null,
        actionCatalog[pieceType].map((action) => gain(boardState, action.mask)),
      );
    });
    const meanBestGain = weightedPieceExpectation(bestGains);
    const regions = emptyRegionStats(boardState);
    const structure = edgeCornerCenterStats(boardState);
    const fragmentationPenalty = regions.regionCount * 1.2 + (1 - regions.largestRegion) * 0.9;
    const structureBonus = (structure.edge + 0.5 * structure.corner + 0.35 * structure.center) * 0.6;
    const diagonalBonus = (structure.mainDiag + structure.antiDiag) * 0.2;
    return Math.max(
      0,
      remainingCount(boardState) / Math.max(1, meanBestGain) + fragmentationPenalty - structureBonus - diagonalBonus,
    );
  }

  function valueEstimate(boardState, caches) {
    if (isTerminal(boardState)) {
      return 0;
    }
    const cacheKey = canonicalizeState(boardState).toString();
    if (caches.value.has(cacheKey)) {
      return caches.value.get(cacheKey);
    }
    const weights = distilledValue.weights || [];
    const heuristic = heuristicValueProxy(boardState);
    const modelEstimate = weights.length > 0 ? dotProduct(weights, valueFeatureVector(boardState)) : heuristic;
    const estimate = Math.max(0, heuristic * 0.65 + modelEstimate * 0.35);
    caches.value.set(cacheKey, estimate);
    return estimate;
  }

  function policyScore(boardState, action) {
    const model = distilledPolicy.models && distilledPolicy.models[action.pieceType];
    if (!model || !model.weights || model.weights.length === 0) {
      return gain(boardState, action.mask);
    }
    return dotProduct(model.weights, policyFeatureVector(boardState, action)) + 0.75 * gain(boardState, action.mask);
  }

  function rankedCandidates(boardState, pieceType) {
    const tableId = tableActionId(boardState, pieceType);
    return actionCatalog[pieceType]
      .map((action) => ({
        action: action,
        tableRank: tableId !== null && action.actionId === tableId ? 0 : 1,
        policyScore: policyScore(boardState, action),
        immediateGain: gain(boardState, action.mask),
      }))
      .sort((left, right) => {
        if (left.tableRank !== right.tableRank) {
          return left.tableRank - right.tableRank;
        }
        if (right.policyScore !== left.policyScore) {
          return right.policyScore - left.policyScore;
        }
        if (right.immediateGain !== left.immediateGain) {
          return right.immediateGain - left.immediateGain;
        }
        return left.action.actionId - right.action.actionId;
      })
      .slice(0, Math.min(RUNTIME_CONFIG.candidateK || 6, actionCatalog[pieceType].length));
  }

  function evaluateAction(boardState, action, depth, caches) {
    const nextState = boardState | action.mask;
    if (isTerminal(nextState)) {
      return 1;
    }
    if (depth <= 0) {
      return 1 + valueEstimate(nextState, caches);
    }
    const valuesByPiece = {};
    for (let index = 0; index < snapshot.pieceTypes.length; index += 1) {
      const pieceType = snapshot.pieceTypes[index];
      valuesByPiece[pieceType] = bestActionValue(nextState, pieceType, depth - 1, caches).estimatedSteps;
    }
    return 1 + weightedPieceExpectation(valuesByPiece);
  }

  function bestActionValueExpectimax(boardState, pieceType, depth, caches) {
    const canonical = snapshot.symmetryEnabled ? canonicalizeStatePiece(boardState, pieceType) : null;
    const cacheKey = snapshot.symmetryEnabled
      ? canonical.canonicalPieceType + "|" + canonical.canonicalState.toString() + "|" + depth
      : pieceType + "|" + boardState.toString() + "|" + depth;
    if (caches.action.has(cacheKey)) {
      return caches.action.get(cacheKey);
    }

    const tableId = tableActionId(boardState, pieceType);
    if (tableId !== null) {
      const tableAction = actionCatalog[pieceType][tableId];
      const tableDecision = {
        action: tableAction,
        source: "table",
        estimatedSteps: evaluateAction(boardState, tableAction, depth, caches),
        policyScore: policyScore(boardState, tableAction),
        newlyFilledCount: gain(boardState, tableAction.mask),
      };
      caches.action.set(cacheKey, tableDecision);
      return tableDecision;
    }

    const candidates = rankedCandidates(boardState, pieceType);
    const topPolicyActionId = candidates[0].action.actionId;
    let bestDecision = null;
    for (let index = 0; index < candidates.length; index += 1) {
      const candidate = candidates[index];
      const estimatedSteps = evaluateAction(boardState, candidate.action, depth, caches);
      const decision = {
        action: candidate.action,
        source: candidate.action.actionId === topPolicyActionId ? "distilled_policy" : "search",
        estimatedSteps: estimatedSteps,
        policyScore: candidate.policyScore,
        newlyFilledCount: candidate.immediateGain,
      };
      const shouldPick =
        bestDecision === null ||
        decision.estimatedSteps < bestDecision.estimatedSteps ||
        (decision.estimatedSteps === bestDecision.estimatedSteps &&
          decision.policyScore > bestDecision.policyScore) ||
        (decision.estimatedSteps === bestDecision.estimatedSteps &&
          decision.policyScore === bestDecision.policyScore &&
          decision.newlyFilledCount > bestDecision.newlyFilledCount) ||
        (decision.estimatedSteps === bestDecision.estimatedSteps &&
          decision.policyScore === bestDecision.policyScore &&
          decision.newlyFilledCount === bestDecision.newlyFilledCount &&
          decision.action.actionId < bestDecision.action.actionId);
      if (shouldPick) {
        bestDecision = decision;
      }
    }

    caches.action.set(cacheKey, bestDecision);
    return bestDecision;
  }

  function exactValueCacheKey(boardState) {
    return snapshot.symmetryEnabled ? canonicalizeState(boardState).toString() : boardState.toString();
  }

  function exactActionCacheKey(boardState, pieceType) {
    return pieceType + "|" + boardState.toString();
  }

  function ensureExactBudget(boardState, caches) {
    const cacheKey = exactValueCacheKey(boardState);
    if (caches.exactValue.has(cacheKey)) {
      return;
    }
    if (caches.exactPending.has(cacheKey)) {
      return;
    }
    const limit = Math.max(0, Number(RUNTIME_CONFIG.exactReachableMaxStates || 0));
    if (limit && caches.exactValue.size + caches.exactPending.size >= limit) {
      throw new Error("exact reachable search budget exceeded");
    }
  }

  function exactStateValue(boardState, caches) {
    if (isTerminal(boardState)) {
      return 0;
    }
    const cacheKey = exactValueCacheKey(boardState);
    if (caches.exactValue.has(cacheKey)) {
      return caches.exactValue.get(cacheKey);
    }
    ensureExactBudget(boardState, caches);
    caches.exactPending.add(cacheKey);
    const valuesByPiece = {};
    try {
      for (let index = 0; index < snapshot.pieceTypes.length; index += 1) {
        const pieceType = snapshot.pieceTypes[index];
        valuesByPiece[pieceType] = exactBestActionValue(boardState, pieceType, caches).estimatedSteps;
      }
    } finally {
      caches.exactPending.delete(cacheKey);
    }
    const value = weightedPieceExpectation(valuesByPiece);
    caches.exactValue.set(cacheKey, value);
    return value;
  }

  function exactBestActionValue(boardState, pieceType, caches) {
    const cacheKey = exactActionCacheKey(boardState, pieceType);
    if (caches.exactAction.has(cacheKey)) {
      return caches.exactAction.get(cacheKey);
    }
    let bestDecision = null;
    const actions = actionCatalog[pieceType];
    for (let index = 0; index < actions.length; index += 1) {
      const action = actions[index];
      const nextState = boardState | action.mask;
      if (nextState === boardState) {
        continue;
      }
      const decision = {
        action: action,
        source: "exact_reachable",
        estimatedSteps: isTerminal(nextState) ? 1 : 1 + exactStateValue(nextState, caches),
        policyScore: gain(boardState, action.mask),
        newlyFilledCount: gain(boardState, action.mask),
      };
      const shouldPick =
        bestDecision === null ||
        decision.estimatedSteps < bestDecision.estimatedSteps ||
        (decision.estimatedSteps === bestDecision.estimatedSteps &&
          decision.newlyFilledCount > bestDecision.newlyFilledCount) ||
        (decision.estimatedSteps === bestDecision.estimatedSteps &&
          decision.newlyFilledCount === bestDecision.newlyFilledCount &&
          decision.action.actionId < bestDecision.action.actionId);
      if (shouldPick) {
        bestDecision = decision;
      }
    }
    if (!bestDecision) {
      throw new Error("no exact candidates available");
    }
    caches.exactAction.set(cacheKey, bestDecision);
    return bestDecision;
  }

  function bestActionValue(boardState, pieceType, depth, caches) {
    if (RUNTIME_CONFIG.runtimeMode === "reachable_exact_v1") {
      try {
        return exactBestActionValue(boardState, pieceType, caches);
      } catch (error) {
        if (!error || !String(error.message || error).includes("budget exceeded")) {
          throw error;
        }
        const fallback = bestActionValueExpectimax(boardState, pieceType, depth, caches);
        return {
          action: fallback.action,
          source: "reachable_exact_fallback",
          estimatedSteps: fallback.estimatedSteps,
          policyScore: fallback.policyScore,
          newlyFilledCount: fallback.newlyFilledCount,
        };
      }
    }
    return bestActionValueExpectimax(boardState, pieceType, depth, caches);
  }

  function getRecommendation(boardState, pieceType) {
    const caches = {
      action: new Map(),
      value: new Map(),
      exactAction: new Map(),
      exactValue: new Map(),
      exactPending: new Set(),
    };
    const decision = bestActionValue(boardState, pieceType, RUNTIME_CONFIG.searchDepth || 2, caches);
    return {
      action: decision.action,
      source: decision.source,
      newlyFilledCount: decision.newlyFilledCount,
      estimatedSteps: decision.estimatedSteps,
    };
  }

  function describeAction(action) {
    return action.label;
  }

  function pieceTitle(pieceType) {
    return PIECE_LABELS[pieceType].title;
  }

  function recalculateRecommendation() {
    if (!state.selectedPieceType) {
      state.recommendation = null;
      return;
    }
    const recommendation = getRecommendation(state.boardState, state.selectedPieceType);
    state.recommendation = {
      actionId: recommendation.action.actionId,
      pieceType: recommendation.action.pieceType,
      coveredCells: recommendation.action.cells,
      coveredMask: recommendation.action.mask,
      newlyFilledCount: recommendation.newlyFilledCount,
      estimatedSteps: recommendation.estimatedSteps,
      label: describeAction(recommendation.action),
      source: recommendation.source,
    };
  }

  function pushHistoryEntry(entry) {
    state.history.unshift(entry);
    if (state.history.length > 12) {
      state.history.length = 12;
    }
  }

  function onPieceSelected(pieceType) {
    state.selectedPieceType = pieceType;
    recalculateRecommendation();
    render();
  }

  function confirmRecommendation() {
    if (!state.recommendation) {
      return;
    }
    pushHistoryEntry({
      type: "move",
      previousState: state.boardState,
      previousPieceType: state.selectedPieceType,
      label:
        pieceTitle(state.selectedPieceType) +
        " -> " +
        state.recommendation.label +
        " (" +
        state.recommendation.source +
        ")",
    });
    state.boardState |= state.recommendation.coveredMask;
    state.selectedPieceType = null;
    state.recommendation = null;
    render();
  }

  function undoLastAction() {
    const entry = state.history.shift();
    if (!entry) {
      return;
    }
    state.boardState = entry.previousState;
    state.selectedPieceType = null;
    state.recommendation = null;
    render();
  }

  function resetBoard() {
    state.boardState = 0n;
    state.selectedPieceType = null;
    state.recommendation = null;
    state.history = [];
    elements.manualMode.checked = false;
    state.manualMode = false;
    render();
  }

  function toggleManualCell(row, col) {
    const bit = bitAt(row, col);
    pushHistoryEntry({
      type: "manual",
      previousState: state.boardState,
      previousPieceType: state.selectedPieceType,
      label: "手動補正 -> (" + row + "," + col + ")",
    });
    if ((state.boardState & bit) !== 0n) {
      state.boardState &= ~bit;
    } else {
      state.boardState |= bit;
    }
    recalculateRecommendation();
    render();
  }

  function renderBoard() {
    elements.board.innerHTML = "";
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        const cell = document.createElement("button");
        cell.type = "button";
        cell.className = "cell";
        const bit = bitAt(row, col);
        const isFilled = (state.boardState & bit) !== 0n;
        const isRecommended =
          state.recommendation &&
          (state.recommendation.coveredMask & bit) !== 0n;
        const isFresh =
          state.recommendation &&
          isRecommended &&
          (state.boardState & bit) === 0n;
        if (isFilled) {
          cell.classList.add("is-filled");
        }
        if (isRecommended) {
          cell.classList.add("is-recommended");
        }
        if (isFresh) {
          cell.classList.add("is-fresh");
        }
        if (state.manualMode) {
          cell.classList.add("is-manual");
          cell.addEventListener("click", function () {
            toggleManualCell(row, col);
          });
        } else {
          cell.disabled = true;
        }
        cell.setAttribute("aria-label", "cell " + row + "," + col);
        elements.board.appendChild(cell);
      }
    }
  }

  function renderPieceButtons() {
    elements.pieceButtons.innerHTML = "";
    snapshot.pieceTypes.forEach((pieceType) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "piece-button";
      if (state.selectedPieceType === pieceType) {
        button.classList.add("is-active");
      }
      button.innerHTML =
        '<span class="piece-button-label">' +
        PIECE_LABELS[pieceType].title +
        "</span>" +
        '<span class="piece-button-sub">' +
        PIECE_LABELS[pieceType].subtitle +
        "</span>";
      button.addEventListener("click", function () {
        onPieceSelected(pieceType);
      });
      elements.pieceButtons.appendChild(button);
    });
  }

  function renderRecommendation() {
    if (!state.recommendation) {
      elements.recommendationEmpty.classList.remove("is-hidden");
      elements.recommendationCard.classList.add("is-hidden");
      return;
    }
    elements.recommendationEmpty.classList.add("is-hidden");
    elements.recommendationCard.classList.remove("is-hidden");
    elements.recommendationLabel.textContent = state.recommendation.label;
    elements.recommendationGain.textContent = String(state.recommendation.newlyFilledCount);
    elements.recommendationPiece.textContent = pieceTitle(state.recommendation.pieceType);
  }

  function renderHistory() {
    elements.historyList.innerHTML = "";
    if (state.history.length === 0) {
      const item = document.createElement("li");
      item.textContent = "まだ履歴はありません。";
      elements.historyList.appendChild(item);
      return;
    }
    state.history.forEach((entry) => {
      const item = document.createElement("li");
      item.textContent = entry.label;
      elements.historyList.appendChild(item);
    });
  }

  function renderStatus() {
    elements.turnCount.textContent = String(state.history.filter((entry) => entry.type === "move").length);
    elements.filledCount.textContent = filledCount(state.boardState) + " / 49";
    elements.confirmButton.disabled = !state.recommendation;
  }

  function render() {
    renderBoard();
    renderPieceButtons();
    renderRecommendation();
    renderHistory();
    renderStatus();
  }

  elements.confirmButton.addEventListener("click", confirmRecommendation);
  elements.undoButton.addEventListener("click", undoLastAction);
  elements.resetButton.addEventListener("click", resetBoard);
  elements.manualMode.addEventListener("change", function (event) {
    state.manualMode = event.target.checked;
    render();
  });

  render();
})();
