import { Tweet, User } from './types';

// Statistical helpers
function mean(numbers: number[]): number {
  if (numbers.length === 0) return 0;
  return numbers.reduce((a, b) => a + b, 0) / numbers.length;
}

function standardDeviation(numbers: number[]): number {
  if (numbers.length === 0) return 0;
  const avg = mean(numbers);
  return Math.sqrt(numbers.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / numbers.length);
}

// Temporal Pattern Analysis
export function analyzeTemporalPatterns(tweets: Tweet[]): {
  regularityScore: number;
  averageInterval: number;
  intervalVariance: number;
  nighttimePostingRatio: number;
} {
  if (tweets.length < 2) {
    return { regularityScore: 0, averageInterval: 0, intervalVariance: 0, nighttimePostingRatio: 0 };
  }

  const sortedTweets = [...tweets].sort((a, b) =>
    new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
  );

  const intervals: number[] = [];
  for (let i = 1; i < sortedTweets.length; i++) {
    const timeDiff = (new Date(sortedTweets[i].created_at).getTime() -
                      new Date(sortedTweets[i-1].created_at).getTime()) / 1000;
    intervals.push(timeDiff);
  }

  const avgInterval = mean(intervals);
  const intervalStdDev = standardDeviation(intervals);
  const coefficientOfVariation = avgInterval > 0 ? intervalStdDev / avgInterval : 0;

  // Lower CV = more regular = more bot-like. Bots avg ~1.34 CV, humans ~1.92
  const regularityScore = Math.max(0, 1 - coefficientOfVariation / 2.5);

  let nighttimePosts = 0;
  sortedTweets.forEach(tweet => {
    const hour = new Date(tweet.created_at).getUTCHours();
    if (hour >= 1 && hour <= 5) nighttimePosts++;
  });
  const nighttimePostingRatio = nighttimePosts / sortedTweets.length;

  return {
    regularityScore,
    averageInterval: avgInterval,
    intervalVariance: coefficientOfVariation,
    nighttimePostingRatio
  };
}

// Content Similarity Analysis
export function analyzeContentSimilarity(tweets: Tweet[]): {
  similarityScore: number;
  avgSimilarity: number;
  duplicateRatio: number;
} {
  if (tweets.length < 2) {
    return { similarityScore: 0, avgSimilarity: 0, duplicateRatio: 0 };
  }

  const texts = tweets.map(t => t.text.toLowerCase().trim());
  const uniqueTexts = new Set(texts);
  const duplicateRatio = 1 - (uniqueTexts.size / texts.length);

  const maxComparisons = Math.min(50, (texts.length * (texts.length - 1)) / 2);
  const similarities: number[] = [];
  let comparisons = 0;

  for (let i = 0; i < texts.length && comparisons < maxComparisons; i++) {
    for (let j = i + 1; j < texts.length && comparisons < maxComparisons; j++) {
      similarities.push(jaccardSimilarity(texts[i], texts[j]));
      comparisons++;
    }
  }

  const avgSimilarity = similarities.length > 0 ? mean(similarities) : 0;
  const similarityScore = Math.max(duplicateRatio, avgSimilarity > 0.5 ? avgSimilarity : 0);

  return { similarityScore, avgSimilarity, duplicateRatio };
}

function jaccardSimilarity(text1: string, text2: string): number {
  const words1 = new Set(text1.split(/\s+/).filter(w => w.length > 0));
  const words2 = new Set(text2.split(/\s+/).filter(w => w.length > 0));

  if (words1.size === 0 && words2.size === 0) return 1;
  if (words1.size === 0 || words2.size === 0) return 0;

  const intersection = new Set([...words1].filter(x => words2.has(x)));
  const union = new Set([...words1, ...words2]);

  return intersection.size / union.size;
}

// Profile Metadata Analysis
export function analyzeProfileMetadata(user: User): {
  suspiciousScore: number;
  hasGenericUsername: boolean;
  hasMinimalDescription: boolean;
  hasNoLocation: boolean;
} {
  const genericPatterns = [
    /^[A-Za-z]+\d{4,}$/,
    /^[A-Za-z]{1,3}\d+$/,
    /^\d+[A-Za-z]+\d+$/,
    /^(bot|user|account)\d+/i,
  ];

  const hasGenericUsername = genericPatterns.some(pattern => pattern.test(user.username));
  const hasMinimalDescription = !user.description || user.description.trim().length < 10;
  const hasNoLocation = !user.location || user.location.trim().length === 0;

  let suspiciousScore = 0;
  if (hasGenericUsername) suspiciousScore += 0.5;
  if (hasMinimalDescription) suspiciousScore += 0.3;
  if (hasNoLocation) suspiciousScore += 0.2;

  return {
    suspiciousScore: Math.min(suspiciousScore, 1.0),
    hasGenericUsername,
    hasMinimalDescription,
    hasNoLocation
  };
}

// Linguistic Pattern Analysis
export function analyzeLinguisticPatterns(tweets: Tweet[]): {
  linguisticScore: number;
  avgTweetLength: number;
  lengthVariance: number;
  hashtagDensity: number;
  urlDensity: number;
} {
  if (tweets.length === 0) {
    return { linguisticScore: 0, avgTweetLength: 0, lengthVariance: 0, hashtagDensity: 0, urlDensity: 0 };
  }

  const lengths = tweets.map(t => t.text.length);
  const avgLength = mean(lengths);
  const lengthStdDev = standardDeviation(lengths);
  const lengthCoV = avgLength > 0 ? lengthStdDev / avgLength : 0;

  // Bots have more consistent tweet lengths (avg CV 0.34 vs humans 0.52)
  const lengthConsistencyScore = Math.max(0, 1 - lengthCoV / 0.6);

  let totalHashtags = 0;
  let totalUrls = 0;
  tweets.forEach(tweet => {
    totalHashtags += (tweet.text.match(/#\w+/g) || []).length;
    totalUrls += (tweet.text.match(/https?:\/\/\S+/g) || []).length;
  });

  const hashtagDensity = totalHashtags / tweets.length;
  const urlDensity = totalUrls / tweets.length;

  // Bots use more hashtags (avg 0.9 vs humans 0.2)
  const hashtagScore = Math.min(hashtagDensity / 2, 1);

  // Bots use fewer URLs (avg 0.33 vs humans 0.58)
  const lowUrlScore = Math.max(0, 1 - urlDensity);

  // Combined linguistic score
  const linguisticScore = hashtagScore * 0.4 + lengthConsistencyScore * 0.4 + lowUrlScore * 0.2;

  return {
    linguisticScore: Math.min(linguisticScore, 1.0),
    avgTweetLength: avgLength,
    lengthVariance: lengthCoV,
    hashtagDensity,
    urlDensity
  };
}

// Enhanced Z-Score Analysis
export function analyzeZScore(zScore: number | undefined): {
  zScoreContribution: number;
  category: 'extreme' | 'very_high' | 'high' | 'moderate' | 'normal';
} {
  if (zScore === undefined) {
    return { zScoreContribution: 0, category: 'normal' };
  }

  // Continuous scoring - bots avg z=0.49, humans avg z=-0.13
  // Use z/3 as a smooth contribution (clamped to 0-1)
  const contribution = Math.max(0, Math.min(zScore / 3, 1));

  let category: 'extreme' | 'very_high' | 'high' | 'moderate' | 'normal';
  if (zScore > 3.0) category = 'extreme';
  else if (zScore > 2.0) category = 'very_high';
  else if (zScore > 1.0) category = 'high';
  else if (zScore > 0.5) category = 'moderate';
  else category = 'normal';

  return { zScoreContribution: contribution, category };
}

// Mention density analysis
function analyzeMentionDensity(tweets: Tweet[]): number {
  if (tweets.length === 0) return 0;
  let mentions = 0;
  tweets.forEach(t => { mentions += (t.text.match(/@\w+/g) || []).length; });
  return mentions / tweets.length;
}

// Hour entropy: bots post across more distinct hours (avg 3.57 vs humans 2.75)
function analyzeHourEntropy(tweets: Tweet[]): number {
  if (tweets.length < 2) return 0;
  const hourBuckets = new Array(24).fill(0);
  tweets.forEach(t => { hourBuckets[new Date(t.created_at).getUTCHours()]++; });
  const probs = hourBuckets.map(c => c / tweets.length).filter(p => p > 0);
  return -probs.reduce((s, p) => s + p * Math.log2(p), 0);
}

// Compute raw heuristic score (normalized 0-1) from features
export function computeHeuristicScore(user: User, tweets: Tweet[]): {
  raw: number;
  scores: {
    temporal: number;
    content: number;
    profile: number;
    linguistic: number;
    zScore: number;
    pythonModel: number;
  };
  reasons: string[];
} {
  const temporal = analyzeTemporalPatterns(tweets);
  const content = analyzeContentSimilarity(tweets);
  const profile = analyzeProfileMetadata(user);
  const linguistic = analyzeLinguisticPatterns(tweets);
  const zScoreAnalysis = analyzeZScore(user.z_score);
  const mentionDensity = analyzeMentionDensity(tweets);
  const hourEntropy = analyzeHourEntropy(tweets);

  const weights = {
    hashtag: 0.1829,
    length: 0.1412,
    temporal: 0.2931,
    tweetCount: 0.0679,
    url: 0.0453,
    mention: 0.0111,
    hourEntropy: 0.2686,
  };

  const temporalScore = temporal.regularityScore;
  const contentScore = content.similarityScore;
  const profileScore = profile.suspiciousScore;
  const zScoreScore = zScoreAnalysis.zScoreContribution;
  const tweetCountScore = Math.min(user.tweet_count / 60, 1);
  const hashtagScore = Math.min(linguistic.hashtagDensity / 2, 1);
  const lengthScore = Math.max(0, 1 - linguistic.lengthVariance / 0.6);
  const lowUrlScore = Math.max(0, 1 - linguistic.urlDensity);
  const mentionPenalty = Math.min(mentionDensity / 0.5, 1);
  const hourEntropyScore = Math.min(hourEntropy / 4.5, 1);

  const raw =
    hashtagScore * weights.hashtag +
    lengthScore * weights.length +
    temporalScore * weights.temporal +
    tweetCountScore * weights.tweetCount +
    lowUrlScore * weights.url -
    mentionPenalty * weights.mention +
    hourEntropyScore * weights.hourEntropy;

  const reasons: string[] = [];
  if (hourEntropyScore > 0.7) reasons.push('Posts spread across many hours');
  if (hashtagScore > 0.3) reasons.push(`High hashtag usage (${linguistic.hashtagDensity.toFixed(1)}/tweet)`);
  if (temporalScore > 0.5) reasons.push('Regular posting pattern');
  if (lengthScore > 0.5) reasons.push('Consistent tweet lengths');
  if (tweetCountScore > 0.4) reasons.push(`High tweet count (${user.tweet_count})`);
  if (lowUrlScore > 0.7) reasons.push('Low URL usage');
  if (contentScore > 0.3) reasons.push('Repetitive content');
  if (profileScore > 0.3) reasons.push('Suspicious profile');

  return {
    raw,
    scores: {
      temporal: temporalScore,
      content: contentScore,
      profile: profileScore,
      linguistic: linguistic.linguisticScore,
      zScore: zScoreScore,
      pythonModel: 0,
    },
    reasons,
  };
}

// Ensemble weights (defaults — overridden by hill-climbing results)
export interface EnsembleWeights {
  weightPython: number;
  weightHeuristic: number;
  threshold: number;
}

export const DEFAULT_ENSEMBLE_WEIGHTS: EnsembleWeights = {
  weightPython: 0.5,
  weightHeuristic: 0.5,
  threshold: 0.45,
};

// Heuristic-only defaults (no python scores)
export const HEURISTIC_ONLY_WEIGHTS: EnsembleWeights = {
  weightPython: 0,
  weightHeuristic: 1,
  threshold: 0.4586,
};

// Combined Bot Detection with Hybrid Ensemble support
export function detectBot(
  user: User,
  tweets: Tweet[],
  pythonScore?: number,
  ensembleWeights?: EnsembleWeights,
): {
  isBot: boolean;
  confidence: number;
  scores: {
    temporal: number;
    content: number;
    profile: number;
    linguistic: number;
    zScore: number;
    pythonModel: number;
    final: number;
  };
  reasoning: string;
} {
  const heuristic = computeHeuristicScore(user, tweets);
  const normalizedHeuristic = Math.max(0, Math.min(heuristic.raw, 1));
  const pyScore = pythonScore ?? 0;
  const hasPython = pythonScore !== undefined;

  // Pick weights: use provided ensemble weights, or fall back to defaults
  const w = ensembleWeights
    ?? (hasPython ? DEFAULT_ENSEMBLE_WEIGHTS : HEURISTIC_ONLY_WEIGHTS);

  const hybridScore = (w.weightHeuristic * normalizedHeuristic) + (w.weightPython * pyScore);
  const isBot = hybridScore >= w.threshold;

  const reasons = [...heuristic.reasons];
  if (hasPython && pyScore > 0.5) reasons.push(`Python RF model (${(pyScore * 100).toFixed(0)}%)`);

  const reasoning = isBot
    ? reasons.length > 0 ? reasons.join(', ') : 'Multiple weak indicators'
    : 'Normal activity patterns';

  return {
    isBot,
    confidence: Math.min(Math.max(hybridScore, 0) / 0.9, 0.99),
    scores: {
      ...heuristic.scores,
      pythonModel: pyScore,
      final: hybridScore,
    },
    reasoning,
  };
}

// Hill-climbing optimizer to find optimal ensemble weights
export function optimizeEnsembleWeights(
  users: User[],
  allTweets: Tweet[],
  pythonScores: Map<string, number>,
  groundTruth: Set<string>,
  iterations: number = 500000,
  onProgress?: (iteration: number, best: { weights: EnsembleWeights; score: number }) => void,
): { weights: EnsembleWeights; score: number } {
  // Pre-compute heuristic scores for all users
  const heuristicScores = new Map<string, number>();
  users.forEach(user => {
    const userTweets = allTweets.filter(t => t.author_id === user.id);
    const h = computeHeuristicScore(user, userTweets);
    heuristicScores.set(user.id, Math.max(0, Math.min(h.raw, 1)));
  });

  // Fitness function: competition score
  function evaluate(w: EnsembleWeights): number {
    let tp = 0, fp = 0, fn = 0;
    users.forEach(user => {
      const hScore = heuristicScores.get(user.id) ?? 0;
      const pScore = pythonScores.get(user.id) ?? 0;
      const hybrid = (w.weightHeuristic * hScore) + (w.weightPython * pScore);
      const predicted = hybrid >= w.threshold;
      const actual = groundTruth.has(user.id);
      if (actual && predicted) tp++;
      else if (actual && !predicted) fn++;
      else if (!actual && predicted) fp++;
    });
    return (tp * 4) + (fn * -1) + (fp * -2);
  }

  // Initialize with defaults
  let best: EnsembleWeights = { weightPython: 0.5, weightHeuristic: 0.5, threshold: 0.45 };
  let bestScore = evaluate(best);

  const stepSizes = [0.05, 0.02, 0.01, 0.005];

  for (let i = 0; i < iterations; i++) {
    // Decay step size over iterations
    const phase = Math.min(Math.floor(i / (iterations / stepSizes.length)), stepSizes.length - 1);
    const step = stepSizes[phase];

    const candidate: EnsembleWeights = {
      weightPython: Math.max(0, Math.min(1, best.weightPython + (Math.random() * 2 - 1) * step)),
      weightHeuristic: Math.max(0, Math.min(1, best.weightHeuristic + (Math.random() * 2 - 1) * step)),
      threshold: Math.max(0.1, Math.min(0.9, best.threshold + (Math.random() * 2 - 1) * step)),
    };

    const score = evaluate(candidate);
    if (score > bestScore) {
      best = candidate;
      bestScore = score;
    }

    if (onProgress && i % 10000 === 0) {
      onProgress(i, { weights: best, score: bestScore });
    }
  }

  return { weights: best, score: bestScore };
}
