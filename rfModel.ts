import { Tweet } from './types';

// ---- Types for the exported scikit-learn RF model ----

interface RFTreeNode {
  f: number;           // feature index (-2 = leaf)
  t: number;           // threshold
  l: number;           // left child index (-1 = none)
  r: number;           // right child index
  v: [number, number]; // [class0_count, class1_count]
}

interface RFModel {
  n_estimators: number;
  features: string[];
  trees: RFTreeNode[][];
}

// ---- RF Inference ----

function predictTree(nodes: RFTreeNode[], features: number[]): number {
  let idx = 0;
  while (nodes[idx].f !== -2) { // not a leaf
    const node = nodes[idx];
    idx = features[node.f] <= node.t ? node.l : node.r;
  }
  const leaf = nodes[idx];
  const total = leaf.v[0] + leaf.v[1];
  return total > 0 ? leaf.v[1] / total : 0;
}

function predictProba(model: RFModel, features: number[]): number {
  let sum = 0;
  for (const tree of model.trees) {
    sum += predictTree(tree, features);
  }
  return sum / model.trees.length;
}

// ---- Feature Extraction (matches detector_model.py exactly) ----

const TRIGGER_WORDS = [
  'check my bio', 'follow me', 'click', 'free', 'giveaway',
  'win', 'bet', 'stream', 'live', 'crypto', 'nft', 'limited time',
  'official', 'update', 'breaking', 'news', 'alert'
];

function countChar(str: string, char: string): number {
  let count = 0;
  for (let i = 0; i < str.length; i++) {
    if (str[i] === char) count++;
  }
  return count;
}

function countSubstring(str: string, sub: string): number {
  let count = 0;
  let pos = 0;
  while ((pos = str.indexOf(sub, pos)) !== -1) {
    count++;
    pos += sub.length;
  }
  return count;
}

function sampleStdDev(arr: number[]): number {
  const n = arr.length;
  if (n <= 1) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / n;
  const sumSq = arr.reduce((s, v) => s + (v - mean) ** 2, 0);
  return Math.sqrt(sumSq / (n - 1)); // ddof=1 to match pandas .std()
}

function extractFeatures(tweets: Tweet[]): number[] {
  const totalTweets = tweets.length;
  const texts = tweets.map(t => t.text ?? '');
  const fullText = texts.join(' ').toLowerCase();

  // 1. vocab_diversity
  const allWords = fullText.split(/\s+/).filter(w => w.length > 0);
  const totalWords = allWords.length;
  const uniqueWords = new Set(allWords).size;
  const vocabDiversity = totalWords > 0 ? uniqueWords / totalWords : 0;

  // 2. repetition_ratio (matches pandas nunique — counts unique non-null)
  const uniqueTexts = new Set(tweets.map(t => t.text)).size;
  const repetitionRatio = 1.0 - (uniqueTexts / totalTweets);

  // 3. text_len_std (matches pandas .str.len().std() with ddof=1)
  const lengths = texts.map(t => t.length);
  const textLenStd = totalTweets > 1 ? sampleStdDev(lengths) : 0;

  // 4. trigger_word_density
  const triggerCount = TRIGGER_WORDS.reduce((sum, w) => sum + countSubstring(fullText, w), 0);
  const triggerWordDensity = triggerCount / totalTweets;

  // 5. link_density
  const linkDensity = countSubstring(fullText, 'http') / totalTweets;

  // 6. mention_density
  const mentionDensity = countChar(fullText, '@') / totalTweets;

  // 7. hashtag_density
  const hashtagDensity = countChar(fullText, '#') / totalTweets;

  // 8-11. temporal features
  let timeStdDev: number;
  let minTimeGap: number;
  let activeHourCount: number;
  let maxTweetsOneHour: number;

  if (totalTweets > 1) {
    const timestamps = tweets
      .map(t => new Date(t.created_at).getTime())
      .sort((a, b) => a - b);

    const deltas: number[] = [];
    for (let i = 1; i < timestamps.length; i++) {
      deltas.push((timestamps[i] - timestamps[i - 1]) / 1000); // seconds
    }

    timeStdDev = sampleStdDev(deltas); // ddof=1 to match pandas
    minTimeGap = Math.min(...deltas);

    // Use UTC hours to match pandas pd.to_datetime().dt.hour on ISO strings
    const hours = tweets.map(t => new Date(t.created_at).getUTCHours());
    activeHourCount = new Set(hours).size;

    const hourCounts = new Map<number, number>();
    for (const h of hours) {
      hourCounts.set(h, (hourCounts.get(h) ?? 0) + 1);
    }
    maxTweetsOneHour = Math.max(...hourCounts.values());
  } else {
    timeStdDev = -1;
    minTimeGap = -1;
    activeHourCount = 1;
    maxTweetsOneHour = 1;
  }

  // 12. punct_density
  const punctCount = countChar(fullText, '!') + countChar(fullText, '?') + countChar(fullText, '.');
  const punctDensity = punctCount / totalTweets;

  // Return in EXACT order matching model.features
  return [
    vocabDiversity, repetitionRatio, textLenStd,
    triggerWordDensity, linkDensity, mentionDensity, hashtagDensity,
    timeStdDev, minTimeGap, activeHourCount,
    maxTweetsOneHour, punctDensity
  ];
}

// ---- High-level API ----

let cachedModel: RFModel | null = null;

export async function computeRFScores(
  tweets: Tweet[],
  modelUrl = '/rf_model.json',
  onProgress?: (msg: string) => void,
): Promise<Map<string, number>> {
  // Load model (cache it)
  if (!cachedModel) {
    onProgress?.('Loading RF model...');
    const res = await fetch(modelUrl);
    if (!res.ok) throw new Error(`Failed to load RF model: ${res.status}`);
    cachedModel = await res.json() as RFModel;
    onProgress?.(`Loaded RF model (${cachedModel.n_estimators} trees)`);
  }

  // Group tweets by author_id
  const tweetsByUser = new Map<string, Tweet[]>();
  for (const tweet of tweets) {
    const list = tweetsByUser.get(tweet.author_id) ?? [];
    list.push(tweet);
    tweetsByUser.set(tweet.author_id, list);
  }

  // Extract features and predict for each user
  onProgress?.(`Computing RF scores for ${tweetsByUser.size} users...`);
  const scores = new Map<string, number>();
  for (const [userId, userTweets] of tweetsByUser) {
    const features = extractFeatures(userTweets);
    // Replace NaN with 0 (matches Python .fillna(0))
    const clean = features.map(f => isNaN(f) ? 0 : f);
    const prob = predictProba(cachedModel, clean);
    scores.set(userId, prob);
  }

  return scores;
}
