import { BotDetectionResult } from './types';

/**
 * Export bot detection results to a text file format for submission
 * Format: one user ID per line (matching dataset.bots.txt format)
 */
export function exportBotDetections(
  results: Map<string, BotDetectionResult>,
  teamName: string
): void {
  // Filter for bot detections only
  const botIds: string[] = [];

  results.forEach((result, userId) => {
    if (result.isBot) {
      botIds.push(userId);
    }
  });

  // Sort for consistency
  botIds.sort();

  // Create file content
  const content = botIds.join('\n');

  // Create filename according to competition specs
  const filename = `${teamName}.detections.en.txt`;

  // Trigger download
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);

  console.log(`✅ Exported ${botIds.length} bot detections to ${filename}`);
}

/**
 * Calculate and log detailed statistics for comparison with ground truth
 */
export function calculateStats(
  results: Map<string, BotDetectionResult>,
  groundTruth: Set<string>
): {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  f1: number;
  score: number;
} {
  let tp = 0; // True Positives: Bot detected as Bot
  let fp = 0; // False Positives: Human detected as Bot
  let fn = 0; // False Negatives: Bot detected as Human
  let tn = 0; // True Negatives: Human detected as Human

  // Get all bot predictions
  const predictedBots = new Set<string>();
  results.forEach((result, userId) => {
    if (result.isBot) {
      predictedBots.add(userId);
    }
  });

  // Calculate TP and FN (iterate over ground truth)
  groundTruth.forEach(botId => {
    if (predictedBots.has(botId)) {
      tp++;
    } else {
      fn++;
    }
  });

  // Calculate FP and TN (iterate over all results)
  results.forEach((result, userId) => {
    if (result.isBot && !groundTruth.has(userId)) {
      fp++;
    } else if (!result.isBot && !groundTruth.has(userId)) {
      tn++;
    }
  });

  // Calculate metrics
  const precision = tp > 0 ? tp / (tp + fp) : 0;
  const recall = tp > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  // Competition score: +4 TP, -1 FN, -2 FP
  const score = (tp * 4) + (fn * -1) + (fp * -2);

  const stats = { tp, fp, fn, tn, precision, recall, f1, score };

  // Log detailed report
  console.log('\n📊 Detection Statistics:');
  console.log('━'.repeat(50));
  console.log(`True Positives (TP):  ${tp} (+4 each = +${tp * 4})`);
  console.log(`False Negatives (FN): ${fn} (-1 each = -${fn * 1})`);
  console.log(`False Positives (FP): ${fp} (-2 each = -${fp * 2})`);
  console.log(`True Negatives (TN):  ${tn}`);
  console.log('━'.repeat(50));
  console.log(`Precision: ${(precision * 100).toFixed(2)}%`);
  console.log(`Recall:    ${(recall * 100).toFixed(2)}%`);
  console.log(`F1 Score:  ${(f1 * 100).toFixed(2)}%`);
  console.log('━'.repeat(50));
  console.log(`🏆 Competition Score: ${score}`);
  console.log('━'.repeat(50));

  return stats;
}

/**
 * Export a detailed analysis report
 */
export function exportDetailedReport(
  results: Map<string, BotDetectionResult>,
  groundTruth: Set<string>,
  datasetName: string
): void {
  const stats = calculateStats(results, groundTruth);

  // Create detailed CSV report
  const csvLines = ['User ID,Predicted,Actual,Confidence,Reasoning,Correct'];

  results.forEach((result, userId) => {
    const actual = groundTruth.has(userId) ? 'Bot' : 'Human';
    const predicted = result.isBot ? 'Bot' : 'Human';
    const correct = actual === predicted ? 'Yes' : 'No';
    const reasoning = result.reasoning.replace(/,/g, ';'); // Escape commas

    csvLines.push(`${userId},${predicted},${actual},${result.confidence.toFixed(3)},${reasoning},${correct}`);
  });

  const csvContent = csvLines.join('\n');
  const filename = `analysis_${datasetName}_${Date.now()}.csv`;

  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);

  console.log(`📄 Exported detailed analysis to ${filename}`);
}
