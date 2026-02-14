export interface Tweet {
  id: string;
  text: string;
  created_at: string;
  author_id: string;
  lang: string;
}

export interface User {
  id: string;
  username: string;
  name: string;
  description: string;
  location: string;
  tweet_count: number;
  z_score?: number;
}

export interface Dataset {
  id: string;
  lang: string;
  start_time: string;
  end_time: string;
  total_users: number;
  total_posts: number;
  topics: string[];
  avg_posts_per_user: number;
  avg_z_score: number;
  posts: Tweet[];
  users: User[];
}

export interface BotDetectionResult {
  userId: string;
  isBot: boolean;
  confidence: number; // 0-1
  reasoning: string;
  method: 'heuristic' | 'ensemble';
}

export interface ScoringMetrics {
  truePositives: number;
  falseNegatives: number;
  falsePositives: number;
  score: number;
}
