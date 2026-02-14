import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Use 30 AND 32 for training if possible to maximize pattern recognition
TRAIN_JSON_FILES = ['practice_data/dataset.posts&users.30.json', 'practice_data/dataset.posts&users.32.json']
TRAIN_BOT_FILES  = ['practice_data/dataset.bots.30.txt', 'practice_data/dataset.bots.32.txt']

# PREDICTION TARGET (This should be your final evaluation file tomorrow)
# For now, test on 32 again to see the recall improve
TEST_JSON_FILE   = 'practice_data/dataset.posts&users.32.json' 
TEST_BOT_FILE    = 'practice_data/dataset.bots.32.txt'

# AGGRESSIVE THRESHOLD
# 0.50 means "If it's a tie, call it a bot."
# We can afford this because your FP rate is so low.
CONFIDENCE_THRESHOLD = 0.50 

# ==========================================
# 2. DATA LOADING
# ==========================================

def load_data(json_paths, bot_txt_paths=None):
    all_posts = []
    bot_ids = set()

    if isinstance(json_paths, str): json_paths = [json_paths]
    for path in json_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'posts' in data:
                    all_posts.extend(data['posts'])
                elif isinstance(data, list):
                    all_posts.extend(data)
        except FileNotFoundError:
            print(f"File not found: {path}")

    if bot_txt_paths:
        if isinstance(bot_txt_paths, str): bot_txt_paths = [bot_txt_paths]
        for path in bot_txt_paths:
            try:
                with open(path, 'r') as f:
                    bot_ids.update(line.strip() for line in f)
            except FileNotFoundError:
                pass

    return pd.DataFrame(all_posts), bot_ids

# ==========================================
# 3. FEATURE ENGINEERING (AGGRESSIVE)
# ==========================================

def extract_features(df_posts):
    print(f"Extracting features for {df_posts['author_id'].nunique()} users...")
    
    df_posts['created_at'] = pd.to_datetime(df_posts['created_at'])
    df_posts['hour'] = df_posts['created_at'].dt.hour
    df_posts['text_len'] = df_posts['text'].str.len().fillna(0)
    
    user_features = []
    
    for author_id, group in df_posts.groupby('author_id'):
        stats = {'author_id': author_id}
        
        texts = group['text'].fillna("").tolist()
        total_tweets = len(texts)
        full_text = " ".join(texts).lower()
        
        # --- 1. CONTENT ENTROPY (Catch the "Smart" Bots) ---
        # Humans use many unique words. Bots recycle the same vocabulary.
        all_words = full_text.split()
        total_words = len(all_words)
        unique_words = len(set(all_words))
        
        # Avoid division by zero
        if total_words > 0:
            stats['vocab_diversity'] = unique_words / total_words
        else:
            stats['vocab_diversity'] = 0
            
        # Repetition (Classic)
        unique_texts = group['text'].nunique()
        stats['repetition_ratio'] = 1.0 - (unique_texts / total_tweets)
        
        # --- 2. STRUCTURAL FEATURES ---
        # Variance in tweet length (Bots = 0 std dev, Humans = high std dev)
        stats['text_len_std'] = group['text_len'].std() if total_tweets > 1 else 0
        stats['avg_text_len'] = group['text_len'].mean()
        
        # Punctuation usage (Bots often under-use or strictly use punctuation)
        punct_count = full_text.count('!') + full_text.count('?') + full_text.count('.')
        stats['punct_density'] = punct_count / total_tweets
        
        # --- 3. SPAM TRIGGERS (Expanded) ---
        triggers = ['check my bio', 'follow me', 'click', 'free', 'giveaway', 
                    'win', 'bet', 'stream', 'live', 'crypto', 'nft', 'limited time',
                    'official', 'update', 'breaking', 'news', 'alert'] # Added news/sports triggers
        stats['trigger_word_density'] = sum(full_text.count(w) for w in triggers) / total_tweets
        
        stats['link_density'] = full_text.count('http') / total_tweets
        stats['mention_density'] = full_text.count('@') / total_tweets
        stats['hashtag_density'] = full_text.count('#') / total_tweets

        # --- 4. TEMPORAL FEATURES ---
        if total_tweets > 1:
            sorted_times = group['created_at'].sort_values()
            deltas = sorted_times.diff().dropna().dt.total_seconds()
            
            stats['time_std_dev'] = deltas.std()
            stats['min_time_gap'] = deltas.min()
            stats['active_hour_count'] = group['hour'].nunique()
            stats['max_tweets_one_hour'] = group['hour'].value_counts().max()
        else:
            stats['time_std_dev'] = -1
            stats['min_time_gap'] = -1
            stats['active_hour_count'] = 1
            stats['max_tweets_one_hour'] = 1

        stats['total_posts'] = total_tweets
        user_features.append(stats)
        
    return pd.DataFrame(user_features).fillna(0)

# ==========================================
# 4. EXECUTION
# ==========================================

def main():
    # 1. LOAD & TRAIN
    print("--- TRAINING (AGGRESSIVE MODE) ---")
    df_raw_train, train_bot_ids = load_data(TRAIN_JSON_FILES, TRAIN_BOT_FILES)
    X_train_df = extract_features(df_raw_train)
    y_train = X_train_df['author_id'].apply(lambda x: 1 if x in train_bot_ids else 0)
    
    # Feature Selection
    features = [
        'vocab_diversity', 'repetition_ratio', 'text_len_std', 
        'trigger_word_density', 'link_density', 'mention_density', 'hashtag_density',
        'time_std_dev', 'min_time_gap', 'active_hour_count', 
        'max_tweets_one_hour', 'punct_density'
    ]
    
    # Random Forest Config
    # Increased n_estimators for stability
    # Removed max_depth to allow it to learn complex "smart bot" patterns
    clf = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42)
    clf.fit(X_train_df[features], y_train)

    # 2. PREDICT
    print("\n--- PREDICTING ---")
    df_raw_test, test_bot_ids = load_data(TEST_JSON_FILE, TEST_BOT_FILE)
    X_test_df = extract_features(df_raw_test)
    
    probs = clf.predict_proba(X_test_df[features])[:, 1]
    
    # Apply Threshold
    X_test_df['prob_bot'] = probs
    X_test_df['pred_bot'] = (probs >= CONFIDENCE_THRESHOLD).astype(int)

    # 3. SCORING
    if test_bot_ids:
        y_test = X_test_df['author_id'].apply(lambda x: 1 if x in test_bot_ids else 0)
        y_pred = X_test_df['pred_bot']
        
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        score = (4 * tp) - (1 * fn) - (2 * fp)
        
        print(f"\nRESULTS (Threshold {CONFIDENCE_THRESHOLD}):")
        print(f"TP (Bots Caught):  {tp}")
        print(f"FP (Humans Hit):   {fp}  <-- If this is < 5, you are safe.")
        print(f"FN (Bots Missed):  {fn}")
        print(f"TOTAL SCORE:       {score}")

    # 4. EXPORT JSON FOR REACT
    prob_map = dict(zip(X_test_df['author_id'], X_test_df['prob_bot']))
    with open('python_scores.json', 'w') as f:
        json.dump(prob_map, f)
    print("\n[DONE] Saved python_scores.json")

    # 5. EXPORT TXT
    detected_bots = X_test_df[X_test_df['pred_bot'] == 1]['author_id'].tolist()
    with open('MyTeam.detections.en.txt', 'w') as f:
        for bot_id in detected_bots:
            f.write(f"{bot_id}\n")

if __name__ == "__main__":
    main()