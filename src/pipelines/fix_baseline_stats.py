"""
Fix: Calculate Baseline Statistics
Run this once to generate missing baseline statistics
"""

import sys
import os
from datetime import datetime
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.supabase_client import SupabaseClient

def calculate_baseline_statistics():
    db = SupabaseClient()
    
    print("Fetching sentiment scores...")
    sentiment_scores = db.get_all_sentiment_scores(is_baseline=True)
    
    print("Fetching embeddings...")
    embeddings = db.get_all_embeddings(is_baseline=True)
    
    if not sentiment_scores or not embeddings:
        print("❌ No baseline data found!")
        return
    
    print(f"Processing {len(sentiment_scores)} articles...")
    
    # Calculate sentiment distribution
    total = len(sentiment_scores)
    positive_count = sum(1 for s in sentiment_scores if s['sentiment_label'] == 'positive')
    neutral_count = sum(1 for s in sentiment_scores if s['sentiment_label'] == 'neutral')
    negative_count = sum(1 for s in sentiment_scores if s['sentiment_label'] == 'negative')
    
    sentiment_distribution = {
        'positive': round(positive_count / total, 4),
        'neutral': round(neutral_count / total, 4),
        'negative': round(negative_count / total, 4)
    }
    
    # Parse embeddings
    parsed_semantic = []
    parsed_sentiment = []
    
    for e in embeddings:
        # Parse semantic embedding
        if isinstance(e['semantic_embedding'], str):
            semantic_str = e['semantic_embedding'].strip('[]')
            semantic_array = [float(x.strip()) for x in semantic_str.split(',')]
            parsed_semantic.append(semantic_array)
        else:
            parsed_semantic.append(e['semantic_embedding'])
        
        # Parse sentiment vector
        if isinstance(e['sentiment_vector'], str):
            sentiment_str = e['sentiment_vector'].strip('[]')
            sentiment_array = [float(x.strip()) for x in sentiment_str.split(',')]
            parsed_sentiment.append(sentiment_array)
        else:
            parsed_sentiment.append(e['sentiment_vector'])
    
    # Calculate mean embeddings
    semantic_embeddings = np.array(parsed_semantic)
    sentiment_vectors = np.array(parsed_sentiment)
    
    mean_semantic_embedding = np.mean(semantic_embeddings, axis=0).tolist()
    mean_sentiment_vector = np.mean(sentiment_vectors, axis=0).tolist()
    
    # Calculate standard deviations
    positive_scores = [s['positive_score'] for s in sentiment_scores]
    neutral_scores = [s['neutral_score'] for s in sentiment_scores]
    negative_scores = [s['negative_score'] for s in sentiment_scores]
    
    std_sentiment_scores = {
        'positive_std': round(float(np.std(positive_scores)), 4),
        'neutral_std': round(float(np.std(neutral_scores)), 4),
        'negative_std': round(float(np.std(negative_scores)), 4)
    }
    
    # Prepare baseline statistics
    baseline_stats = {
        'version': 1,
        'baseline_date': datetime.now().isoformat(),
        'total_articles': total,
        'sentiment_distribution': sentiment_distribution,
        'mean_semantic_embedding': mean_semantic_embedding,
        'mean_sentiment_vector': mean_sentiment_vector,
        'std_sentiment_scores': std_sentiment_scores
    }
    
    # Insert into database
    print("Storing baseline statistics...")
    db.insert_baseline_statistics(baseline_stats)
    
    print("✅ Baseline statistics stored successfully!")
    print(f"   - Total articles: {total}")
    print(f"   - Sentiment distribution: {sentiment_distribution}")

if __name__ == "__main__":
    calculate_baseline_statistics()