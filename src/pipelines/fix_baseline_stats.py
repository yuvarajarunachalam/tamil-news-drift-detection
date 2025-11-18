"""
Fix: Calculate Baseline Statistics  
Run this once to generate missing baseline statistics from ALL 4548 articles
"""

import sys
import os
from datetime import datetime
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.supabase_client import SupabaseClient

def calculate_baseline_statistics():
    db = SupabaseClient()
    
    print("Fetching ALL sentiment scores (with pagination)...")
    sentiment_scores = db.get_all_sentiment_scores(is_baseline=True)
    print(f"✅ Fetched {len(sentiment_scores)} sentiment scores")
    
    print("Fetching ALL embeddings (with pagination)...")
    embeddings = db.get_all_embeddings(is_baseline=True)
    print(f"✅ Fetched {len(embeddings)} embeddings")
    
    if not sentiment_scores or not embeddings:
        print("❌ No baseline data found!")
        return
    
    if len(sentiment_scores) != len(embeddings):
        print(f"⚠️ Warning: Mismatch! {len(sentiment_scores)} sentiment scores vs {len(embeddings)} embeddings")
    
    print(f"\n📊 Processing {len(sentiment_scores)} articles...")
    
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
    
    print(f"\n📈 Sentiment Distribution:")
    print(f"   Positive: {positive_count} ({sentiment_distribution['positive']*100:.1f}%)")
    print(f"   Neutral:  {neutral_count} ({sentiment_distribution['neutral']*100:.1f}%)")
    print(f"   Negative: {negative_count} ({sentiment_distribution['negative']*100:.1f}%)")
    
    # Parse embeddings
    print("\n🧠 Parsing embeddings...")
    parsed_semantic = []
    parsed_sentiment = []
    
    for i, e in enumerate(embeddings):
        if i % 500 == 0:
            print(f"   Parsed {i}/{len(embeddings)} embeddings...")
        
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
    
    print(f"✅ Parsed all {len(embeddings)} embeddings")
    
    # Calculate mean embeddings
    print("\n🔢 Calculating mean embeddings...")
    semantic_embeddings = np.array(parsed_semantic)
    sentiment_vectors = np.array(parsed_sentiment)
    
    mean_semantic_embedding = np.mean(semantic_embeddings, axis=0).tolist()
    mean_sentiment_vector = np.mean(sentiment_vectors, axis=0).tolist()
    
    # Calculate standard deviations
    print("📊 Calculating standard deviations...")
    positive_scores = [s['positive_score'] for s in sentiment_scores]
    neutral_scores = [s['neutral_score'] for s in sentiment_scores]
    negative_scores = [s['negative_score'] for s in sentiment_scores]
    
    std_sentiment_scores = {
        'positive_std': round(float(np.std(positive_scores)), 4),
        'neutral_std': round(float(np.std(neutral_scores)), 4),
        'negative_std': round(float(np.std(negative_scores)), 4)
    }
    
    print(f"\n📉 Standard Deviations:")
    print(f"   Positive: {std_sentiment_scores['positive_std']}")
    print(f"   Neutral:  {std_sentiment_scores['neutral_std']}")
    print(f"   Negative: {std_sentiment_scores['negative_std']}")
    
    # Delete old baseline statistics (version 1)
    print("\n🗑️ Deleting old baseline statistics...")
    try:
        db.client.table('baseline_statistics').delete().eq('version', 1).execute()
        print("✅ Old statistics deleted")
    except Exception as e:
        print(f"⚠️ Could not delete old stats: {e}")
    
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
    print("\n💾 Storing baseline statistics...")
    db.insert_baseline_statistics(baseline_stats)
    
    print("\n✅ Baseline statistics stored successfully!")
    print(f"   📊 Total articles: {total}")
    print(f"   📈 Sentiment distribution: {sentiment_distribution}")
    print(f"   📉 Std deviations: {std_sentiment_scores}")

if __name__ == "__main__":
    calculate_baseline_statistics()