#!/usr/bin/env python3
"""
Drift Detection Pipeline
Main orchestrator script for running the complete drift detection workflow
"""

import os
import sys
from datetime import datetime

# ADD THESE DEBUG LINES
print("\n" + "="*60)
print("🔧 SCRIPT STARTED - DEBUG MODE")
print(f"🔧 Current directory: {os.getcwd()}")
print(f"🔧 Python executable: {sys.executable}")
print("="*60 + "\n")
# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.supabase_client import get_client
from utils.sampling import sample_articles
from modeling.sentiment_analyzer import SentimentAnalyzer
from modeling.embedding_generator import EmbeddingGenerator
from modeling.drift_calculator import DriftCalculator


def main():
    """Main pipeline execution"""
    
    print("\n" + "="*60)
    print("🚀 TAMIL NEWS DRIFT DETECTION PIPELINE")
    print("="*60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Configuration from environment
    baseline_size = 10  # int(os.getenv("BASELINE_SAMPLE_SIZE", 500))
    test_size = 10      # int(os.getenv("TEST_SAMPLE_SIZE", 500))
    batch_size = 5  
    
    kl_threshold = float(os.getenv("KL_DIVERGENCE_THRESHOLD", 0.05))
    cosine_threshold = float(os.getenv("COSINE_SIMILARITY_THRESHOLD", 0.85))
    
    try:
        # ========================================
        # STEP 1: DATABASE CONNECTION
        # ========================================
        print("\n📡 STEP 1: Connecting to database...")
        db = get_client()
        
        # ========================================
        # STEP 2: SAMPLE ARTICLES
        # ========================================
        print("\n📊 STEP 2: Sampling articles...")
        baseline_articles, test_articles = sample_articles(
            db,
            baseline_size=baseline_size,
            test_size=test_size
        )
        
        baseline_ids = [a["id"] for a in baseline_articles]
        test_ids = [a["id"] for a in test_articles]
        
        # ========================================
        # STEP 3: SENTIMENT ANALYSIS - BASELINE
        # ========================================
        print("\n🤖 STEP 3: Analyzing sentiment for baseline articles...")
        sentiment_model = SentimentAnalyzer()
        baseline_outputs = sentiment_model.analyze_batch(baseline_articles, batch_size=batch_size)
        
        print(f"\n💾 Storing {len(baseline_outputs)} baseline sentiment results...")
        db.batch_insert_model_outputs(baseline_outputs)
        print("✅ Baseline sentiment results stored!")
        
        # ========================================
        # STEP 4: SENTIMENT ANALYSIS - TEST
        # ========================================
        print("\n🤖 STEP 4: Analyzing sentiment for test articles...")
        test_outputs = sentiment_model.analyze_batch(test_articles, batch_size=batch_size)
        
        print(f"\n💾 Storing {len(test_outputs)} test sentiment results...")
        db.batch_insert_model_outputs(test_outputs)
        print("✅ Test sentiment results stored!")
        
        # ========================================
        # STEP 5: GENERATE EMBEDDINGS - BASELINE
        # ========================================
        print("\n🧠 STEP 5: Generating embeddings for baseline articles...")
        embedding_model = EmbeddingGenerator()
        baseline_embeddings = embedding_model.process_articles_with_sentiment(
            baseline_articles,
            baseline_outputs,
            batch_size=batch_size
        )
        
        print(f"\n💾 Storing {len(baseline_embeddings)} baseline embeddings...")
        db.batch_insert_embeddings(baseline_embeddings)
        print("✅ Baseline embeddings stored!")
        
        # ========================================
        # STEP 6: GENERATE EMBEDDINGS - TEST
        # ========================================
        print("\n🧠 STEP 6: Generating embeddings for test articles...")
        test_embeddings = embedding_model.process_articles_with_sentiment(
            test_articles,
            test_outputs,
            batch_size=batch_size
        )
        
        print(f"\n💾 Storing {len(test_embeddings)} test embeddings...")
        db.batch_insert_embeddings(test_embeddings)
        print("✅ Test embeddings stored!")
        
        # ========================================
        # STEP 7: CALCULATE DRIFT METRICS
        # ========================================
        print("\n📐 STEP 7: Calculating drift metrics...")
        calculator = DriftCalculator(
            kl_threshold=kl_threshold,
            cosine_threshold=cosine_threshold
        )
        
        drift_metrics = calculator.calculate_all_drift_metrics(
            baseline_outputs=baseline_outputs,
            test_outputs=test_outputs,
            baseline_embeddings=baseline_embeddings,
            test_embeddings=test_embeddings,
            baseline_article_ids=baseline_ids,
            test_article_ids=test_ids
        )
        
        # ========================================
        # STEP 8: STORE DRIFT METRICS
        # ========================================
        print("\n💾 STEP 8: Storing drift metrics...")
        for metric in drift_metrics:
            db.insert_drift_metric(metric)
        
        print("✅ All drift metrics stored!")
        
        # ========================================
        # FINAL SUMMARY
        # ========================================
        print("\n" + "="*60)
        print(" PIPELINE EXECUTION COMPLETE!")
        print("="*60)
        print(f" Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n Processing Summary:")
        print(f"  Baseline articles: {len(baseline_articles)}")
        print(f"  Test articles: {len(test_articles)}")
        print(f"  Sentiment analyses: {len(baseline_outputs) + len(test_outputs)}")
        print(f"  Embeddings generated: {len(baseline_embeddings) + len(test_embeddings)}")
        print(f"  Drift metrics calculated: {len(drift_metrics)}")
        
        drift_detected_count = sum(1 for m in drift_metrics if m["drift_detected"])
        print(f"\n Drift Status: {drift_detected_count}/{len(drift_metrics)} metrics detected drift")
        
        print("\n✅ All results stored in Supabase database!")
        print("="*60)
        
    except Exception as e:
        print(f"\n ERROR: Pipeline failed!")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
