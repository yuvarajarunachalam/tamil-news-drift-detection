"""
Generate Historical Drift Metrics
Creates realistic drift detection data for thesis visualization
"""

import os
import sys
import numpy as np
import random
from datetime import datetime, timedelta
from supabase import create_client, Client
import json
import logging
import time  # Add this to imports


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_detection.drift_detector import DriftDetector, calculate_batch_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalDriftGenerator:
    def __init__(self):
        """Initialize with Supabase connection"""
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        logger.info("✅ Supabase client initialized")
        
        # Load baseline statistics
        self.baseline_stats = self._load_baseline_stats()
        self.baseline_version = self.baseline_stats['version']
        
        # Initialize drift detector
        self.detector = DriftDetector(self.baseline_stats)
        
        # Load all baseline articles
        self.baseline_articles = self._load_baseline_articles()
        logger.info(f"✅ Loaded {len(self.baseline_articles)} baseline articles")
    
    def _load_baseline_stats(self) -> dict:
        """Load latest baseline statistics"""
        response = self.client.table('baseline_statistics')\
            .select('*')\
            .order('version', desc=True)\
            .limit(1)\
            .execute()
        
        if not response.data:
            raise ValueError("No baseline statistics found")
        
        stats = response.data[0]
        
        # Parse JSON fields
        stats['sentiment_distribution'] = json.loads(stats['sentiment_distribution']) \
            if isinstance(stats['sentiment_distribution'], str) else stats['sentiment_distribution']
        
        # Parse embeddings (stored as strings in pgvector format)
        mean_semantic = stats['mean_semantic_embedding']
        if isinstance(mean_semantic, str):
            mean_semantic = json.loads(mean_semantic.replace('[', '').replace(']', ''))
        stats['mean_semantic_embedding'] = mean_semantic
        
        mean_sentiment = stats['mean_sentiment_vector']
        if isinstance(mean_sentiment, str):
            mean_sentiment = json.loads(mean_sentiment.replace('[', '').replace(']', ''))
        stats['mean_sentiment_vector'] = mean_sentiment
        
        logger.info(f"✅ Loaded baseline version {stats['version']}")
        return stats
    
    def _load_baseline_articles(self) -> list:
        """Load all baseline articles with sentiment and embeddings"""
        # Fetch articles with their sentiment and embeddings
        response = self.client.table('news_cleaned')\
            .select('id, title, source, pub_date, content_full')\
            .eq('is_baseline', True)\
            .execute()
        
        articles = response.data
        
        # Fetch sentiment data
        sentiment_response = self.client.table('model_output')\
            .select('*')\
            .eq('is_baseline', True)\
            .execute()
        
        sentiment_map = {s['article_id']: s for s in sentiment_response.data}
        
        # Fetch embeddings
        embedding_response = self.client.table('article_embeddings')\
            .select('*')\
            .eq('is_baseline', True)\
            .execute()
        
        embedding_map = {e['article_id']: e for e in embedding_response.data}
        
        # Combine data
        complete_articles = []
        for article in articles:
            article_id = article['id']
            
            if article_id in sentiment_map and article_id in embedding_map:
                sentiment = sentiment_map[article_id]
                embedding = embedding_map[article_id]
                
                # Parse embeddings
                semantic_emb = embedding['semantic_embedding']
                if isinstance(semantic_emb, str):
                    semantic_emb = json.loads(semantic_emb.replace('[', '').replace(']', ''))
                
                sentiment_vec = embedding['sentiment_vector']
                if isinstance(sentiment_vec, str):
                    sentiment_vec = json.loads(sentiment_vec.replace('[', '').replace(']', ''))
                
                complete_articles.append({
                    'id': article_id,
                    'title': article['title'],
                    'source': article['source'],
                    'pub_date': article['pub_date'],
                    'content_full': article['content_full'],
                    'sentiment_label': sentiment['sentiment_label'],
                    'positive_score': sentiment['positive_score'],
                    'neutral_score': sentiment['neutral_score'],
                    'negative_score': sentiment['negative_score'],
                    'semantic_embedding': np.array(semantic_emb),
                    'sentiment_vector': np.array(sentiment_vec)
                })
        
        return complete_articles
    
    def _create_no_drift_batch(self, batch_size: int = 300) -> list:
        """Create a batch with no drift (random sampling from baseline)"""
        # Use current timestamp as part of seed for true randomness
        random.seed(int(time.time() * 1000000) % (2**32))
        return random.sample(self.baseline_articles, min(batch_size, len(self.baseline_articles)))

    def _create_mild_drift_batch(self, batch_size: int = 300) -> list:
        """Create a batch with mild drift"""
        random.seed(int(time.time() * 1000000) % (2**32))
        
        # Get fresh sample
        batch = random.sample(self.baseline_articles, min(batch_size, len(self.baseline_articles)))
        
        # DEEP COPY to avoid modifying originals
        batch = [article.copy() for article in batch]
        
        # Modify 10-15% of articles
        num_to_modify = int(len(batch) * random.uniform(0.10, 0.15))
        articles_to_modify = random.sample(range(len(batch)), num_to_modify)
        
        for idx in articles_to_modify:
            # Shift sentiment scores
            shift = random.uniform(0.05, 0.10)
            
            if random.choice([True, False]):
                # Boost positive
                batch[idx]['positive_score'] = min(1.0, batch[idx]['positive_score'] + shift)
                batch[idx]['negative_score'] = max(0.0, batch[idx]['negative_score'] - shift/2)
                batch[idx]['neutral_score'] = 1.0 - batch[idx]['positive_score'] - batch[idx]['negative_score']
            else:
                # Boost negative
                batch[idx]['negative_score'] = min(1.0, batch[idx]['negative_score'] + shift)
                batch[idx]['positive_score'] = max(0.0, batch[idx]['positive_score'] - shift/2)
                batch[idx]['neutral_score'] = 1.0 - batch[idx]['negative_score'] - batch[idx]['positive_score']
            
            # Update label
            scores = {
                'positive': batch[idx]['positive_score'],
                'neutral': batch[idx]['neutral_score'],
                'negative': batch[idx]['negative_score']
            }
            batch[idx]['sentiment_label'] = max(scores, key=scores.get)
            
            # Update sentiment vector
            batch[idx]['sentiment_vector'] = np.array([
                batch[idx]['positive_score'],
                batch[idx]['neutral_score'],
                batch[idx]['negative_score']
            ])
        
        return batch

    def _create_strong_drift_batch(self, batch_size: int = 300) -> list:
        """Create a batch with strong drift"""
        random.seed(int(time.time() * 1000000) % (2**32))
        
        # Get fresh sample
        batch = random.sample(self.baseline_articles, min(batch_size, len(self.baseline_articles)))
        
        # DEEP COPY
        batch = [article.copy() for article in batch]
        
        # Modify 30-40% significantly
        num_to_modify = int(len(batch) * random.uniform(0.30, 0.40))
        articles_to_modify = random.sample(range(len(batch)), num_to_modify)
        
        for idx in articles_to_modify:
            if random.choice([True, False]):
                # Make very positive
                batch[idx]['positive_score'] = 0.7 + random.uniform(0, 0.3)
                batch[idx]['negative_score'] = random.uniform(0, 0.1)
                batch[idx]['neutral_score'] = 1.0 - batch[idx]['positive_score'] - batch[idx]['negative_score']
                batch[idx]['sentiment_label'] = 'positive'
            else:
                # Make very negative
                batch[idx]['negative_score'] = 0.7 + random.uniform(0, 0.3)
                batch[idx]['positive_score'] = random.uniform(0, 0.1)
                batch[idx]['neutral_score'] = 1.0 - batch[idx]['negative_score'] - batch[idx]['positive_score']
                batch[idx]['sentiment_label'] = 'negative'
            
            # Update sentiment vector
            batch[idx]['sentiment_vector'] = np.array([
                batch[idx]['positive_score'],
                batch[idx]['neutral_score'],
                batch[idx]['negative_score']
            ])
            
            # Add noise to semantic embedding
            noise = np.random.normal(0, 0.15, batch[idx]['semantic_embedding'].shape)
            batch[idx]['semantic_embedding'] = batch[idx]['semantic_embedding'] + noise
            # Normalize
            batch[idx]['semantic_embedding'] = batch[idx]['semantic_embedding'] / np.linalg.norm(batch[idx]['semantic_embedding'])
        
        return batch
    
    def generate_drift_run(self, detection_date: datetime, drift_type: str) -> dict:
        """
        Generate a single drift detection run
        
        Args:
            detection_date: Date of detection
            drift_type: 'none', 'mild', or 'strong'
            
        Returns:
            Dict with drift metrics
        """
        # Create test batch based on drift type
        if drift_type == 'none':
            test_batch_articles = self._create_no_drift_batch()
        elif drift_type == 'mild':
            test_batch_articles = self._create_mild_drift_batch()
        else:  # strong
            test_batch_articles = self._create_strong_drift_batch()
        
        # Calculate batch metrics
        batch_metrics = calculate_batch_metrics(test_batch_articles)
        
        # Detect drift
        drift_results = self.detector.detect_drift(batch_metrics)
        
        # Prepare record for database
        record = {
            'detection_date': detection_date.date().isoformat(),
            'baseline_version': self.baseline_version,
            'test_article_count': drift_results['test_article_count'],
            'kl_divergence': drift_results['kl_divergence'],
            'cosine_similarity_semantic': drift_results['cosine_similarity_semantic'],
            'cosine_similarity_sentiment': drift_results['cosine_similarity_sentiment'],
            'drift_detected': drift_results['drift_detected'],
            'notes': f"Simulated {drift_type} drift scenario - Severity: {drift_results['severity']}"
        }
        
        return record
    
    def generate_historical_data(self, num_runs: int = 50, days_back: int = 10):
        """
        Generate historical drift detection data
        
        Args:
            num_runs: Number of drift detection runs to generate
            days_back: How many days back to spread the data
        """
        logger.info("=" * 70)
        logger.info("🚀 GENERATING HISTORICAL DRIFT METRICS")
        logger.info("=" * 70)
        
        # Determine drift scenario distribution
        # 60% no drift, 30% mild, 10% strong
        scenarios = (
            ['none'] * int(num_runs * 0.60) +
            ['mild'] * int(num_runs * 0.30) +
            ['strong'] * int(num_runs * 0.10)
        )
        
        # Shuffle scenarios
        random.shuffle(scenarios)
        
        # Ensure we have exactly num_runs
        scenarios = scenarios[:num_runs]
        
        logger.info(f"\n📊 Scenario Distribution:")
        logger.info(f"   No drift: {scenarios.count('none')} ({scenarios.count('none')/num_runs*100:.1f}%)")
        logger.info(f"   Mild drift: {scenarios.count('mild')} ({scenarios.count('mild')/num_runs*100:.1f}%)")
        logger.info(f"   Strong drift: {scenarios.count('strong')} ({scenarios.count('strong')/num_runs*100:.1f}%)")
        
        # Generate timestamps spread over days_back
        start_date = datetime.now() - timedelta(days=days_back)
        timestamps = []
        
        for i in range(num_runs):
            # Random day within range
            random_day = start_date + timedelta(
                days=random.randint(0, days_back),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            timestamps.append(random_day)
        
        # Sort timestamps
        timestamps.sort()
        
        # Generate drift runs
        drift_records = []
        
        for i, (timestamp, scenario) in enumerate(zip(timestamps, scenarios), 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Run {i}/{num_runs} - {timestamp.strftime('%Y-%m-%d %H:%M')} - Scenario: {scenario.upper()}")
            logger.info(f"{'='*70}")
            
            record = self.generate_drift_run(timestamp, scenario)
            drift_records.append(record)
        
        # Insert all records to database
        logger.info(f"\n{'='*70}")
        logger.info("💾 INSERTING TO DATABASE")
        logger.info(f"{'='*70}")
        
        try:
            response = self.client.table('drift_metrics').insert(drift_records).execute()
            logger.info(f"✅ Successfully inserted {len(response.data)} drift metric records")
        except Exception as e:
            logger.error(f"❌ Error inserting records: {e}")
            raise
        
        # Summary statistics
        drift_detected_count = sum(1 for r in drift_records if r['drift_detected'])
        
        logger.info(f"\n{'='*70}")
        logger.info("✅ GENERATION COMPLETE!")
        logger.info(f"{'='*70}")
        logger.info(f"\n📊 Summary:")
        logger.info(f"   Total runs: {len(drift_records)}")
        logger.info(f"   Drift detected: {drift_detected_count} ({drift_detected_count/len(drift_records)*100:.1f}%)")
        logger.info(f"   Time range: {timestamps[0].strftime('%Y-%m-%d')} to {timestamps[-1].strftime('%Y-%m-%d')}")
        logger.info(f"\n🔍 Next step: Query drift_metrics table and create visualizations!")
        logger.info(f"{'='*70}\n")


def main():
    """Main execution"""
    generator = HistoricalDriftGenerator()
    generator.generate_historical_data(num_runs=50, days_back=10)


if __name__ == "__main__":
    main()