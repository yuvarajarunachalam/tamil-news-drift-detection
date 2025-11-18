"""
Continuous Drift Monitoring
Runs drift detection every N minutes and sends metrics to DataDog
"""

import os
import sys
import time
import numpy as np
import random
from datetime import datetime, timedelta
from supabase import create_client, Client
from datadog import initialize, statsd
import json
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_detection.drift_detector import DriftDetector, calculate_batch_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContinuousDriftMonitor:
    def __init__(self):
        """Initialize monitor with Supabase and DataDog connections"""
        # Supabase
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        logger.info("✅ Supabase client initialized")
        
        # DataDog
        datadog_api_key = os.getenv('DATADOG_API_KEY')
        
        if not datadog_api_key:
            raise ValueError("DATADOG_API_KEY must be set")
        
        options = {
            'api_key': datadog_api_key,
            'app_key': os.getenv('DATADOG_APP_KEY', ''),  # Optional
            'statsd_host': 'localhost',
            'statsd_port': 8125,
        }
        
        initialize(**options)
        logger.info("✅ DataDog initialized (US5 region)")
        
        # Load baseline statistics
        self.baseline_stats = self._load_baseline_stats()
        self.baseline_version = self.baseline_stats['version']
        
        # Initialize drift detector
        self.detector = DriftDetector(self.baseline_stats)
        
        # Load all baseline articles
        self.baseline_articles = self._load_baseline_articles()
        logger.info(f"✅ Loaded {len(self.baseline_articles)} baseline articles")
        
        # Monitoring state
        self.run_count = 0
        self.start_time = datetime.now()
    
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
        
        # Helper function to parse pgvector format
        def parse_vector(vector_str):
            """Parse pgvector string to numpy array"""
            if isinstance(vector_str, str):
                # Remove brackets and split by comma
                vector_str = vector_str.strip('[]')
                values = [float(x.strip()) for x in vector_str.split(',')]
                return np.array(values)
            elif isinstance(vector_str, list):
                return np.array(vector_str)
            else:
                return np.array(vector_str)
        
        # Parse embeddings using helper function
        stats['mean_semantic_embedding'] = parse_vector(stats['mean_semantic_embedding'])
        stats['mean_sentiment_vector'] = parse_vector(stats['mean_sentiment_vector'])
        
        logger.info(f"✅ Loaded baseline version {stats['version']}")
        logger.info(f"   Semantic embedding shape: {stats['mean_semantic_embedding'].shape}")
        logger.info(f"   Sentiment vector shape: {stats['mean_sentiment_vector'].shape}")
        
        return stats
    def _load_baseline_articles(self) -> list:
        """Load all baseline articles with sentiment and embeddings"""
        # Fetch articles - LIMIT TO 1000 FOR SPEED
        response = self.client.table('news_cleaned')\
            .select('id, title, source, pub_date, content_full')\
            .eq('is_baseline', True)\
            .limit(1000)\
            .execute()
        
        articles = response.data
        logger.info(f"   Fetched {len(articles)} articles")
        
        article_ids = [a['id'] for a in articles]
        
        # Fetch sentiment data in SMALLER BATCHES (100 at a time)
        sentiment_map = {}
        BATCH_SIZE = 100
        
        for i in range(0, len(article_ids), BATCH_SIZE):
            batch_ids = article_ids[i:i+BATCH_SIZE]
            try:
                sentiment_response = self.client.table('model_output')\
                    .select('*')\
                    .in_('article_id', batch_ids)\
                    .execute()
                sentiment_map.update({s['article_id']: s for s in sentiment_response.data})
                logger.info(f"   Fetched sentiment batch {i//BATCH_SIZE + 1}/{(len(article_ids) + BATCH_SIZE - 1)//BATCH_SIZE}")
            except Exception as e:
                logger.warning(f"   Error fetching sentiment batch: {e}")
        
        logger.info(f"   Total sentiment records: {len(sentiment_map)}")
        
        # Fetch embeddings in SMALLER BATCHES
        embedding_map = {}
        for i in range(0, len(article_ids), BATCH_SIZE):
            batch_ids = article_ids[i:i+BATCH_SIZE]
            try:
                embedding_response = self.client.table('article_embeddings')\
                    .select('*')\
                    .in_('article_id', batch_ids)\
                    .execute()
                embedding_map.update({e['article_id']: e for e in embedding_response.data})
                logger.info(f"   Fetched embedding batch {i//BATCH_SIZE + 1}/{(len(article_ids) + BATCH_SIZE - 1)//BATCH_SIZE}")
            except Exception as e:
                logger.warning(f"   Error fetching embedding batch: {e}")
        
        logger.info(f"   Total embedding records: {len(embedding_map)}")
        
        # Helper function to parse vectors
        def parse_vector(vector_str):
            """Parse pgvector string to numpy array"""
            if isinstance(vector_str, str):
                vector_str = vector_str.strip('[]')
                values = [float(x.strip()) for x in vector_str.split(',')]
                return np.array(values)
            elif isinstance(vector_str, list):
                return np.array(vector_str)
            else:
                return np.array(vector_str)
        
        # Combine data
        complete_articles = []
        for article in articles:
            article_id = article['id']
            
            if article_id in sentiment_map and article_id in embedding_map:
                sentiment = sentiment_map[article_id]
                embedding = embedding_map[article_id]
                
                try:
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
                        'semantic_embedding': parse_vector(embedding['semantic_embedding']),
                        'sentiment_vector': parse_vector(embedding['sentiment_vector'])
                    })
                except Exception as e:
                    logger.warning(f"   Skipping article {article_id}: {e}")
                    continue
        
        logger.info(f"   Successfully loaded {len(complete_articles)} complete articles")
        return complete_articles
    
    def _create_test_batch(self, scenario: str, batch_size: int = 300) -> list:
        """Create test batch based on scenario"""
        batch = random.sample(self.baseline_articles, min(batch_size, len(self.baseline_articles)))
        
        if scenario == 'none':
            # No modifications - natural variation
            return batch
        
        elif scenario == 'mild':
            # Modify 10-15% slightly
            num_to_modify = int(len(batch) * random.uniform(0.10, 0.15))
            articles_to_modify = random.sample(range(len(batch)), num_to_modify)
            
            for idx in articles_to_modify:
                shift = random.uniform(0.05, 0.10)
                
                if random.choice([True, False]):
                    batch[idx]['positive_score'] = min(1.0, batch[idx]['positive_score'] + shift)
                    batch[idx]['negative_score'] = max(0.0, batch[idx]['negative_score'] - shift/2)
                else:
                    batch[idx]['negative_score'] = min(1.0, batch[idx]['negative_score'] + shift)
                    batch[idx]['positive_score'] = max(0.0, batch[idx]['positive_score'] - shift/2)
                
                scores = {
                    'positive': batch[idx]['positive_score'],
                    'neutral': batch[idx]['neutral_score'],
                    'negative': batch[idx]['negative_score']
                }
                batch[idx]['sentiment_label'] = max(scores, key=scores.get)
        
        else:  # strong
            # Modify 30-40% significantly
            num_to_modify = int(len(batch) * random.uniform(0.30, 0.40))
            articles_to_modify = random.sample(range(len(batch)), num_to_modify)
            
            for idx in articles_to_modify:
                if random.choice([True, False]):
                    batch[idx]['positive_score'] = min(1.0, 0.7 + random.uniform(0, 0.3))
                    batch[idx]['negative_score'] = max(0.0, random.uniform(0, 0.1))
                    batch[idx]['neutral_score'] = max(0.0, 1.0 - batch[idx]['positive_score'] - batch[idx]['negative_score'])
                    batch[idx]['sentiment_label'] = 'positive'
                else:
                    batch[idx]['negative_score'] = min(1.0, 0.7 + random.uniform(0, 0.3))
                    batch[idx]['positive_score'] = max(0.0, random.uniform(0, 0.1))
                    batch[idx]['neutral_score'] = max(0.0, 1.0 - batch[idx]['negative_score'] - batch[idx]['positive_score'])
                    batch[idx]['sentiment_label'] = 'negative'
                
                noise = np.random.normal(0, 0.1, batch[idx]['semantic_embedding'].shape)
                batch[idx]['semantic_embedding'] = batch[idx]['semantic_embedding'] + noise
        
        return batch
    
    def _determine_scenario(self, run_number: int, total_runs: int) -> str:
        """
        Determine scenario based on run number
        Simulates realistic monitoring:
        - First hour (0-33%): Mostly normal, 1 mild alert
        - Second hour (33-66%): Mix of normal and mild
        - Third hour (66-100%): 1 strong alert towards end
        """
        progress = run_number / total_runs
        
        if progress < 0.33:  # First hour
            # 90% no drift, 10% mild
            return 'mild' if random.random() < 0.10 else 'none'
        
        elif progress < 0.66:  # Second hour
            # 70% no drift, 30% mild
            return 'mild' if random.random() < 0.30 else 'none'
        
        else:  # Third hour
            # 60% no drift, 30% mild, 10% strong
            rand = random.random()
            if rand < 0.10:
                return 'strong'
            elif rand < 0.40:
                return 'mild'
            else:
                return 'none'
    
    def run_drift_detection(self, run_number: int, total_runs: int) -> dict:
        """Run a single drift detection check"""
        start = time.time()
        
        # Determine scenario
        scenario = self._determine_scenario(run_number, total_runs)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 Run {run_number}/{total_runs} - Scenario: {scenario.upper()}")
        logger.info(f"{'='*70}")
        
        # Create test batch
        test_batch_articles = self._create_test_batch(scenario)
        
        # Calculate metrics
        batch_metrics = calculate_batch_metrics(test_batch_articles)
        
        # Detect drift
        drift_results = self.detector.detect_drift(batch_metrics)
        
        processing_time = time.time() - start
        
        # Send to DataDog
        self._send_to_datadog(drift_results, processing_time)
        
        # Save to database
        record = {
            'detection_date': datetime.now().date().isoformat(),
            'baseline_version': self.baseline_version,
            'test_article_count': drift_results['test_article_count'],
            'kl_divergence': drift_results['kl_divergence'],
            'cosine_similarity_semantic': drift_results['cosine_similarity_semantic'],
            'cosine_similarity_sentiment': drift_results['cosine_similarity_sentiment'],
            'drift_detected': drift_results['drift_detected'],
            'notes': f"Continuous monitoring run {run_number} - Scenario: {scenario} - Severity: {drift_results['severity']}"
        }
        
        try:
            self.client.table('drift_metrics').insert(record).execute()
            logger.info("✅ Saved to database")
        except Exception as e:
            logger.error(f"❌ Database error: {e}")
        
        return drift_results
    
    def _send_to_datadog(self, drift_results: dict, processing_time: float):
        """Send metrics to DataDog"""
        try:
            # Drift metrics
            statsd.gauge('drift.kl_divergence', drift_results['kl_divergence'], 
                        tags=['env:production', 'service:drift-detection'])
            
            statsd.gauge('drift.cosine_similarity_semantic', drift_results['cosine_similarity_semantic'],
                        tags=['env:production', 'service:drift-detection'])
            
            statsd.gauge('drift.cosine_similarity_sentiment', drift_results['cosine_similarity_sentiment'],
                        tags=['env:production', 'service:drift-detection'])
            
            # Alert counter
            if drift_results['drift_detected']:
                statsd.increment('drift.alerts', 
                               tags=['env:production', 'service:drift-detection', 
                                     f"severity:{drift_results['severity'].lower()}"])
            
            # Processing time
            statsd.timing('drift.processing_time_ms', processing_time * 1000,
                         tags=['env:production', 'service:drift-detection'])
            
            # Batch size
            statsd.gauge('drift.test_batch_size', drift_results['test_article_count'],
                        tags=['env:production', 'service:drift-detection'])
            
            logger.info("✅ Metrics sent to DataDog")
            
        except Exception as e:
            logger.error(f"⚠️ DataDog error: {e}")
    
    def run_continuous_monitoring(self, duration_hours: float = 3.0, 
                                  interval_minutes: int = 5):
        """
        Run continuous monitoring for specified duration
        
        Args:
            duration_hours: How long to run (default: 3 hours)
            interval_minutes: How often to check (default: 5 minutes)
        """
        duration_seconds = duration_hours * 3600
        interval_seconds = interval_minutes * 60
        
        total_runs = int(duration_seconds / interval_seconds)
        
        logger.info("=" * 70)
        logger.info("🚀 STARTING CONTINUOUS DRIFT MONITORING")
        logger.info("=" * 70)
        logger.info(f"\n📊 Configuration:")
        logger.info(f"   Duration: {duration_hours} hours")
        logger.info(f"   Interval: {interval_minutes} minutes")
        logger.info(f"   Total runs: {total_runs}")
        logger.info(f"   Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   End time: {(self.start_time + datetime.timedelta(hours=duration_hours)).strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
        # Send initial heartbeat to DataDog
        statsd.event('Drift Monitoring Started', 
                    f'Starting {total_runs} drift detection runs over {duration_hours} hours',
                    tags=['env:production', 'service:drift-detection'])
        
        run_number = 1
        
        try:
            while run_number <= total_runs:
                # Run drift detection
                drift_results = self.run_drift_detection(run_number, total_runs)
                
                self.run_count = run_number
                run_number += 1
                
                # Log progress
                elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
                remaining = duration_hours - elapsed
                
                logger.info(f"\n⏱️  Progress: {run_number-1}/{total_runs} runs complete")
                logger.info(f"   Elapsed: {elapsed:.2f} hours")
                logger.info(f"   Remaining: {remaining:.2f} hours")
                
                # Wait for next interval (unless this is the last run)
                if run_number <= total_runs:
                    logger.info(f"   ⏸️  Waiting {interval_minutes} minutes until next check...\n")
                    time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Monitoring interrupted by user")
        
        except Exception as e:
            logger.error(f"\n❌ Error during monitoring: {e}")
            raise
        
        finally:
            # Send completion event to DataDog
            statsd.event('Drift Monitoring Completed', 
                        f'Completed {self.run_count} drift detection runs',
                        tags=['env:production', 'service:drift-detection'])
            
            logger.info("\n" + "=" * 70)
            logger.info("✅ MONITORING SESSION COMPLETE")
            logger.info("=" * 70)
            logger.info(f"\n📊 Session Summary:")
            logger.info(f"   Total runs: {self.run_count}")
            logger.info(f"   Duration: {(datetime.now() - self.start_time).total_seconds() / 3600:.2f} hours")
            logger.info(f"   Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 70)


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run continuous drift monitoring')
    parser.add_argument('--duration', type=float, default=3.0, 
                       help='Duration in hours (default: 3.0)')
    parser.add_argument('--interval', type=int, default=5,
                       help='Interval in minutes (default: 5)')
    
    args = parser.parse_args()
    
    monitor = ContinuousDriftMonitor()
    monitor.run_continuous_monitoring(
        duration_hours=args.duration,
        interval_minutes=args.interval
    )


if __name__ == "__main__":
    main()