"""
Drift Detector
Calculates drift metrics by comparing new articles against baseline
"""

import sys
import os
from datetime import datetime, date
from typing import Dict, Any, List
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.supabase_client import SupabaseClient
from monitoring.datadog_client import DataDogClient
from utils.logger import setup_logger


class DriftDetector:
    """Detect drift between new articles and baseline"""
    
    def __init__(self):
        """Initialize drift detector"""
        self.db = SupabaseClient()
        self.datadog = DataDogClient()
        self.logger = setup_logger()
        
        # Drift thresholds
        self.kl_warning = 0.15
        self.kl_critical = 0.25
        self.cosine_warning = 0.85
    
    def run(self, new_article_ids: List[str] = None):
        """
        Main execution method
        
        Args:
            new_article_ids: List of new article IDs to check (if None, check today's articles)
        """
        self.logger.info("Starting drift detection...")
        
        # Get baseline statistics
        baseline_stats = self._get_baseline_statistics()
        
        if not baseline_stats:
            self.logger.error("No baseline statistics found")
            return
        
        # Get new articles (either provided IDs or today's articles)
        if new_article_ids:
            new_sentiment = self._get_sentiment_by_ids(new_article_ids)
            new_embeddings = self._get_embeddings_by_ids(new_article_ids)
        else:
            new_sentiment = self._get_todays_sentiment()
            new_embeddings = self._get_todays_embeddings()
        
        if not new_sentiment or not new_embeddings:
            self.logger.warning("No new articles to compare")
            return
        
        self.logger.info(f"Comparing {len(new_sentiment)} new articles against baseline")
        
        # Calculate drift metrics
        metrics = self._calculate_drift_metrics(baseline_stats, new_sentiment, new_embeddings)
        
        # Store metrics in database
        self._store_drift_metrics(metrics, len(new_sentiment))
        
        # Send to DataDog
        self.datadog.send_drift_metrics(metrics)
        
        # Check for drift and send alerts
        drift_detected = self._check_drift_thresholds(metrics)
        
        if drift_detected:
            self.logger.warning(f"Drift detected! Metrics: {metrics}")
        else:
            self.logger.info(f"No significant drift detected. Metrics: {metrics}")
        
        self.logger.info("Drift detection complete")
    
    def _get_baseline_statistics(self) -> Dict[str, Any]:
        """Get baseline statistics from database"""
        response = self.db.client.table('baseline_statistics').select('*').eq('version', 1).execute()
        
        if response.data:
            return response.data[0]
        return None
    
    def _get_todays_sentiment(self) -> List[Dict[str, Any]]:
        """Get sentiment scores for today's new articles"""
        today = date.today().isoformat()
        response = self.db.client.table('model_output').select('*').eq('is_baseline', False).gte('processed_at', today).execute()
        return response.data
    
    def _get_todays_embeddings(self) -> List[Dict[str, Any]]:
        """Get embeddings for today's new articles"""
        today = date.today().isoformat()
        response = self.db.client.table('article_embeddings').select('*').eq('is_baseline', False).gte('created_at', today).execute()
        return response.data
    
    def _get_sentiment_by_ids(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        """Get sentiment scores for specific article IDs"""
        response = self.db.client.table('model_output').select('*').in_('article_id', article_ids).execute()
        return response.data
    
    def _get_embeddings_by_ids(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        """Get embeddings for specific article IDs"""
        response = self.db.client.table('article_embeddings').select('*').in_('article_id', article_ids).execute()
        return response.data
    
    def _calculate_drift_metrics(
    self,
    baseline_stats: Dict[str, Any],
    new_sentiment: List[Dict[str, Any]],
    new_embeddings: List[Dict[str, Any]]
) -> Dict[str, float]:
        """Calculate drift metrics"""
        
        # 1. KL Divergence (sentiment distribution)
        baseline_dist = baseline_stats['sentiment_distribution']
        new_dist = self._calculate_sentiment_distribution(new_sentiment)
        
        kl_div = self._calculate_kl_divergence(baseline_dist, new_dist)
        
        # 2. Parse baseline embeddings (they're stored as strings)
        if isinstance(baseline_stats['mean_semantic_embedding'], str):
            baseline_semantic_str = baseline_stats['mean_semantic_embedding'].strip('[]')
            baseline_semantic = np.array([float(x.strip()) for x in baseline_semantic_str.split(',')])
        else:
            baseline_semantic = np.array(baseline_stats['mean_semantic_embedding'])
        
        if isinstance(baseline_stats['mean_sentiment_vector'], str):
            baseline_sentiment_str = baseline_stats['mean_sentiment_vector'].strip('[]')
            baseline_sentiment_vec = np.array([float(x.strip()) for x in baseline_sentiment_str.split(',')])
        else:
            baseline_sentiment_vec = np.array(baseline_stats['mean_sentiment_vector'])
        
        # 3. Parse new embeddings (also stored as strings)
        parsed_semantic = []
        parsed_sentiment = []
        
        for e in new_embeddings:
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
        
        # 4. Calculate means for new articles
        new_semantic = np.mean(parsed_semantic, axis=0)
        new_sentiment_vec = np.mean(parsed_sentiment, axis=0)
        
        # 5. Cosine Similarity - Semantic (topic drift)
        cosine_semantic = 1 - cosine(baseline_semantic, new_semantic)
        
        # 6. Cosine Similarity - Sentiment Vectors
        cosine_sentiment = 1 - cosine(baseline_sentiment_vec, new_sentiment_vec)
        
        return {
            'kl_divergence': float(kl_div),
            'cosine_semantic': float(cosine_semantic),
            'cosine_sentiment': float(cosine_sentiment)
        }
    
    def _calculate_sentiment_distribution(self, sentiment_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate sentiment distribution from sentiment data"""
        total = len(sentiment_data)
        positive = sum(1 for s in sentiment_data if s['sentiment_label'] == 'positive')
        neutral = sum(1 for s in sentiment_data if s['sentiment_label'] == 'neutral')
        negative = sum(1 for s in sentiment_data if s['sentiment_label'] == 'negative')
        
        return {
            'positive': positive / total,
            'neutral': neutral / total,
            'negative': negative / total
        }
    
    def _calculate_kl_divergence(self, baseline_dist: Dict[str, float], new_dist: Dict[str, float]) -> float:
        """Calculate KL divergence between two distributions"""
        # Convert to arrays in same order
        p = np.array([baseline_dist['positive'], baseline_dist['neutral'], baseline_dist['negative']])
        q = np.array([new_dist['positive'], new_dist['neutral'], new_dist['negative']])
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        # Calculate KL divergence
        return float(entropy(p, q))
    
    def _store_drift_metrics(self, metrics: Dict[str, float], article_count: int):
        """Store drift metrics in database"""
        data = {
            'detection_date': date.today().isoformat(),
            'baseline_version': 1,
            'test_article_count': article_count,
            'kl_divergence': metrics['kl_divergence'],
            'cosine_similarity_semantic': metrics['cosine_semantic'],
            'cosine_similarity_sentiment': metrics['cosine_sentiment'],
            'drift_detected': self._check_drift_thresholds(metrics),
            'created_at': datetime.now().isoformat()
        }
        
        try:
            self.db.client.table('drift_metrics').insert(data).execute()
            self.logger.info("Drift metrics stored successfully")
        except Exception as e:
            self.logger.error(f"Error storing drift metrics: {e}")
    
    def _check_drift_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check if any drift threshold is breached"""
        kl_breach = metrics['kl_divergence'] > self.kl_warning
        cosine_breach = metrics['cosine_semantic'] < self.cosine_warning
        
        return kl_breach or cosine_breach


if __name__ == "__main__":
    detector = DriftDetector()

    detector.run()
