"""
Drift Detection Core Module
Calculates drift metrics between baseline and test batches
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    def __init__(self, baseline_stats: Dict):
        """
        Initialize drift detector with baseline statistics
        
        Args:
            baseline_stats: Dict containing baseline statistics from database
        """
        self.baseline_stats = baseline_stats
        self.baseline_sentiment = baseline_stats['sentiment_distribution']
        self.baseline_semantic = np.array(baseline_stats['mean_semantic_embedding'])
        self.baseline_sentiment_vector = np.array(baseline_stats['mean_sentiment_vector'])
        
        # Thresholds
        self.kl_threshold = 0.1
        self.cosine_semantic_threshold = 0.95
        self.cosine_sentiment_threshold = 0.95
        
        logger.info("✅ Drift detector initialized")
    
    def calculate_kl_divergence(self, test_sentiment_dist: Dict) -> float:
        """
        Calculate KL divergence between baseline and test sentiment distributions
        
        Args:
            test_sentiment_dist: Dict with sentiment distribution from test batch
            
        Returns:
            KL divergence score
        """
        # Extract probabilities
        baseline_probs = np.array([
            self.baseline_sentiment['positive']['proportion'],
            self.baseline_sentiment['neutral']['proportion'],
            self.baseline_sentiment['negative']['proportion']
        ])
        
        test_probs = np.array([
            test_sentiment_dist['positive']['proportion'],
            test_sentiment_dist['neutral']['proportion'],
            test_sentiment_dist['negative']['proportion']
        ])
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        baseline_probs = baseline_probs + epsilon
        test_probs = test_probs + epsilon
        
        # Normalize
        baseline_probs = baseline_probs / baseline_probs.sum()
        test_probs = test_probs / test_probs.sum()
        
        # Calculate KL divergence
        kl_div = entropy(test_probs, baseline_probs)
        
        return float(kl_div)
    
    def calculate_cosine_similarity(self, test_embedding: np.ndarray, 
                                   baseline_embedding: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            test_embedding: Test embedding vector
            baseline_embedding: Baseline embedding vector
            
        Returns:
            Cosine similarity score (0-1, higher is more similar)
        """
        # Cosine distance = 1 - cosine similarity
        cos_dist = cosine(test_embedding, baseline_embedding)
        cos_sim = 1 - cos_dist
        
        return float(cos_sim)
    
    def detect_drift(self, test_batch: Dict) -> Dict:
        """
        Perform complete drift detection on test batch
        
        Args:
            test_batch: Dict containing test batch metrics
                - sentiment_distribution: Dict
                - mean_semantic_embedding: np.ndarray
                - mean_sentiment_vector: np.ndarray
                - article_count: int
                
        Returns:
            Dict with drift metrics and detection result
        """
        logger.info(f"🔍 Analyzing batch of {test_batch['article_count']} articles...")
        
        # Calculate KL divergence
        kl_div = self.calculate_kl_divergence(test_batch['sentiment_distribution'])
        
        # Calculate semantic similarity
        cos_semantic = self.calculate_cosine_similarity(
            test_batch['mean_semantic_embedding'],
            self.baseline_semantic
        )
        
        # Calculate sentiment similarity
        cos_sentiment = self.calculate_cosine_similarity(
            test_batch['mean_sentiment_vector'],
            self.baseline_sentiment_vector
        )
        
        # Determine if drift detected
        drift_detected = (
            kl_div > self.kl_threshold or
            cos_semantic < self.cosine_semantic_threshold or
            cos_sentiment < self.cosine_sentiment_threshold
        )
        
        # Determine drift severity
        if drift_detected:
            if kl_div > 0.3 or cos_semantic < 0.85 or cos_sentiment < 0.85:
                severity = "STRONG"
            else:
                severity = "MILD"
        else:
            severity = "NONE"
        
        results = {
            'kl_divergence': kl_div,
            'cosine_similarity_semantic': cos_semantic,
            'cosine_similarity_sentiment': cos_sentiment,
            'drift_detected': drift_detected,
            'severity': severity,
            'test_article_count': test_batch['article_count']
        }
        
        # Log results
        if drift_detected:
            logger.warning(f"⚠️ DRIFT DETECTED ({severity}):")
            logger.warning(f"   KL Divergence: {kl_div:.4f} (threshold: {self.kl_threshold})")
            logger.warning(f"   Cosine Semantic: {cos_semantic:.4f} (threshold: {self.cosine_semantic_threshold})")
            logger.warning(f"   Cosine Sentiment: {cos_sentiment:.4f} (threshold: {self.cosine_sentiment_threshold})")
        else:
            logger.info(f"✅ No drift detected:")
            logger.info(f"   KL Divergence: {kl_div:.4f}")
            logger.info(f"   Cosine Semantic: {cos_semantic:.4f}")
            logger.info(f"   Cosine Sentiment: {cos_sentiment:.4f}")
        
        return results


def calculate_batch_metrics(articles: List[Dict]) -> Dict:
    """
    Calculate aggregate metrics for a batch of articles
    
    Args:
        articles: List of article dicts with sentiment and embedding data
        
    Returns:
        Dict with batch metrics
    """
    # Sentiment distribution
    sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    sentiment_scores = {'positive': [], 'neutral': [], 'negative': []}
    
    semantic_embeddings = []
    sentiment_vectors = []
    
    for article in articles:
        # Count sentiments
        label = article['sentiment_label']
        sentiment_counts[label] += 1
        
        # Collect scores
        sentiment_scores['positive'].append(article['positive_score'])
        sentiment_scores['neutral'].append(article['neutral_score'])
        sentiment_scores['negative'].append(article['negative_score'])
        
        # Collect embeddings
        if article.get('semantic_embedding'):
            semantic_embeddings.append(article['semantic_embedding'])
        
        if article.get('sentiment_vector'):
            sentiment_vectors.append(article['sentiment_vector'])
    
    total = len(articles)
    
    # Calculate sentiment distribution
    sentiment_dist = {
        'positive': {
            'count': sentiment_counts['positive'],
            'proportion': sentiment_counts['positive'] / total,
            'mean_score': np.mean(sentiment_scores['positive'])
        },
        'neutral': {
            'count': sentiment_counts['neutral'],
            'proportion': sentiment_counts['neutral'] / total,
            'mean_score': np.mean(sentiment_scores['neutral'])
        },
        'negative': {
            'count': sentiment_counts['negative'],
            'proportion': sentiment_counts['negative'] / total,
            'mean_score': np.mean(sentiment_scores['negative'])
        }
    }
    
    # Calculate mean embeddings
    mean_semantic = np.mean(semantic_embeddings, axis=0) if semantic_embeddings else None
    mean_sentiment_vec = np.mean(sentiment_vectors, axis=0) if sentiment_vectors else None
    
    return {
        'sentiment_distribution': sentiment_dist,
        'mean_semantic_embedding': mean_semantic,
        'mean_sentiment_vector': mean_sentiment_vec,
        'article_count': total
    }