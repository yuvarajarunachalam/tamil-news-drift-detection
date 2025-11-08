"""
Drift Calculator
Calculates drift metrics between baseline and test sets
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
from scipy.spatial.distance import cosine
from scipy.special import kl_div


class DriftCalculator:
    """Calculates various drift metrics"""
    
    def __init__(
        self,
        kl_threshold: float = 0.05,
        cosine_threshold: float = 0.85
    ):
        """
        Initialize drift calculator
        
        Args:
            kl_threshold: Threshold for KL divergence (higher = more drift)
            cosine_threshold: Threshold for cosine similarity (lower = more drift)
        """
        self.kl_threshold = kl_threshold
        self.cosine_threshold = cosine_threshold
    
    # ========================================
    # SENTIMENT DISTRIBUTION DRIFT (KL DIVERGENCE)
    # ========================================
    
    def calculate_kl_divergence(
        self,
        baseline_outputs: List[Dict[str, Any]],
        test_outputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate KL divergence between baseline and test sentiment distributions
        
        Args:
            baseline_outputs: List of sentiment analysis results for baseline
            test_outputs: List of sentiment analysis results for test
        
        Returns:
            Dictionary with KL divergence metric
        """
        print("\n" + "="*60)
        print("📊 CALCULATING KL DIVERGENCE (Sentiment Distribution)")
        print("="*60)
        
        # Get distributions
        baseline_dist = self._get_sentiment_distribution(baseline_outputs, "Baseline")
        test_dist = self._get_sentiment_distribution(test_outputs, "Test")
        
        # Calculate KL divergence
        # KL(P || Q) where P is test (new) and Q is baseline (reference)
        baseline_probs = np.array([baseline_dist[label] for label in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']])
        test_probs = np.array([test_dist[label] for label in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']])
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        baseline_probs = baseline_probs + epsilon
        test_probs = test_probs + epsilon
        
        # Normalize after adding epsilon
        baseline_probs = baseline_probs / baseline_probs.sum()
        test_probs = test_probs / test_probs.sum()
        
        # Calculate KL divergence
        kl_divergence = np.sum(kl_div(test_probs, baseline_probs))
        
        # Detect drift
        drift_detected = kl_divergence > self.kl_threshold
        
        print(f"\n📈 KL Divergence: {kl_divergence:.6f}")
        print(f"🎯 Threshold: {self.kl_threshold}")
        print(f"{'🚨 DRIFT DETECTED!' if drift_detected else '✅ No significant drift'}")
        
        # Calculate percentage shifts
        shifts = self._calculate_percentage_shifts(baseline_dist, test_dist)
        
        return {
            "metric_type": "kl_divergence",
            "metric_value": float(kl_divergence),
            "drift_detected": drift_detected,
            "threshold_used": self.kl_threshold,
            "baseline_distribution": baseline_dist,
            "comparison_distribution": test_dist,
            "details": {
                "percentage_shifts": shifts,
                "interpretation": self._interpret_kl_divergence(kl_divergence)
            }
        }
    
    def _get_sentiment_distribution(
        self,
        outputs: List[Dict[str, Any]],
        label: str
    ) -> Dict[str, float]:
        """Calculate sentiment label distribution"""
        total = len(outputs)
        counts = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
        
        for output in outputs:
            sentiment = output["sentiment_label"]
            counts[sentiment] += 1
        
        # Convert to probabilities
        distribution = {
            label: counts[label] / total
            for label in counts
        }
        
        # Print distribution
        print(f"\n{label} Distribution:")
        for sentiment_label in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            count = counts[sentiment_label]
            prob = distribution[sentiment_label]
            print(f"  {sentiment_label}: {count} ({prob*100:.1f}%)")
        
        return distribution
    
    def _calculate_percentage_shifts(
        self,
        baseline_dist: Dict[str, float],
        test_dist: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate percentage shift for each sentiment"""
        shifts = {}
        
        print(f"\n📊 Percentage Shifts:")
        for label in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            baseline_val = baseline_dist[label]
            test_val = test_dist[label]
            
            if baseline_val > 0:
                shift = ((test_val - baseline_val) / baseline_val) * 100
            else:
                shift = 0.0
            
            shifts[label] = shift
            
            direction = "↑" if shift > 0 else "↓" if shift < 0 else "→"
            print(f"  {label}: {shift:+.1f}% {direction}")
        
        return shifts
    
    def _interpret_kl_divergence(self, kl_value: float) -> str:
        """Provide interpretation of KL divergence value"""
        if kl_value < 0.01:
            return "Very similar distributions (negligible drift)"
        elif kl_value < 0.05:
            return "Similar distributions (no significant drift)"
        elif kl_value < 0.1:
            return "Moderate difference (drift detected)"
        else:
            return "Large difference (significant drift)"
    
    # ========================================
    # SEMANTIC DRIFT (COSINE SIMILARITY)
    # ========================================
    
    def calculate_semantic_drift(
        self,
        baseline_embeddings: List[Dict[str, Any]],
        test_embeddings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate cosine similarity between average semantic embeddings
        
        Args:
            baseline_embeddings: List of embedding dictionaries for baseline
            test_embeddings: List of embedding dictionaries for test
        
        Returns:
            Dictionary with cosine similarity metric
        """
        print("\n" + "="*60)
        print("🧠 CALCULATING SEMANTIC DRIFT (Cosine Similarity)")
        print("="*60)
        
        # Extract semantic embeddings
        baseline_semantic = [e["semantic_embedding"] for e in baseline_embeddings]
        test_semantic = [e["semantic_embedding"] for e in test_embeddings]
        
        # Calculate average embeddings
        baseline_avg = np.mean(baseline_semantic, axis=0)
        test_avg = np.mean(test_semantic, axis=0)
        
        # Calculate cosine similarity (1 - cosine distance)
        cosine_distance = cosine(baseline_avg, test_avg)
        cosine_similarity = 1 - cosine_distance
        
        # Detect drift (low similarity = drift)
        drift_detected = cosine_similarity < self.cosine_threshold
        
        print(f"\n📈 Cosine Similarity: {cosine_similarity:.6f}")
        print(f"🎯 Threshold: {self.cosine_threshold}")
        print(f"{'🚨 DRIFT DETECTED! (Topics shifted)' if drift_detected else '✅ Topics remain similar'}")
        
        return {
            "metric_type": "cosine_similarity_semantic",
            "metric_value": float(cosine_similarity),
            "drift_detected": drift_detected,
            "threshold_used": self.cosine_threshold,
            "baseline_distribution": {
                "average_embedding_norm": float(np.linalg.norm(baseline_avg)),
                "num_articles": len(baseline_semantic)
            },
            "comparison_distribution": {
                "average_embedding_norm": float(np.linalg.norm(test_avg)),
                "num_articles": len(test_semantic)
            },
            "details": {
                "interpretation": self._interpret_cosine_similarity(cosine_similarity)
            }
        }
    
    # ========================================
    # SENTIMENT VECTOR DRIFT (COSINE SIMILARITY)
    # ========================================
    
    def calculate_sentiment_vector_drift(
        self,
        baseline_embeddings: List[Dict[str, Any]],
        test_embeddings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate cosine similarity between average sentiment vectors
        
        Args:
            baseline_embeddings: List of embedding dictionaries for baseline
            test_embeddings: List of embedding dictionaries for test
        
        Returns:
            Dictionary with cosine similarity metric
        """
        print("\n" + "="*60)
        print("💭 CALCULATING SENTIMENT PATTERN DRIFT (Sentiment Vector Similarity)")
        print("="*60)
        
        # Extract sentiment vectors
        baseline_vectors = [e["sentiment_vector"] for e in baseline_embeddings]
        test_vectors = [e["sentiment_vector"] for e in test_embeddings]
        
        # Calculate average vectors
        baseline_avg = np.mean(baseline_vectors, axis=0)
        test_avg = np.mean(test_vectors, axis=0)
        
        print(f"\nBaseline avg sentiment vector: {baseline_avg}")
        print(f"Test avg sentiment vector: {test_avg}")
        
        # Calculate cosine similarity
        cosine_distance = cosine(baseline_avg, test_avg)
        cosine_similarity = 1 - cosine_distance
        
        # Detect drift
        drift_detected = cosine_similarity < self.cosine_threshold
        
        print(f"\n📈 Cosine Similarity: {cosine_similarity:.6f}")
        print(f"🎯 Threshold: {self.cosine_threshold}")
        print(f"{'🚨 DRIFT DETECTED! (Sentiment patterns changed)' if drift_detected else '✅ Sentiment patterns similar'}")
        
        return {
            "metric_type": "cosine_similarity_sentiment",
            "metric_value": float(cosine_similarity),
            "drift_detected": drift_detected,
            "threshold_used": self.cosine_threshold,
            "baseline_distribution": {
                "average_sentiment_vector": baseline_avg.tolist(),
                "num_articles": len(baseline_vectors)
            },
            "comparison_distribution": {
                "average_sentiment_vector": test_avg.tolist(),
                "num_articles": len(test_vectors)
            },
            "details": {
                "interpretation": self._interpret_cosine_similarity(cosine_similarity)
            }
        }
    
    def _interpret_cosine_similarity(self, similarity: float) -> str:
        """Provide interpretation of cosine similarity value"""
        if similarity > 0.95:
            return "Very high similarity (almost identical)"
        elif similarity > 0.85:
            return "High similarity (no significant drift)"
        elif similarity > 0.70:
            return "Moderate similarity (some drift detected)"
        else:
            return "Low similarity (significant drift)"
    
    # ========================================
    # UNIFIED DRIFT CALCULATION
    # ========================================
    
    def calculate_all_drift_metrics(
        self,
        baseline_outputs: List[Dict[str, Any]],
        test_outputs: List[Dict[str, Any]],
        baseline_embeddings: List[Dict[str, Any]],
        test_embeddings: List[Dict[str, Any]],
        baseline_article_ids: List[str],
        test_article_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Calculate all drift metrics
        
        Args:
            baseline_outputs: Sentiment analysis results for baseline
            test_outputs: Sentiment analysis results for test
            baseline_embeddings: Embeddings for baseline
            test_embeddings: Embeddings for test
            baseline_article_ids: List of baseline article IDs
            test_article_ids: List of test article IDs
        
        Returns:
            List of drift metric dictionaries ready for database insertion
        """
        print("\n" + "="*60)
        print("🔍 COMPREHENSIVE DRIFT ANALYSIS")
        print("="*60)
        
        metrics = []
        
        # 1. KL Divergence (Sentiment Distribution)
        kl_metric = self.calculate_kl_divergence(baseline_outputs, test_outputs)
        kl_metric.update({
            "baseline_article_count": len(baseline_outputs),
            "comparison_article_count": len(test_outputs),
            "baseline_sample_ids": baseline_article_ids,
            "comparison_sample_ids": test_article_ids
        })
        metrics.append(kl_metric)
        
        # 2. Semantic Drift (Cosine Similarity on Embeddings)
        semantic_metric = self.calculate_semantic_drift(baseline_embeddings, test_embeddings)
        semantic_metric.update({
            "baseline_article_count": len(baseline_embeddings),
            "comparison_article_count": len(test_embeddings),
            "baseline_sample_ids": baseline_article_ids,
            "comparison_sample_ids": test_article_ids
        })
        metrics.append(semantic_metric)
        
        # 3. Sentiment Vector Drift (Cosine Similarity on Sentiment Vectors)
        sentiment_vector_metric = self.calculate_sentiment_vector_drift(
            baseline_embeddings,
            test_embeddings
        )
        sentiment_vector_metric.update({
            "baseline_article_count": len(baseline_embeddings),
            "comparison_article_count": len(test_embeddings),
            "baseline_sample_ids": baseline_article_ids,
            "comparison_sample_ids": test_article_ids
        })
        metrics.append(sentiment_vector_metric)
        
        # Summary
        print("\n" + "="*60)
        print("📋 DRIFT ANALYSIS SUMMARY")
        print("="*60)
        
        drift_count = sum(1 for m in metrics if m["drift_detected"])
        print(f"\n🎯 Drift detected in {drift_count}/3 metrics")
        
        for metric in metrics:
            status = "🚨 DRIFT" if metric["drift_detected"] else "✅ NO DRIFT"
            print(f"  {metric['metric_type']}: {metric['metric_value']:.6f} - {status}")
        
        return metrics


# Convenience function
def calculate_drift(
    baseline_outputs: List[Dict[str, Any]],
    test_outputs: List[Dict[str, Any]],
    baseline_embeddings: List[Dict[str, Any]],
    test_embeddings: List[Dict[str, Any]],
    baseline_article_ids: List[str],
    test_article_ids: List[str],
    kl_threshold: float = 0.05,
    cosine_threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Calculate all drift metrics
    
    Returns:
        List of drift metric dictionaries
    """
    calculator = DriftCalculator(kl_threshold, cosine_threshold)
    return calculator.calculate_all_drift_metrics(
        baseline_outputs,
        test_outputs,
        baseline_embeddings,
        test_embeddings,
        baseline_article_ids,
        test_article_ids
    )

