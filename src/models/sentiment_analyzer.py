"""
Sentiment Analysis Model
Wrapper for cardiffnlp/twitter-roberta-base-sentiment-latest
"""

import time
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


class SentimentAnalyzer:
    """Sentiment analysis using pre-trained RoBERTa model"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\n🤖 Loading sentiment model: {model_name}")
        print(f"📱 Device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Label mapping
        self.labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Labels: {self.labels}")
    
    def analyze_single(
        self,
        text: str,
        max_length: int = 2000
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Input text to analyze
            max_length: Maximum characters to analyze (truncate if longer)
        
        Returns:
            Dictionary with sentiment analysis results
        """
        # Truncate text if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        start_time = time.time()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # Model's max token limit
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits[0].cpu().numpy()
            scores = self._softmax(scores)
        
        # Get label and confidence
        predicted_idx = np.argmax(scores)
        predicted_label = self.labels[predicted_idx]
        confidence = float(scores[predicted_idx])
        
        # Create scores dict
        sentiment_scores = {
            "negative": float(scores[0]),
            "neutral": float(scores[1]),
            "positive": float(scores[2])
        }
        
        inference_time = int((time.time() - start_time) * 1000)  # Convert to ms
        
        return {
            "sentiment_label": predicted_label,
            "confidence": confidence,
            "sentiment_scores": sentiment_scores,
            "inference_time_ms": inference_time,
            "content_length": len(text)
        }
    
    def analyze_batch(
        self,
        articles: List[Dict[str, Any]],
        batch_size: int = 50,
        max_length: int = 2000
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of articles
        
        Args:
            articles: List of article dictionaries (must have 'content_full' and 'id' fields)
            batch_size: Number of articles to process at once
            max_length: Maximum characters per article
        
        Returns:
            List of analysis results
        """
        total_articles = len(articles)
        results = []
        
        print(f"\n🔄 Processing {total_articles} articles in batches of {batch_size}...")
        
        for i in range(0, total_articles, batch_size):
            batch = articles[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_articles + batch_size - 1) // batch_size
            
            print(f"  Batch {batch_num}/{total_batches} (articles {i+1}-{min(i+batch_size, total_articles)})...", end=" ")
            
            batch_start = time.time()
            
            for article in batch:
                # Get article content
                content = article.get("content_full", "")
                
                if not content or content.strip() == "":
                    print(f"\n⚠️  Warning: Article {article.get('id', 'unknown')} has no content, skipping...")
                    continue
                
                # Analyze sentiment
                analysis = self.analyze_single(content, max_length)
                
                # Add article_id and model version
                result = {
                    "article_id": article["id"],
                    "sentiment_label": analysis["sentiment_label"],
                    "confidence": analysis["confidence"],
                    "sentiment_scores": analysis["sentiment_scores"],
                    "model_version": self.model_name,
                    "content_length_analyzed": analysis["content_length"],
                    "model_inference_time_ms": analysis["inference_time_ms"]
                }
                
                results.append(result)
            
            batch_time = time.time() - batch_start
            avg_time = batch_time / len(batch) if len(batch) > 0 else 0
            
            print(f"✓ ({batch_time:.2f}s, {avg_time:.3f}s/article)")
        
        print(f"\n✅ Batch processing complete!")
        print(f"📊 Successfully processed: {len(results)}/{total_articles} articles")
        
        # Show sentiment distribution
        self._print_sentiment_distribution(results)
        
        return results
    
    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """Apply softmax to convert logits to probabilities"""
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()
    
    def _print_sentiment_distribution(self, results: List[Dict[str, Any]]):
        """Print sentiment distribution summary"""
        sentiment_counts = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
        total_confidence = 0
        
        for result in results:
            sentiment_counts[result["sentiment_label"]] += 1
            total_confidence += result["confidence"]
        
        total = len(results)
        avg_confidence = total_confidence / total if total > 0 else 0
        
        print(f"\n📊 Sentiment Distribution:")
        for label in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            count = sentiment_counts[label]
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        print(f"\n📈 Average Confidence: {avg_confidence:.3f}")


# Convenience function
def analyze_articles(
    articles: List[Dict[str, Any]],
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    batch_size: int = 50
) -> List[Dict[str, Any]]:
    """
    Analyze sentiment for a list of articles
    
    Args:
        articles: List of article dictionaries
        model_name: HuggingFace model identifier
        batch_size: Batch size for processing
    
    Returns:
        List of analysis results
    """
    analyzer = SentimentAnalyzer(model_name)
    return analyzer.analyze_batch(articles, batch_size)