"""
Embedding Generator
Creates semantic embeddings and sentiment vectors for drift detection
"""

import time
import hashlib
from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingGenerator:
    """Generates embeddings for articles"""
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize embedding generator
        
        Args:
            embedding_model_name: Sentence transformer model name
        """
        self.embedding_model_name = embedding_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\n🧠 Loading embedding model: {embedding_model_name}")
        print(f"📱 Device: {self.device}")
        
        # Load sentence transformer model
        self.model = SentenceTransformer(embedding_model_name, device=self.device)
        
        print(f"✅ Embedding model loaded!")
        print(f"📊 Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_semantic_embedding(
        self,
        text: str,
        max_length: int = 2000
    ) -> List[float]:
        """
        Generate semantic embedding for text
        
        Args:
            text: Input text
            max_length: Maximum characters to process
        
        Returns:
            384-dimensional embedding vector
        """
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        return embedding.tolist()
    
    def generate_sentiment_vector(
        self,
        sentiment_scores: Dict[str, float]
    ) -> List[float]:
        """
        Convert sentiment scores to vector
        
        Args:
            sentiment_scores: Dictionary with 'positive', 'neutral', 'negative' scores
        
        Returns:
            3-dimensional sentiment vector [pos, neu, neg]
        """
        return [
            sentiment_scores.get("positive", 0.0),
            sentiment_scores.get("neutral", 0.0),
            sentiment_scores.get("negative", 0.0)
        ]
    
    def generate_content_hash(self, text: str) -> str:
        """
        Generate MD5 hash of content for change detection
        
        Args:
            text: Content to hash
        
        Returns:
            MD5 hash string
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def process_articles_with_sentiment(
        self,
        articles: List[Dict[str, Any]],
        model_outputs: List[Dict[str, Any]],
        batch_size: int = 50,
        max_length: int = 2000
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for articles that have sentiment analysis results
        
        Args:
            articles: List of article dictionaries
            model_outputs: List of sentiment analysis results (from sentiment_analyzer)
            batch_size: Number of articles to process at once
            max_length: Maximum characters per article
        
        Returns:
            List of embedding dictionaries ready for database insertion
        """
        # Create mapping of article_id to sentiment scores
        sentiment_map = {
            output["article_id"]: output["sentiment_scores"]
            for output in model_outputs
        }
        
        # Filter articles to only those with sentiment results
        articles_to_process = [
            article for article in articles
            if article["id"] in sentiment_map
        ]
        
        total_articles = len(articles_to_process)
        results = []
        
        print(f"\n🧠 Generating embeddings for {total_articles} articles...")
        print(f"📦 Batch size: {batch_size}")
        
        for i in range(0, total_articles, batch_size):
            batch = articles_to_process[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_articles + batch_size - 1) // batch_size
            
            print(f"  Batch {batch_num}/{total_batches} (articles {i+1}-{min(i+batch_size, total_articles)})...", end=" ")
            
            batch_start = time.time()
            
            # Process batch
            for article in batch:
                article_id = article["id"]
                content = article.get("content_full", "")
                
                if not content or content.strip() == "":
                    print(f"\n⚠️  Warning: Article {article_id} has no content, skipping...")
                    continue
                
                # Generate semantic embedding
                semantic_embedding = self.generate_semantic_embedding(content, max_length)
                
                # Generate sentiment vector
                sentiment_scores = sentiment_map[article_id]
                sentiment_vector = self.generate_sentiment_vector(sentiment_scores)
                
                # Generate content hash
                content_hash = self.generate_content_hash(content)
                
                # Create result dictionary
                result = {
                    "article_id": article_id,
                    "semantic_embedding": semantic_embedding,
                    "sentiment_vector": sentiment_vector,
                    "embedding_model": self.embedding_model_name,
                    "content_hash": content_hash
                }
                
                results.append(result)
            
            batch_time = time.time() - batch_start
            avg_time = batch_time / len(batch)
            
            print(f"✓ ({batch_time:.2f}s, {avg_time:.3f}s/article)")
        
        print(f"\n✅ Embedding generation complete!")
        print(f"📊 Successfully generated: {len(results)}/{total_articles} embeddings")
        
        return results
    
    def calculate_average_embeddings(
        self,
        embeddings_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate average semantic embedding and sentiment vector for a set of articles
        
        Args:
            embeddings_data: List of embedding dictionaries from database
        
        Returns:
            Dictionary with averaged embeddings
        """
        if not embeddings_data:
            raise ValueError("No embeddings provided")
        
        # Extract embeddings
        semantic_embeddings = []
        sentiment_vectors = []
        
        for item in embeddings_data:
            if "semantic_embedding" in item and item["semantic_embedding"]:
                semantic_embeddings.append(item["semantic_embedding"])
            
            if "sentiment_vector" in item and item["sentiment_vector"]:
                sentiment_vectors.append(item["sentiment_vector"])
        
        # Calculate averages
        avg_semantic = np.mean(semantic_embeddings, axis=0).tolist() if semantic_embeddings else None
        avg_sentiment = np.mean(sentiment_vectors, axis=0).tolist() if sentiment_vectors else None
        
        return {
            "average_semantic_embedding": avg_semantic,
            "average_sentiment_vector": avg_sentiment,
            "num_articles": len(embeddings_data)
        }


# Convenience function
def generate_embeddings(
    articles: List[Dict[str, Any]],
    model_outputs: List[Dict[str, Any]],
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 50
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for articles
    
    Args:
        articles: List of article dictionaries
        model_outputs: List of sentiment analysis results
        embedding_model_name: Sentence transformer model name
        batch_size: Batch size for processing
    
    Returns:
        List of embedding dictionaries
    """
    generator = EmbeddingGenerator(embedding_model_name)
    return generator.process_articles_with_sentiment(articles, model_outputs, batch_size)
