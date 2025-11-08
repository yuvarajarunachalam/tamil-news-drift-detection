"""
Baseline Generator
Processes all baseline articles and generates sentiment scores, embeddings, and statistics
"""

import sys
import os
from datetime import datetime
import time
import numpy as np
from typing import List, Dict, Any

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))  # pipelines/
src_dir = os.path.dirname(current_dir)  # src/
sys.path.insert(0, src_dir)

from database.supabase_client import SupabaseClient
from models.sentiment_analyzer import SentimentAnalyzer
from models.embedding_generator import EmbeddingGenerator
from monitoring.telegram_notifier import TelegramNotifier
from utils.logger import setup_logger


class BaselineGenerator:
    def __init__(self, batch_size: int = 50):
        """
        Initialize baseline generator
        
        Args:
            batch_size: Number of articles to process per batch
        """
        self.batch_size = batch_size
        self.db = SupabaseClient()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.embedding_generator = EmbeddingGenerator()
        self.telegram = TelegramNotifier()
        self.logger = setup_logger()
        
    def run(self):
        """Main execution method"""
        start_time = time.time()
        
        try:
            # Get total article count
            total_articles = self.db.get_baseline_count()
            total_batches = (total_articles + self.batch_size - 1) // self.batch_size
            
            self.logger.info(f"Starting baseline generation: {total_articles} articles, {total_batches} batches")
            self.telegram.processing_started(total_articles, total_batches)
            
            # Get completed batches (for resume capability)
            completed_batches = self.db.get_completed_batches()
            self.logger.info(f"Found {len(completed_batches)} already completed batches")
            
            # Process batches
            for batch_num in range(1, total_batches + 1):
                # Skip already completed batches
                if batch_num in completed_batches:
                    self.logger.info(f"Skipping batch {batch_num}/{total_batches} (already completed)")
                    continue
                
                try:
                    self._process_batch(batch_num, total_batches)
                except Exception as e:
                    error_msg = str(e)
                    self.logger.error(f"Batch {batch_num} failed: {error_msg}")
                    self.telegram.batch_failed(batch_num, total_batches, error_msg)
                    self.db.log_batch_failed(batch_num, error_msg)
                    # Continue with next batch instead of stopping
                    continue
            
            # Calculate and store baseline statistics
            self.logger.info("Calculating baseline statistics...")
            self._calculate_baseline_statistics()
            
            # Send completion notification
            elapsed_time = self._format_elapsed_time(time.time() - start_time)
            self.logger.info(f"Baseline generation complete! Time: {elapsed_time}")
            self.telegram.processing_complete(total_articles, elapsed_time)
            
        except Exception as e:
            error_msg = f"Critical error in baseline generation: {str(e)}"
            self.logger.error(error_msg)
            self.telegram.error_alert(error_msg)
            raise
    
    def _process_batch(self, batch_num: int, total_batches: int):
        """
        Process a single batch of articles
        
        Args:
            batch_num: Current batch number (1-indexed)
            total_batches: Total number of batches
        """
        offset = (batch_num - 1) * self.batch_size
        batch_start = offset
        batch_end = offset + self.batch_size
        
        self.logger.info(f"Processing batch {batch_num}/{total_batches} (articles {batch_start}-{batch_end})")
        
        # Log batch start (ignore duplicate key errors)
        try:
            self.db.log_batch_start(batch_num, batch_start, batch_end)
        except Exception as e:
            self.logger.warning(f"Could not log batch start (may already exist): {e}")
        
        # Fetch articles
        articles = self.db.get_baseline_articles(limit=self.batch_size, offset=offset)
        
        if not articles or len(articles) == 0:
            self.logger.warning(f"No articles found for batch {batch_num}")
            return
        
        # DEBUG: Log the type and structure of articles
        self.logger.info(f"DEBUG: articles type: {type(articles)}")
        self.logger.info(f"DEBUG: articles length: {len(articles)}")
        if articles:
            self.logger.info(f"DEBUG: first article type: {type(articles[0])}")
            if isinstance(articles[0], dict):
                self.logger.info(f"DEBUG: first article keys: {list(articles[0].keys())}")
                self.logger.info(f"DEBUG: first article id: {articles[0].get('id', 'NO ID')}")
                self.logger.info(f"DEBUG: first article has content_full: {'content_full' in articles[0]}")
            else:
                self.logger.info(f"DEBUG: first article is NOT a dict, it's: {str(articles[0])[:200]}")
        
        # Generate sentiment scores - pass articles directly
        self.logger.info(f"Generating sentiment scores for {len(articles)} articles...")
        
        try:
            sentiment_results = self.sentiment_analyzer.analyze_batch(articles)
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
        
        if not sentiment_results:
            self.logger.error(f"No sentiment results returned for batch {batch_num}")
            return
        
        # Prepare sentiment data for insertion
        sentiment_data = []
        for result in sentiment_results:
            sentiment_data.append({
                'article_id': result['article_id'],
                'sentiment_label': result['sentiment_label'].lower(),  # Convert POSITIVE -> positive
                'positive_score': result['sentiment_scores']['positive'],
                'neutral_score': result['sentiment_scores']['neutral'],
                'negative_score': result['sentiment_scores']['negative'],
                'is_baseline': True,
                'processed_at': datetime.now().isoformat()
            })
        
        # Insert sentiment scores
        self.logger.info(f"Inserting {len(sentiment_data)} sentiment scores...")
        self.db.insert_sentiment_scores(sentiment_data)
        
        # Generate embeddings - extract texts for embedding generator
        self.logger.info(f"Generating embeddings for {len(articles)} articles...")
        texts = [article['content_full'] for article in articles]
        semantic_embeddings = self.embedding_generator.generate_semantic_embeddings(texts)
        
        # Prepare embedding data for insertion
        embedding_data = []
        for i, result in enumerate(sentiment_results):
            sentiment_vector = [
                result['sentiment_scores']['positive'],
                result['sentiment_scores']['neutral'],
                result['sentiment_scores']['negative']
            ]
            
            embedding_data.append({
                'article_id': result['article_id'],
                'semantic_embedding': semantic_embeddings[i].tolist(),
                'sentiment_vector': sentiment_vector,
                'is_baseline': True,
                'created_at': datetime.now().isoformat()
            })
        
        # Insert embeddings
        self.logger.info(f"Inserting {len(embedding_data)} embeddings...")
        self.db.insert_embeddings(embedding_data)
        
        # Update processed_date in news_cleaned
        article_ids = [result['article_id'] for result in sentiment_results]
        self.db.update_processed_date(article_ids, datetime.now().isoformat())
        
        # Log batch completion
        self.db.log_batch_complete(batch_num)
        self.logger.info(f"Batch {batch_num}/{total_batches} completed successfully")
        self.telegram.batch_complete(batch_num, total_batches)
    
    def _calculate_baseline_statistics(self):
        """Calculate and store baseline statistics"""
        self.logger.info("Fetching all sentiment scores and embeddings...")
        
        # Fetch all baseline data
        sentiment_scores = self.db.get_all_sentiment_scores(is_baseline=True)
        embeddings = self.db.get_all_embeddings(is_baseline=True)
        
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
        
        # Calculate mean embeddings
        semantic_embeddings = np.array([e['semantic_embedding'] for e in embeddings])
        sentiment_vectors = np.array([e['sentiment_vector'] for e in embeddings])
        
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
        self.logger.info("Storing baseline statistics...")
        self.db.insert_baseline_statistics(baseline_stats)
        
        self.logger.info(f"Baseline statistics stored successfully:")
        self.logger.info(f"  - Total articles: {total}")
        self.logger.info(f"  - Sentiment distribution: {sentiment_distribution}")
        self.logger.info(f"  - Std deviations: {std_sentiment_scores}")
    
    def _format_elapsed_time(self, seconds: float) -> str:
        """Format elapsed time in hours and minutes"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"


if __name__ == "__main__":
    generator = BaselineGenerator(batch_size=50)
    generator.run()