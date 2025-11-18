"""
New Article Processor
Fetches new articles daily, processes them, and stores results
"""

import sys
import os
from datetime import datetime
from typing import List, Dict, Any
import uuid

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.supabase_client import SupabaseClient
from models.sentiment_analyzer import SentimentAnalyzer
from models.embedding_generator import EmbeddingGenerator
from utils.rss_fetcher import RSSFetcher
from utils.logger import setup_logger


class NewArticleProcessor:
    """Process new articles from RSS feed"""
    
    def __init__(self):
        """Initialize processor"""
        self.db = SupabaseClient()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.embedding_generator = EmbeddingGenerator()
        self.rss_fetcher = RSSFetcher()
        self.logger = setup_logger()
    
    def run(self) -> Dict[str, Any]:
        """
        Main execution method
        
        Returns:
            Dictionary with processing results
        """
        self.logger.info("Starting new article processing...")
        
        # Fetch latest articles from RSS
        rss_articles = self.rss_fetcher.fetch_latest_articles(max_articles=50)
        
        if not rss_articles:
            self.logger.warning("No articles fetched from RSS feed")
            return {"new_articles": 0, "processed": 0}
        
        # Filter out duplicates (check against existing articles)
        new_articles = self._filter_duplicates(rss_articles)
        
        if not new_articles:
            self.logger.info("No new articles found (all duplicates)")
            return {"new_articles": 0, "processed": 0}
        
        self.logger.info(f"Found {len(new_articles)} new articles")
        
        # Scrape full content for new articles
        new_articles_with_content = self._scrape_full_content(new_articles)
        
        # Insert new articles into news_cleaned
        inserted_articles = self._insert_articles(new_articles_with_content)
        
        if not inserted_articles:
            self.logger.error("Failed to insert new articles")
            return {"new_articles": len(new_articles), "processed": 0}
        
        # Process sentiment and embeddings
        sentiment_results = self.sentiment_analyzer.analyze_batch(inserted_articles)
        
        # Insert sentiment scores
        sentiment_data = self._prepare_sentiment_data(sentiment_results)
        self.db.insert_sentiment_scores(sentiment_data)
        
        # Generate and insert embeddings
        texts = [article['content_full'] for article in inserted_articles]
        semantic_embeddings = self.embedding_generator.generate_semantic_embeddings(texts)
        
        embedding_data = self._prepare_embedding_data(sentiment_results, semantic_embeddings)
        self.db.insert_embeddings(embedding_data)
        
        self.logger.info(f"Successfully processed {len(inserted_articles)} new articles")
        
        return {
            "new_articles": len(new_articles),
            "processed": len(inserted_articles),
            "article_ids": [a['id'] for a in inserted_articles]
        }
    
    def _filter_duplicates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out articles that already exist in database"""
        new_articles = []
        
        for article in articles:
            # Check by link or guid
            link = article.get('link', '')
            guid = article.get('guid', '')
            
            # Query database
            existing = self.db.client.table('news_cleaned').select('id').or_(
                f"link.eq.{link},guid.eq.{guid}"
            ).execute()
            
            if not existing.data:
                new_articles.append(article)
        
        self.logger.info(f"Filtered {len(articles) - len(new_articles)} duplicates")
        return new_articles
    
    def _scrape_full_content(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scrape full content for articles"""
        self.logger.info(f"Scraping full content for {len(articles)} articles...")
        
        for article in articles:
            url = article.get('link', '')
            if url:
                content = self.rss_fetcher.scrape_full_content(url)
                article['content_full'] = content if content else article.get('description', '')
            else:
                article['content_full'] = article.get('description', '')
        
        return articles
    
    def _insert_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Insert new articles into news_cleaned table"""
        self.logger.info(f"Inserting {len(articles)} new articles...")
        
        articles_to_insert = []
        
        for article in articles:
            article_data = {
                'id': str(uuid.uuid4()),
                'title': article.get('title', ''),
                'link': article.get('link', ''),
                'category': article.get('category', ''),
                'source': article.get('source', ''),
                'pub_date': article.get('pub_date'),
                'scraped_at': datetime.now().isoformat(),
                'description': article.get('description', ''),
                'content_full': article.get('content_full', ''),
                'guid': article.get('guid', ''),
                'image_url': article.get('image_url', ''),
                'has_description': bool(article.get('description')),
                'has_image': bool(article.get('image_url')),
                'needs_full_scrape': False,
                'raw_json': article.get('raw_json', ''),
                'is_baseline': False,  # New articles are NOT baseline
                'processed_date': datetime.now().isoformat()
            }
            
            articles_to_insert.append(article_data)
        
        try:
            response = self.db.client.table('news_cleaned').insert(articles_to_insert).execute()
            self.logger.info(f"Inserted {len(articles_to_insert)} articles successfully")
            return articles_to_insert
        except Exception as e:
            self.logger.error(f"Error inserting articles: {e}")
            return []
    
    def _prepare_sentiment_data(self, sentiment_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare sentiment data for database insertion"""
        sentiment_data = []
        
        for result in sentiment_results:
            sentiment_data.append({
                'article_id': result['article_id'],
                'sentiment_label': result['sentiment_label'].lower(),
                'positive_score': result['sentiment_scores']['positive'],
                'neutral_score': result['sentiment_scores']['neutral'],
                'negative_score': result['sentiment_scores']['negative'],
                'is_baseline': False,
                'processed_at': datetime.now().isoformat()
            })
        
        return sentiment_data
    
    def _prepare_embedding_data(
        self,
        sentiment_results: List[Dict[str, Any]],
        semantic_embeddings
    ) -> List[Dict[str, Any]]:
        """Prepare embedding data for database insertion"""
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
                'is_baseline': False,
                'created_at': datetime.now().isoformat()
            })
        
        return embedding_data


if __name__ == "__main__":
    processor = NewArticleProcessor()
    results = processor.run()
    print(f"\n Processing complete: {results}")