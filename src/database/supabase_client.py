"""
Supabase Database Client
Manages connections and queries to Supabase PostgreSQL database
"""

import os
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
import json


class SupabaseClient:
    def __init__(self):
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        self.client: Client = create_client(url, key)
    
    # ==================== BASELINE QUERIES ====================
    
    def get_baseline_articles(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Fetch baseline articles from news_cleaned
        
        Args:
            limit: Number of articles to fetch (None for all)
            offset: Starting index for pagination
        
        Returns:
            List of article dictionaries
        """
        query = self.client.table('news_cleaned').select('*').eq('is_baseline', True).order('id')
        
        if limit:
            query = query.limit(limit)
        
        if offset > 0:
            query = query.range(offset, offset + limit - 1)
        
        response = query.execute()
        return response.data
    
    def get_baseline_count(self) -> int:
        """Get total count of baseline articles"""
        response = self.client.table('news_cleaned').select('id', count='exact').eq('is_baseline', True).execute()
        return response.count
    
    def insert_sentiment_scores(self, data: List[Dict[str, Any]]) -> bool:
        """
        Insert sentiment scores into model_output table
        
        Args:
            data: List of sentiment score dictionaries
        
        Returns:
            True if successful
        """
        try:
            response = self.client.table('model_output').insert(data).execute()
            return True
        except Exception as e:
            print(f"Error inserting sentiment scores: {e}")
            return False
    
    def insert_embeddings(self, data: List[Dict[str, Any]]) -> bool:
        """
        Insert embeddings into article_embeddings table
        
        Args:
            data: List of embedding dictionaries
        
        Returns:
            True if successful
        """
        try:
            response = self.client.table('article_embeddings').insert(data).execute()
            return True
        except Exception as e:
            print(f"Error inserting embeddings: {e}")
            return False
    
    def update_processed_date(self, article_ids: List[str], processed_date: str) -> bool:
        """
        Update processed_date for articles in news_cleaned
        
        Args:
            article_ids: List of article UUIDs
            processed_date: ISO format timestamp
        
        Returns:
            True if successful
        """
        try:
            for article_id in article_ids:
                self.client.table('news_cleaned').update({
                    'processed_date': processed_date
                }).eq('id', article_id).execute()
            return True
        except Exception as e:
            print(f"Error updating processed_date: {e}")
            return False
    
    def insert_baseline_statistics(self, stats: Dict[str, Any]) -> bool:
        """
        Insert baseline statistics into baseline_statistics table
        
        Args:
            stats: Dictionary containing baseline metrics
        
        Returns:
            True if successful
        """
        try:
            response = self.client.table('baseline_statistics').insert(stats).execute()
            return True
        except Exception as e:
            print(f"Error inserting baseline statistics: {e}")
            return False
    
    def get_all_sentiment_scores(self, is_baseline: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch all sentiment scores for baseline or test articles
        
        Args:
            is_baseline: True for baseline, False for test articles
        
        Returns:
            List of sentiment score dictionaries
        """
        response = self.client.table('model_output').select('*').eq('is_baseline', is_baseline).execute()
        return response.data
    
    def get_all_embeddings(self, is_baseline: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch all embeddings for baseline or test articles
        
        Args:
            is_baseline: True for baseline, False for test articles
        
        Returns:
            List of embedding dictionaries
        """
        response = self.client.table('article_embeddings').select('*').eq('is_baseline', is_baseline).execute()
        return response.data
    
    # ==================== PROCESSING LOG QUERIES ====================
    
    def get_completed_batches(self) -> List[int]:
        """
        Get list of completed batch numbers
        
        Returns:
            List of batch numbers that have been successfully processed
        """
        response = self.client.table('processing_log').select('batch_number').eq('status', 'completed').execute()
        return [item['batch_number'] for item in response.data]
    
    def log_batch_start(self, batch_num: int, batch_start: int, batch_end: int) -> bool:
        """Log batch processing start"""
        try:
            data = {
                'batch_number': batch_num,
                'batch_start': batch_start,
                'batch_end': batch_end,
                'status': 'in_progress'
            }
            self.client.table('processing_log').insert(data).execute()
            return True
        except Exception as e:
            print(f"Error logging batch start: {e}")
            return False
    
    def log_batch_complete(self, batch_num: int) -> bool:
        """Log batch processing completion"""
        try:
            self.client.table('processing_log').update({
                'status': 'completed'
            }).eq('batch_number', batch_num).execute()
            return True
        except Exception as e:
            print(f"Error logging batch completion: {e}")
            return False
    
    def log_batch_failed(self, batch_num: int, error_message: str) -> bool:
        """Log batch processing failure"""
        try:
            self.client.table('processing_log').update({
                'status': 'failed',
                'error_message': error_message
            }).eq('batch_number', batch_num).execute()
            return True
        except Exception as e:
            print(f"Error logging batch failure: {e}")
            return False