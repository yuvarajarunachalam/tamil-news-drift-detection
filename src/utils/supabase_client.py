"""
Supabase Database Client
Manages connections and queries to Supabase PostgreSQL database
"""

import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv


class SupabaseClient:
    """Wrapper for Supabase database operations"""
    
    def __init__(self, env_path: str = "config/.env"):
        """
        Initialize Supabase client
        
        Args:
            env_path: Path to .env file containing credentials
        """
        load_dotenv(env_path)
        
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment variables")
        
        self.client: Client = create_client(self.url, self.key)
        print(f"✅ Connected to Supabase: {self.url[:30]}...")
    
    # ========================================
    # NEWS_CLEANED TABLE OPERATIONS
    # ========================================
    
    def get_random_articles(
        self, 
        limit: int, 
        exclude_ids: Optional[List[str]] = None,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get random articles from news_cleaned table
        
        Args:
            limit: Number of articles to retrieve
            exclude_ids: List of article IDs to exclude
            source_filter: Filter by source ('The Hindu - RSS' or 'The Hindu - Archive')
        
        Returns:
            List of article dictionaries
        """
        try:
            query = self.client.table("news_cleaned").select("*")
            
            # Apply source filter if specified
            if source_filter:
                query = query.eq("source", source_filter)
            
            # Exclude specific IDs if provided
            if exclude_ids:
                query = query.not_.in_("id", exclude_ids)
            
            # Get articles
            response = query.limit(limit).execute()
            
            articles = response.data
            
            # If we need more randomness, we can shuffle in Python
            import random
            random.shuffle(articles)
            
            print(f"✅ Retrieved {len(articles)} random articles")
            return articles[:limit]
            
        except Exception as e:
            print(f"❌ Error fetching random articles: {e}")
            raise
    
    def get_article_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Get single article by ID
        
        Args:
            article_id: UUID of article
        
        Returns:
            Article dictionary or None if not found
        """
        try:
            response = self.client.table("news_cleaned").select("*").eq("id", article_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"❌ Error fetching article {article_id}: {e}")
            return None
    
    def count_articles(self, source_filter: Optional[str] = None) -> int:
        """
        Count total articles in database
        
        Args:
            source_filter: Filter by source if specified
        
        Returns:
            Count of articles
        """
        try:
            query = self.client.table("news_cleaned").select("id", count="exact")
            
            if source_filter:
                query = query.eq("source", source_filter)
            
            response = query.execute()
            return response.count
            
        except Exception as e:
            print(f"❌ Error counting articles: {e}")
            return 0
    
    # ========================================
    # MODEL_OUTPUT TABLE OPERATIONS
    # ========================================
    
    def insert_model_output(self, output_data: Dict[str, Any]) -> bool:
        """
        Insert sentiment analysis result into model_output table
        
        Args:
            output_data: Dictionary containing model output fields
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.table("model_output").insert(output_data).execute()
            return True
        except Exception as e:
            print(f"❌ Error inserting model output: {e}")
            return False
    
    def batch_insert_model_outputs(self, outputs: List[Dict[str, Any]]) -> int:
        """
        Batch insert multiple model outputs
        
        Args:
            outputs: List of output dictionaries
        
        Returns:
            Number of successful insertions
        """
        try:
            self.client.table("model_output").insert(outputs).execute()
            print(f"✅ Inserted {len(outputs)} model outputs")
            return len(outputs)
        except Exception as e:
            print(f"❌ Error batch inserting model outputs: {e}")
            return 0
    
    def get_model_outputs_by_article_ids(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get model outputs for specific articles
        
        Args:
            article_ids: List of article UUIDs
        
        Returns:
            List of model output dictionaries
        """
        try:
            response = self.client.table("model_output").select("*").in_("article_id", article_ids).execute()
            return response.data
        except Exception as e:
            print(f"❌ Error fetching model outputs: {e}")
            return []
    
    def article_already_processed(self, article_id: str) -> bool:
        """
        Check if article already has model output
        
        Args:
            article_id: UUID of article
        
        Returns:
            True if already processed, False otherwise
        """
        try:
            response = self.client.table("model_output").select("id").eq("article_id", article_id).execute()
            return len(response.data) > 0
        except Exception as e:
            print(f"❌ Error checking if article processed: {e}")
            return False
    
    # ========================================
    # ARTICLE_EMBEDDINGS TABLE OPERATIONS
    # ========================================
    
    def insert_embedding(self, embedding_data: Dict[str, Any]) -> bool:
        """
        Insert article embedding into article_embeddings table
        
        Args:
            embedding_data: Dictionary containing embedding fields
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.table("article_embeddings").insert(embedding_data).execute()
            return True
        except Exception as e:
            print(f"❌ Error inserting embedding: {e}")
            return False
    
    def batch_insert_embeddings(self, embeddings: List[Dict[str, Any]]) -> int:
        """
        Batch insert multiple embeddings
        
        Args:
            embeddings: List of embedding dictionaries
        
        Returns:
            Number of successful insertions
        """
        try:
            self.client.table("article_embeddings").insert(embeddings).execute()
            print(f"✅ Inserted {len(embeddings)} embeddings")
            return len(embeddings)
        except Exception as e:
            print(f"❌ Error batch inserting embeddings: {e}")
            return 0
    
    def get_embeddings_by_article_ids(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get embeddings for specific articles
        
        Args:
            article_ids: List of article UUIDs
        
        Returns:
            List of embedding dictionaries
        """
        try:
            response = self.client.table("article_embeddings").select("*").in_("article_id", article_ids).execute()
            return response.data
        except Exception as e:
            print(f"❌ Error fetching embeddings: {e}")
            return []
    
    # ========================================
    # DRIFT_METRICS TABLE OPERATIONS
    # ========================================
    
    def insert_drift_metric(self, drift_data: Dict[str, Any]) -> bool:
        """
        Insert drift metric calculation result
        
        Args:
            drift_data: Dictionary containing drift metric fields
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.table("drift_metrics").insert(drift_data).execute()
            print(f"✅ Inserted drift metric: {drift_data.get('metric_type')}")
            return True
        except Exception as e:
            print(f"❌ Error inserting drift metric: {e}")
            return False
    
    def get_latest_drift_metrics(self) -> List[Dict[str, Any]]:
        """
        Get most recent drift metrics (all types)
        
        Returns:
            List of drift metric dictionaries
        """
        try:
            response = (
                self.client.table("drift_metrics")
                .select("*")
                .order("calculated_at", desc=True)
                .limit(10)
                .execute()
            )
            return response.data
        except Exception as e:
            print(f"❌ Error fetching latest drift metrics: {e}")
            return []


# Convenience function for getting a client instance
def get_client(env_path: str = "config/.env") -> SupabaseClient:
    """
    Get a Supabase client instance
    
    Args:
        env_path: Path to .env file
    
    Returns:
        SupabaseClient instance
    """
    return SupabaseClient(env_path)
