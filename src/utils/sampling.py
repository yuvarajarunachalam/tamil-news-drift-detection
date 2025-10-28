"""
Article Sampling Utilities
Handles random sampling of articles for baseline and test sets
"""

import random
from typing import List, Dict, Any, Tuple
from .supabase_client import SupabaseClient


class ArticleSampler:
    """Handles sampling strategies for drift detection"""
    
    def __init__(self, db_client: SupabaseClient):
        """
        Initialize sampler with database client
        
        Args:
            db_client: SupabaseClient instance
        """
        self.db = db_client
    
    def sample_baseline_and_test(
        self,
        baseline_size: int = 500,
        test_size: int = 500,
        rss_first_for_test: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Sample baseline and test sets with RSS prioritization for test
        
        Strategy:
        - Baseline: Pure random from entire news_cleaned table
        - Test: Prioritize RSS articles, fill remaining with random archive articles
        
        Args:
            baseline_size: Number of articles for baseline (default: 500)
            test_size: Number of articles for test (default: 500)
            rss_first_for_test: Whether to prioritize RSS articles for test set
        
        Returns:
            Tuple of (baseline_articles, test_articles)
        """
        print("\n" + "="*60)
        print("🎲 SAMPLING BASELINE AND TEST SETS")
        print("="*60)
        
        # Check if we have enough articles
        total_count = self.db.count_articles()
        required = baseline_size + test_size
        
        print(f"📊 Total articles in database: {total_count}")
        print(f"📊 Required: {required} (baseline: {baseline_size}, test: {test_size})")
        
        if total_count < required:
            raise ValueError(
                f"Insufficient articles! Need {required}, but only have {total_count}"
            )
        
        # Step 1: Sample baseline (pure random from all articles)
        print(f"\n🔵 Sampling {baseline_size} baseline articles (pure random)...")
        baseline_articles = self.db.get_random_articles(limit=baseline_size)
        baseline_ids = [article["id"] for article in baseline_articles]
        print(f"✅ Baseline sample complete: {len(baseline_articles)} articles")
        
        # Step 2: Sample test set
        if rss_first_for_test:
            test_articles = self._sample_test_rss_first(
                test_size=test_size,
                exclude_ids=baseline_ids
            )
        else:
            # Simple random sampling excluding baseline
            print(f"\n🟢 Sampling {test_size} test articles (pure random, excluding baseline)...")
            test_articles = self.db.get_random_articles(
                limit=test_size,
                exclude_ids=baseline_ids
            )
        
        print(f"✅ Test sample complete: {len(test_articles)} articles")
        
        # Summary
        print("\n" + "="*60)
        print("📊 SAMPLING SUMMARY")
        print("="*60)
        print(f"Baseline articles: {len(baseline_articles)}")
        print(f"Test articles: {len(test_articles)}")
        print(f"Total sampled: {len(baseline_articles) + len(test_articles)}")
        
        # Show source distribution
        self._print_source_distribution(baseline_articles, "Baseline")
        self._print_source_distribution(test_articles, "Test")
        
        return baseline_articles, test_articles
    
    def _sample_test_rss_first(
        self,
        test_size: int,
        exclude_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Sample test set prioritizing RSS articles
        
        Args:
            test_size: Number of test articles needed
            exclude_ids: Article IDs to exclude (baseline IDs)
        
        Returns:
            List of test articles
        """
        print(f"\n🟢 Sampling {test_size} test articles (RSS-first strategy)...")
        
        # Get RSS articles count (excluding baseline)
        rss_count = self.db.count_articles(source_filter="The Hindu - RSS")
        print(f"📰 Available RSS articles: {rss_count}")
        
        # Get all available RSS articles (excluding baseline)
        rss_articles = self.db.get_random_articles(
            limit=rss_count,
            exclude_ids=exclude_ids,
            source_filter="The Hindu - RSS"
        )
        
        # Filter out any that might be in baseline
        rss_articles = [a for a in rss_articles if a["id"] not in exclude_ids]
        
        print(f"✅ Retrieved {len(rss_articles)} RSS articles (after excluding baseline)")
        
        test_articles = []
        
        # Priority 1: Use RSS articles
        if len(rss_articles) >= test_size:
            # We have enough RSS articles
            random.shuffle(rss_articles)
            test_articles = rss_articles[:test_size]
            print(f"✅ Test set complete: {test_size} RSS articles")
        else:
            # Use all RSS articles + fill remaining with archive
            test_articles.extend(rss_articles)
            remaining_needed = test_size - len(rss_articles)
            
            print(f"⚠️  Only {len(rss_articles)} RSS articles available")
            print(f"📚 Filling remaining {remaining_needed} from archive...")
            
            # Get archive articles
            rss_ids = [a["id"] for a in rss_articles]
            all_exclude_ids = exclude_ids + rss_ids
            
            archive_articles = self.db.get_random_articles(
                limit=remaining_needed,
                exclude_ids=all_exclude_ids,
                source_filter="The Hindu - Archive"
            )
            
            test_articles.extend(archive_articles)
            
            print(f"✅ Test set complete: {len(rss_articles)} RSS + {len(archive_articles)} Archive")
        
        return test_articles
    
    def _print_source_distribution(self, articles: List[Dict[str, Any]], label: str):
        """
        Print source distribution for a set of articles
        
        Args:
            articles: List of articles
            label: Label for the set (e.g., "Baseline", "Test")
        """
        rss_count = sum(1 for a in articles if a.get("source") == "The Hindu - RSS")
        archive_count = sum(1 for a in articles if a.get("source") == "The Hindu - Archive")
        
        print(f"\n{label} Source Distribution:")
        print(f"  📰 RSS: {rss_count} ({rss_count/len(articles)*100:.1f}%)")
        print(f"  📚 Archive: {archive_count} ({archive_count/len(articles)*100:.1f}%)")
    
    def get_article_ids(self, articles: List[Dict[str, Any]]) -> List[str]:
        """
        Extract article IDs from article list
        
        Args:
            articles: List of article dictionaries
        
        Returns:
            List of article IDs
        """
        return [article["id"] for article in articles]


# Convenience function
def sample_articles(
    db_client: SupabaseClient,
    baseline_size: int = 500,
    test_size: int = 500
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Sample baseline and test articles
    
    Args:
        db_client: SupabaseClient instance
        baseline_size: Number of baseline articles
        test_size: Number of test articles
    
    Returns:
        Tuple of (baseline_articles, test_articles)
    """
    sampler = ArticleSampler(db_client)
    return sampler.sample_baseline_and_test(baseline_size, test_size)
