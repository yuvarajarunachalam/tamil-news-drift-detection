"""
RSS Feed Fetcher
Fetches articles from The Hindu Tamil Nadu RSS feed
"""

import feedparser
import requests
from typing import List, Dict, Any
from datetime import datetime
import time


class RSSFetcher:
    """Fetch articles from RSS feeds"""
    
    def __init__(self, feed_url: str = "https://www.thehindu.com/news/national/tamil-nadu/feeder/default.rss"):
        """
        Initialize RSS fetcher
        
        Args:
            feed_url: RSS feed URL
        """
        self.feed_url = feed_url
    
    def fetch_latest_articles(self, max_articles: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch latest articles from RSS feed
        
        Args:
            max_articles: Maximum number of articles to fetch
        
        Returns:
            List of article dictionaries
        """
        print(f"\n📡 Fetching RSS feed: {self.feed_url}")
        
        try:
            # Parse RSS feed
            feed = feedparser.parse(self.feed_url)
            
            if not feed.entries:
                print("⚠️ No articles found in RSS feed")
                return []
            
            articles = []
            
            for entry in feed.entries[:max_articles]:
                # Extract article data
                article = {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "description": entry.get("summary", ""),
                    "pub_date": self._parse_date(entry.get("published", "")),
                    "guid": entry.get("id", entry.get("link", "")),
                    "source": "The Hindu",
                    "category": "Tamil Nadu",
                    "raw_json": str(entry)
                }
                
                # Try to extract image
                if hasattr(entry, 'media_content'):
                    article["image_url"] = entry.media_content[0].get('url', '') if entry.media_content else ''
                elif hasattr(entry, 'enclosures') and entry.enclosures:
                    article["image_url"] = entry.enclosures[0].get('href', '')
                else:
                    article["image_url"] = ''
                
                articles.append(article)
            
            print(f"✅ Fetched {len(articles)} articles from RSS feed")
            return articles
            
        except Exception as e:
            print(f"Error fetching RSS feed: {e}")
            return []
    
    def _parse_date(self, date_str: str) -> str:
        """Parse publication date to ISO format"""
        try:
            # Try parsing common RSS date formats
            parsed = feedparser._parse_date(date_str)
            if parsed:
                return datetime(*parsed[:6]).isoformat()
        except:
            pass
        
        # Fallback to current time
        return datetime.now().isoformat()
    
    def scrape_full_content(self, url: str) -> str:
        """
        Scrape full article content from URL
        
        Args:
            url: Article URL
        
        Returns:
            Full article text
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Basic extraction (you can enhance this with BeautifulSoup if needed)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # The Hindu article content is typically in <div class="articlebodycontent">
            content_div = soup.find('div', class_='articlebodycontent')
            
            if content_div:
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs])
                return content
            else:
                # Fallback: get all paragraphs
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs[:10]])
                return content
                
        except Exception as e:
            print(f"Could not scrape content from {url}: {e}")
            return ""