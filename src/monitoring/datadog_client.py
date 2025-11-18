"""
DataDog Client
Sends metrics and events to DataDog
"""

import os
from typing import Dict, Any
from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v1.api.metrics_api import MetricsApi
from datadog_api_client.v1.model.metrics_payload import MetricsPayload
from datadog_api_client.v1.model.series import Series
from datadog_api_client.v1.model.point import Point
import time


class DataDogClient:
    """Send metrics to DataDog"""
    
    def __init__(self):
        """Initialize DataDog client"""
        self.api_key = os.getenv('DATADOG_API_KEY')
        self.app_key = os.getenv('DATADOG_APP_KEY')
        
        if not self.api_key:
            print("⚠️ Warning: DATADOG_API_KEY not found in environment")
            self.enabled = False
        else:
            self.enabled = True
            self.configuration = Configuration()
            self.configuration.api_key['apiKeyAuth'] = self.api_key
            self.configuration.api_key['appKeyAuth'] = self.app_key
    
    def send_drift_metrics(self, metrics: Dict[str, float]):
        """
        Send drift metrics to DataDog
        
        Args:
            metrics: Dictionary with drift metrics
        """
        if not self.enabled:
            print("[DATADOG DISABLED] Would send metrics:", metrics)
            return
        
        try:
            timestamp = int(time.time())
            
            with ApiClient(self.configuration) as api_client:
                api_instance = MetricsApi(api_client)
                
                series = [
                    Series(
                        metric="tamil_news.drift.kl_divergence",
                        type="gauge",
                        points=[Point([timestamp, metrics['kl_divergence']])],
                        tags=["env:production", "source:github_actions"]
                    ),
                    Series(
                        metric="tamil_news.drift.cosine_semantic",
                        type="gauge",
                        points=[Point([timestamp, metrics['cosine_semantic']])],
                        tags=["env:production", "source:github_actions"]
                    ),
                    Series(
                        metric="tamil_news.drift.cosine_sentiment",
                        type="gauge",
                        points=[Point([timestamp, metrics['cosine_sentiment']])],
                        tags=["env:production", "source:github_actions"]
                    )
                ]
                
                body = MetricsPayload(series=series)
                api_instance.submit_metrics(body=body)
                
                print("✅ Metrics sent to DataDog successfully")
                
        except Exception as e:
            print(f"❌ Error sending metrics to DataDog: {e}")
    
    def send_pipeline_metrics(self, article_count: int, execution_time: float):
        """Send pipeline execution metrics"""
        if not self.enabled:
            print(f"[DATADOG DISABLED] Would send pipeline metrics: {article_count} articles, {execution_time}s")
            return
        
        try:
            timestamp = int(time.time())
            
            with ApiClient(self.configuration) as api_client:
                api_instance = MetricsApi(api_client)
                
                series = [
                    Series(
                        metric="tamil_news.pipeline.articles_processed",
                        type="count",
                        points=[Point([timestamp, article_count])],
                        tags=["env:production"]
                    ),
                    Series(
                        metric="tamil_news.pipeline.execution_time",
                        type="gauge",
                        points=[Point([timestamp, execution_time])],
                        tags=["env:production"]
                    )
                ]
                
                body = MetricsPayload(series=series)
                api_instance.submit_metrics(body=body)
                
        except Exception as e:
            print(f"❌ Error sending pipeline metrics: {e}")