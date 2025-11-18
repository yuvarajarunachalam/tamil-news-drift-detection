"""
DataDog Client
Sends metrics and events to DataDog with proper error handling
"""

import os
from typing import Dict, Any
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
            print(f"✅ DataDog enabled. API Key: {self.api_key[:10]}...")
            
            # Import only if enabled
            try:
                from datadog_api_client import ApiClient, Configuration
                from datadog_api_client.v1.api.metrics_api import MetricsApi
                from datadog_api_client.v1.model.metrics_payload import MetricsPayload
                from datadog_api_client.v1.model.series import Series
                from datadog_api_client.v1.model.point import Point
                
                self.configuration = Configuration()

                # CRITICAL: Configure for US5 DataDog site
                self.configuration.server_variables["site"] = "us5.datadoghq.com"

                self.configuration.api_key['apiKeyAuth'] = self.api_key
                self.configuration.api_key['appKeyAuth'] = self.app_key

                print(f"✅ DataDog configured for US5 site")
                
                # Store classes
                self.ApiClient = ApiClient
                self.MetricsApi = MetricsApi
                self.MetricsPayload = MetricsPayload
                self.Series = Series
                self.Point = Point
                
                print("✅ DataDog client initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize DataDog client: {e}")
                self.enabled = False
    
    def send_drift_metrics(self, metrics: Dict[str, float]):
        """
        Send drift metrics to DataDog
        
        Args:
            metrics: Dictionary with drift metrics
        """
        if not self.enabled:
            print("[DATADOG DISABLED] Would send metrics:", metrics)
            return False
        
        try:
            timestamp = int(time.time())
            
            print(f"\n📤 Sending metrics to DataDog at timestamp {timestamp}:")
            print(f"   KL Divergence: {metrics['kl_divergence']}")
            print(f"   Cosine Semantic: {metrics['cosine_semantic']}")
            print(f"   Cosine Sentiment: {metrics['cosine_sentiment']}")
            
            with self.ApiClient(self.configuration) as api_client:
                api_instance = self.MetricsApi(api_client)
                
                series = [
                    self.Series(
                        metric="tamil_news.drift.kl_divergence",
                        type="gauge",
                        points=[self.Point([timestamp, metrics['kl_divergence']])],
                        tags=["env:production", "source:github_actions"]
                    ),
                    self.Series(
                        metric="tamil_news.drift.cosine_semantic",
                        type="gauge",
                        points=[self.Point([timestamp, metrics['cosine_semantic']])],
                        tags=["env:production", "source:github_actions"]
                    ),
                    self.Series(
                        metric="tamil_news.drift.cosine_sentiment",
                        type="gauge",
                        points=[self.Point([timestamp, metrics['cosine_sentiment']])],
                        tags=["env:production", "source:github_actions"]
                    )
                ]
                
                body = self.MetricsPayload(series=series)
                response = api_instance.submit_metrics(body=body)
                
                print(f"✅ DataDog API Response: {response}")
                print("✅ Metrics sent to DataDog successfully")
                return True
                
        except Exception as e:
            print(f"❌ Error sending metrics to DataDog:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            
            # Print more details if available
            if hasattr(e, 'status'):
                print(f"   HTTP Status: {e.status}")
            if hasattr(e, 'reason'):
                print(f"   Reason: {e.reason}")
            if hasattr(e, 'body'):
                print(f"   Body: {e.body}")
            
            import traceback
            print(f"\n📋 Full traceback:")
            traceback.print_exc()
            
            return False
    
    def send_pipeline_metrics(self, article_count: int, execution_time: float):
        """Send pipeline execution metrics"""
        if not self.enabled:
            print(f"[DATADOG DISABLED] Would send pipeline metrics: {article_count} articles, {execution_time}s")
            return False
        
        try:
            timestamp = int(time.time())
            
            with self.ApiClient(self.configuration) as api_client:
                api_instance = self.MetricsApi(api_client)
                
                series = [
                    self.Series(
                        metric="tamil_news.pipeline.articles_processed",
                        type="count",
                        points=[self.Point([timestamp, article_count])],
                        tags=["env:production"]
                    ),
                    self.Series(
                        metric="tamil_news.pipeline.execution_time",
                        type="gauge",
                        points=[self.Point([timestamp, execution_time])],
                        tags=["env:production"]
                    )
                ]
                
                body = self.MetricsPayload(series=series)
                api_instance.submit_metrics(body=body)
                
                print(f"✅ Pipeline metrics sent to DataDog")
                return True
                
        except Exception as e:
            print(f"❌ Error sending pipeline metrics to DataDog: {e}")
            return False