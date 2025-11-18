"""
DataDog Client
Sends metrics and events to DataDog using v2 API
"""

import os
from typing import Dict, Any
import time
from datetime import datetime


class DataDogClient:
    """Send metrics to DataDog using v2 API"""
    
    def __init__(self):
        """Initialize DataDog client"""
        self.api_key = os.getenv('DATADOG_API_KEY')
        self.app_key = os.getenv('DATADOG_APP_KEY')
        self.dd_site = os.getenv('DD_SITE', 'us5.datadoghq.com')
        
        if not self.api_key:
            print("⚠️ Warning: DATADOG_API_KEY not found in environment")
            self.enabled = False
        else:
            self.enabled = True
            print(f"✅ DataDog enabled. API Key: {self.api_key[:10]}...")
            print(f"📍 DataDog Site: {self.dd_site}")
            
            # Import v2 API (correct version)
            try:
                from datadog_api_client import ApiClient, Configuration
                from datadog_api_client.v2.api.metrics_api import MetricsApi
                from datadog_api_client.v2.model.metric_payload import MetricPayload
                from datadog_api_client.v2.model.metric_series import MetricSeries
                from datadog_api_client.v2.model.metric_point import MetricPoint
                from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
                
                self.configuration = Configuration()
                self.configuration.server_variables["site"] = self.dd_site
                self.configuration.api_key['apiKeyAuth'] = self.api_key
                
                # Store classes
                self.ApiClient = ApiClient
                self.MetricsApi = MetricsApi
                self.MetricPayload = MetricPayload
                self.MetricSeries = MetricSeries
                self.MetricPoint = MetricPoint
                self.MetricIntakeType = MetricIntakeType
                
                print(f"✅ DataDog v2 API client initialized for site: {self.dd_site}")
            except Exception as e:
                print(f"❌ Failed to initialize DataDog client: {e}")
                import traceback
                traceback.print_exc()
                self.enabled = False
    
    def send_drift_metrics(self, metrics: Dict[str, float]):
        """
        Send drift metrics to DataDog using v2 API
        
        Args:
            metrics: Dictionary with drift metrics
        """
        if not self.enabled:
            print("[DATADOG DISABLED] Would send metrics:", metrics)
            return False
        
        try:
            timestamp = int(datetime.now().timestamp())
            
            print(f"\n📤 Sending metrics to DataDog (v2 API) at timestamp {timestamp}:")
            print(f"   KL Divergence: {metrics['kl_divergence']}")
            print(f"   Cosine Semantic: {metrics['cosine_semantic']}")
            print(f"   Cosine Sentiment: {metrics['cosine_sentiment']}")
            
            with self.ApiClient(self.configuration) as api_client:
                api_instance = self.MetricsApi(api_client)
                
                # Create metric series using v2 API format
                series = [
                    self.MetricSeries(
                        metric="tamil_news.drift.kl_divergence",
                        type=self.MetricIntakeType.GAUGE,
                        points=[
                            self.MetricPoint(
                                timestamp=timestamp,
                                value=metrics['kl_divergence']
                            )
                        ],
                        tags=["env:production", "source:github_actions"]
                    ),
                    self.MetricSeries(
                        metric="tamil_news.drift.cosine_semantic",
                        type=self.MetricIntakeType.GAUGE,
                        points=[
                            self.MetricPoint(
                                timestamp=timestamp,
                                value=metrics['cosine_semantic']
                            )
                        ],
                        tags=["env:production", "source:github_actions"]
                    ),
                    self.MetricSeries(
                        metric="tamil_news.drift.cosine_sentiment",
                        type=self.MetricIntakeType.GAUGE,
                        points=[
                            self.MetricPoint(
                                timestamp=timestamp,
                                value=metrics['cosine_sentiment']
                            )
                        ],
                        tags=["env:production", "source:github_actions"]
                    )
                ]
                
                body = self.MetricPayload(series=series)
                response = api_instance.submit_metrics(body=body)
                
                print(f"✅ DataDog v2 API Response: {response}")
                print("✅ Metrics sent to DataDog successfully")
                return True
                
        except Exception as e:
            print(f"❌ Error sending metrics to DataDog:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            
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
        """Send pipeline execution metrics using v2 API"""
        if not self.enabled:
            print(f"[DATADOG DISABLED] Would send pipeline metrics: {article_count} articles, {execution_time}s")
            return False
        
        try:
            timestamp = int(datetime.now().timestamp())
            
            with self.ApiClient(self.configuration) as api_client:
                api_instance = self.MetricsApi(api_client)
                
                series = [
                    self.MetricSeries(
                        metric="tamil_news.pipeline.articles_processed",
                        type=self.MetricIntakeType.COUNT,
                        points=[
                            self.MetricPoint(
                                timestamp=timestamp,
                                value=float(article_count)
                            )
                        ],
                        tags=["env:production"]
                    ),
                    self.MetricSeries(
                        metric="tamil_news.pipeline.execution_time",
                        type=self.MetricIntakeType.GAUGE,
                        points=[
                            self.MetricPoint(
                                timestamp=timestamp,
                                value=execution_time
                            )
                        ],
                        tags=["env:production"]
                    )
                ]
                
                body = self.MetricPayload(series=series)
                api_instance.submit_metrics(body=body)
                
                print(f"✅ Pipeline metrics sent to DataDog (v2 API)")
                return True
                
        except Exception as e:
            print(f"❌ Error sending pipeline metrics to DataDog: {e}")
            return False