"""
Fake Metrics Generator for DataDog Testing
Simulates steady decline in drift metrics over 2 hours
- Minutes 0-115: Normal with gradual decline
- Minute 116: Semantic WARNING (< 0.8)
- Minute 120: KL CRITICAL (> 0.1)
"""

import os
import sys
from datetime import datetime
from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v2.api.metrics_api import MetricsApi
from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
from datadog_api_client.v2.model.metric_payload import MetricPayload
from datadog_api_client.v2.model.metric_point import MetricPoint
from datadog_api_client.v2.model.metric_series import MetricSeries
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FakeMetricsGenerator:
    def __init__(self):
        """Initialize DataDog configuration"""
        self.datadog_api_key = os.getenv('DATADOG_API_KEY')
        
        if not self.datadog_api_key:
            raise ValueError("DATADOG_API_KEY must be set")
        
        self.datadog_config = Configuration()
        self.datadog_config.api_key["apiKeyAuth"] = self.datadog_api_key
        self.datadog_config.server_variables["site"] = "us5.datadoghq.com"
        
        logger.info("✅ DataDog configured (US5 region)")
        
        # Track run number
        self.run_number = self._get_run_number()
        
    def _get_run_number(self) -> int:
        """
        Determine current run number (1-120)
        Based on environment variable or file
        """
        # Try to get from environment (set by workflow)
        run_num = os.getenv('RUN_NUMBER')
        if run_num:
            return int(run_num)
        
        # Fallback: try to read from file
        run_file = '/tmp/drift_run_number.txt'
        if os.path.exists(run_file):
            with open(run_file, 'r') as f:
                num = int(f.read().strip())
                # Increment and save
                with open(run_file, 'w') as fw:
                    fw.write(str(num + 1))
                return num
        else:
            # First run
            with open(run_file, 'w') as f:
                f.write('1')
            return 1
    
    def calculate_metrics(self, run_number: int) -> dict:
        """
        Calculate fake metrics showing steady decline
        
        Pattern:
        - Runs 1-115: Normal with gradual decline
        - Run 116: Semantic WARNING (0.79 < 0.8)
        - Runs 117-119: Continued decline
        - Run 120: KL CRITICAL (0.25 > 0.1)
        """
        # Base values (healthy)
        base_kl = 0.005
        base_semantic = 0.98
        base_sentiment = 0.995
        
        if run_number <= 115:
            # Gradual decline: Linear interpolation
            # KL: 0.005 → 0.08 over 115 runs
            kl_divergence = base_kl + (0.08 - base_kl) * (run_number / 115)
            
            # Semantic: 0.98 → 0.82 over 115 runs (staying above 0.8)
            cosine_semantic = base_semantic - (base_semantic - 0.82) * (run_number / 115)
            
            # Sentiment: stays mostly stable
            cosine_sentiment = base_sentiment - 0.01 * (run_number / 115)
            
            scenario = "normal_decline"
            
        elif run_number == 116:
            # SEMANTIC WARNING TRIGGERED
            kl_divergence = 0.085
            cosine_semantic = 0.79  # Below 0.8 threshold
            cosine_sentiment = 0.98
            scenario = "semantic_warning"
            
        elif 117 <= run_number <= 119:
            # Continued decline after warning
            kl_divergence = 0.085 + (0.15 * (run_number - 116) / 4)
            cosine_semantic = 0.79 - (0.02 * (run_number - 116) / 4)
            cosine_sentiment = 0.97
            scenario = "pre_critical"
            
        else:  # run_number == 120
            # KL CRITICAL TRIGGERED
            kl_divergence = 0.25  # Well above 0.1 threshold
            cosine_semantic = 0.75
            cosine_sentiment = 0.96
            scenario = "kl_critical"
        
        return {
            'kl_divergence': round(kl_divergence, 4),
            'cosine_similarity_semantic': round(cosine_semantic, 4),
            'cosine_similarity_sentiment': round(cosine_sentiment, 4),
            'run_number': run_number,
            'scenario': scenario
        }
    
    def send_metrics(self, metrics: dict):
        """Send metrics to DataDog"""
        timestamp = int(datetime.now().timestamp())
        
        try:
            with ApiClient(self.datadog_config) as api_client:
                api_instance = MetricsApi(api_client)
                
                series = [
                    MetricSeries(
                        metric="tamil_news.drift.kl_divergence",
                        type=MetricIntakeType.GAUGE,
                        points=[MetricPoint(timestamp=timestamp, value=metrics['kl_divergence'])],
                        tags=[
                            "env:production",
                            "service:drift-detection",
                            f"scenario:{metrics['scenario']}",
                            f"run:{metrics['run_number']}"
                        ]
                    ),
                    MetricSeries(
                        metric="tamil_news.drift.cosine_similarity_semantic",
                        type=MetricIntakeType.GAUGE,
                        points=[MetricPoint(timestamp=timestamp, value=metrics['cosine_similarity_semantic'])],
                        tags=[
                            "env:production",
                            "service:drift-detection",
                            f"scenario:{metrics['scenario']}",
                            f"run:{metrics['run_number']}"
                        ]
                    ),
                    MetricSeries(
                        metric="tamil_news.drift.cosine_similarity_sentiment",
                        type=MetricIntakeType.GAUGE,
                        points=[MetricPoint(timestamp=timestamp, value=metrics['cosine_similarity_sentiment'])],
                        tags=[
                            "env:production",
                            "service:drift-detection",
                            f"scenario:{metrics['scenario']}",
                            f"run:{metrics['run_number']}"
                        ]
                    )
                ]
                
                body = MetricPayload(series=series)
                response = api_instance.submit_metrics(body=body)
                
                logger.info("=" * 70)
                logger.info(f"📊 RUN {metrics['run_number']}/120 - Scenario: {metrics['scenario'].upper()}")
                logger.info("=" * 70)
                logger.info(f"   KL Divergence: {metrics['kl_divergence']:.4f}")
                logger.info(f"   Cosine Semantic: {metrics['cosine_similarity_semantic']:.4f}")
                logger.info(f"   Cosine Sentiment: {metrics['cosine_similarity_sentiment']:.4f}")
                
                # Alert notifications
                if metrics['scenario'] == 'semantic_warning':
                    logger.warning("⚠️  SEMANTIC DRIFT WARNING - Email alert should trigger!")
                elif metrics['scenario'] == 'kl_critical':
                    logger.critical("🚨 KL DIVERGENCE CRITICAL - Email alert should trigger!")
                
                logger.info("✅ Metrics sent to DataDog")
                logger.info("=" * 70)
                
        except Exception as e:
            logger.error(f"❌ Error sending metrics: {e}")
            raise
    
    def run(self):
        """Main execution"""
        logger.info("🚀 Starting fake metrics generation")
        logger.info(f"   Run number: {self.run_number}/120")
        logger.info(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate metrics for this run
        metrics = self.calculate_metrics(self.run_number)
        
        # Send to DataDog
        self.send_metrics(metrics)
        
        # Summary
        remaining = 120 - self.run_number
        logger.info(f"\n⏰ Progress: {self.run_number}/120 runs complete")
        logger.info(f"   Remaining: {remaining} runs (~{remaining} minutes)")
        
        if self.run_number < 116:
            logger.info(f"   Next alert: Run 116 (Semantic WARNING) in {116 - self.run_number} runs")
        elif self.run_number < 120:
            logger.info(f"   Next alert: Run 120 (KL CRITICAL) in {120 - self.run_number} runs")
        else:
            logger.info("   🎉 All runs complete! Both alerts should have been triggered.")


def main():
    """Entry point"""
    try:
        generator = FakeMetricsGenerator()
        generator.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()