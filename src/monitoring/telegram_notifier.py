"""
Telegram Bot Notifier
Sends batch processing status updates to Telegram
"""

import os
import requests
from typing import Optional


class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        self.enable = False
    
    def send_message(self, message: str) -> bool:
        """Send a message to Telegram"""
        if not self.enabled:
            print(f"[TELEGRAM DISABLED] {message}")
            return False
        
        try:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(self.base_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ Failed to send Telegram message: {e}")
            return False
    
    def batch_complete(self, batch_num: int, total_batches: int):
        """Send batch completion message"""
        message = f"✅ Batch {batch_num}/{total_batches} done"
        self.send_message(message)
    
    def batch_failed(self, batch_num: int, total_batches: int, error: str):
        """Send batch failure message"""
        message = f"❌ Batch {batch_num}/{total_batches} failed: {error}"
        self.send_message(message)
    
    def processing_started(self, total_articles: int, total_batches: int):
        """Send processing start notification"""
        message = f"🚀 <b>Baseline Generation Started</b>\n\n📊 Total Articles: {total_articles}\n📦 Total Batches: {total_batches}"
        self.send_message(message)
    
    def processing_complete(self, total_articles: int, elapsed_time: str):
        """Send processing completion notification"""
        message = f"🎉 <b>Baseline Generation Complete!</b>\n\n✅ Processed: {total_articles} articles\n⏱️ Time: {elapsed_time}"
        self.send_message(message)
    
    def error_alert(self, error_message: str):
        """Send critical error alert"""
        message = f"🚨 <b>CRITICAL ERROR</b>\n\n{error_message}"
        self.send_message(message)