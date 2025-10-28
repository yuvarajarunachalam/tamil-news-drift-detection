from dotenv import load_dotenv
import os

load_dotenv('config/.env')

print("Testing configuration...")
print(f"Supabase URL: {os.getenv('SUPABASE_URL')[:30]}...")  # Show first 30 chars
print(f"Supabase Key: {os.getenv('SUPABASE_KEY')[:30]}...")
print("✅ Environment variables loaded successfully!")