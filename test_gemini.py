import os
import vertexai
from vertexai.generative_models import GenerativeModel

# Set your project
os.environ['GOOGLE_CLOUD_PROJECT'] = 'agro-bot-1212'

# Test different regions
regions = ['us-central1', 'us-east1', 'europe-west1']

for region in regions:
    print(f"\n🌍 Testing region: {region}")
    try:
        # Initialize Vertex AI
        vertexai.init(project='agro-bot-1212', location=region)
        
        # Try to create model
        model = GenerativeModel('gemini-1.5-pro')
        
        # Test simple generation
        response = model.generate_content("Say hello")
        print(f"✅ SUCCESS in {region}:", response.text[:50])
        break
        
    except Exception as e:
        print(f"❌ FAILED in {region}:", str(e)[:100])