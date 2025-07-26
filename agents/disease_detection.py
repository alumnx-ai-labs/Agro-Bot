# agents/disease_detection.py
import json
import logging
from typing import Dict, Any
from utils.gemini_client import GeminiClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiseaseDetectionAgent:
    """
    Specialized agent for crop disease detection
    """
    
    def __init__(self):
        self.gemini_client = GeminiClient()
        logger.info("DiseaseDetectionAgent initialized")
    
    def analyze(self, input_data: Dict[str, Any], entities: Dict[str, Any], farm_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze crop disease from image"""
        
        # Check if we have an image to analyze
        if not input_data.get('image_data'):
            return {
                'type': 'error',
                'message': 'Please provide an image of your crop for disease detection.',
                'agent': 'disease_detection'
            }
        
        try:
            # Create comprehensive analysis prompt
            analysis_prompt = self.create_analysis_prompt(input_data, entities, farm_settings)
            
            # Analyze image using Gemini Pro Vision with retry logic
            max_retries = 2
            for attempt in range(max_retries):
                analysis_result = self.gemini_client.analyze_image(
                    analysis_prompt, 
                    input_data['image_data']
                )
                
                # Check if response seems complete
                if len(analysis_result) > 100 and analysis_result.strip().endswith('}'):
                    break
                
                logger.warning(f"Attempt {attempt + 1}: Response seems incomplete, retrying...")
                
                if attempt == max_retries - 1:
                    logger.error(f"Final response after {max_retries} attempts: {analysis_result}")
                        
            # Try to parse structured response
            try:
                # Clean JSON response (remove markdown wrappers)
                cleaned_response = self.clean_json_response(analysis_result)
                structured_response = json.loads(cleaned_response)
                
                return {
                    'type': 'disease_analysis',
                    'analysis': structured_response,
                    'raw_response': analysis_result,
                    'agent': 'disease_detection'
                }
                
            except json.JSONDecodeError:
                # If JSON parsing fails, create structured response from plain text
                logger.warning("Could not parse JSON response, creating structured fallback")
                
                fallback_response = self.create_fallback_response(analysis_result)
                
                return {
                    'type': 'disease_analysis',
                    'analysis': fallback_response,
                    'raw_response': analysis_result,
                    'agent': 'disease_detection',
                    'note': 'Structured analysis not available, showing fallback response'
                }
                
        except Exception as e:
            logger.error(f"Disease analysis failed: {e}")
            
            return {
                'type': 'error',
                'message': f'Sorry, I had trouble analyzing the image: {str(e)}',
                'agent': 'disease_detection'
            }
    
    def create_analysis_prompt(self, input_data: Dict[str, Any], entities: Dict[str, Any], farm_settings: Dict[str, Any] = None) -> str:
        """Create a comprehensive prompt for disease analysis"""
        
        # Get additional context
        text_description = input_data.get('text', input_data.get('translated_text', ''))
        location = entities.get('location', 'India')
        crop_type = entities.get('crop_mentioned', farm_settings.get('cropType', 'crop') if farm_settings else 'crop')

        prompt = f"""
        You are a plant pathologist AI. Analyze this {crop_type} image for diseases only.

        CONTEXT:
        - Location: {location}
        - Description: {text_description}
        
        RESPONSE FORMAT (JSON only):
        {{
            "has_disease": true/false,
            "primary_disease": {{
                "name": "Disease name or null if healthy",
                "confidence": 0.85
            }},
            "possible_diseases": [
                {{
                    "name": "Disease name 1",
                    "confidence": 0.85
                }},
                {{
                    "name": "Disease name 2", 
                    "confidence": 0.65
                }}
            ]
        }}
        
        RULES:
        1. If no disease detected, set has_disease to false and primary_disease.name to null
        2. If confident (>90%), return only primary_disease
        3. If uncertain, return up to 5 possible diseases sorted by confidence
        4. Use precise disease names in English only
        
        CRITICAL: Return ONLY the JSON object. No markdown blocks.
        """
        
        return prompt
    
    def create_fallback_response(self, raw_text: str) -> Dict[str, Any]:
        """Create a basic structured response from plain text"""
        
        return {
            'has_disease': True,
            'primary_disease': {
                'name': 'Analysis available in raw text',
                'confidence': 0.5
            },
            'possible_diseases': [],
            'raw_analysis': raw_text[:500] + '...' if len(raw_text) > 500 else raw_text
        }
    
    def clean_json_response(self, response: str) -> str:
        """Clean and validate JSON response"""
        import re
        
        # Log the raw response for debugging
        logger.info(f"Raw response length: {len(response)}")
        logger.info(f"Raw response preview: {response[:300]}")
        
        # Remove markdown wrappers
        cleaned = re.sub(r'^```json\s*', '', response.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned.strip(), flags=re.MULTILINE)
        
        # Try to find a complete JSON object
        if not cleaned.strip().startswith('{'):
            # Look for the first { and try to extract JSON
            start_idx = cleaned.find('{')
            if start_idx != -1:
                cleaned = cleaned[start_idx:]
        
        # Check if JSON seems complete
        if not cleaned.strip().endswith('}'):
            logger.warning("JSON response appears incomplete")
            logger.warning(f"Response ends with: {cleaned[-50:]}")
        
        return cleaned.strip()