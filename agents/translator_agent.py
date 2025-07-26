# agents/translator_agent.py
import logging
from typing import Dict, Any, Optional
from utils.gemini_client import GeminiClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslatorAgent:
    """
    Agent for translating text between different languages using Gemini Flash
    """
    
    def __init__(self):
        self.gemini_client = GeminiClient()
        
        # Supported languages mapping
        self.supported_languages = {
            'english': 'English',
            'en': 'English',
            'hindi': 'Hindi',
            'hi': 'Hindi',
            'bengali': 'Bengali',
            'bn': 'Bengali',
            'tamil': 'Tamil',
            'ta': 'Tamil',
            'telugu': 'Telugu',
            'te': 'Telugu',
            'marathi': 'Marathi',
            'mr': 'Marathi',
            'gujarati': 'Gujarati',
            'gu': 'Gujarati',
            'kannada': 'Kannada',
            'kn': 'Kannada',
            'punjabi': 'Punjabi',
            'pa': 'Punjabi',
            'malayalam': 'Malayalam',
            'ml': 'Malayalam',
            'odia': 'Odia',
            'or': 'Odia',
            'assamese': 'Assamese',
            'as': 'Assamese',
            'urdu': 'Urdu',
            'ur': 'Urdu',
            'sanskrit': 'Sanskrit',
            'sa': 'Sanskrit',
            'nepali': 'Nepali',
            'ne': 'Nepali',
            'spanish': 'Spanish',
            'es': 'Spanish',
            'french': 'French',
            'fr': 'French',
            'german': 'German',
            'de': 'German',
            'chinese': 'Chinese',
            'zh': 'Chinese',
            'japanese': 'Japanese',
            'ja': 'Japanese',
            'arabic': 'Arabic',
            'ar': 'Arabic',
            'russian': 'Russian',
            'ru': 'Russian',
            'portuguese': 'Portuguese',
            'pt': 'Portuguese',
            'italian': 'Italian',
            'it': 'Italian',
        }
        
        logger.info("TranslatorAgent initialized successfully")
    
    def translate(self, translate_from: str, translate_to: str, text_to_translate: str, farm_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Translate text from one language to another
        
        Args:
            translate_from: Source language (e.g., 'english', 'hindi', 'en', 'hi')
            translate_to: Target language (e.g., 'hindi', 'english', 'hi', 'en')
            text_to_translate: Text to be translated
            
        Returns:
            Dict containing translation result and metadata
        """
        # Check farm settings for preferred language override
        if farm_settings:
            preferred_languages = farm_settings.get('preferredLanguages', [])
            if preferred_languages:
                # Use first preferred language as target if not English
                preferred_lang = preferred_languages[0].lower()
                if preferred_lang not in ['english', 'en']:
                    translate_to = preferred_lang
                    logger.info(f"Using preferred language from farm settings: {preferred_lang}")
        try:
            # Validate inputs
            if not text_to_translate or not text_to_translate.strip():
                return {
                    'success': False,
                    'error': 'Text to translate cannot be empty',
                    'agent': 'translator'
                }
            
            # Normalize and validate languages
            source_lang = self._normalize_language(translate_from)
            target_lang = self._normalize_language(translate_to)
            
            if not source_lang:
                return {
                    'success': False,
                    'error': f'Unsupported source language: {translate_from}',
                    'supported_languages': list(set(self.supported_languages.values())),
                    'agent': 'translator'
                }
            
            if not target_lang:
                return {
                    'success': False,
                    'error': f'Unsupported target language: {translate_to}',
                    'supported_languages': list(set(self.supported_languages.values())),
                    'agent': 'translator'
                }
            
            # Check if translation is needed
            if source_lang.lower() == target_lang.lower():
                return {
                    'success': True,
                    'translated_text': text_to_translate,
                    'source_language': source_lang,
                    'target_language': target_lang,
                    'confidence': 1.0,
                    'note': 'No translation needed - source and target languages are the same',
                    'agent': 'translator'
                }
            
            logger.info(f"Translating from {source_lang} to {target_lang}: {text_to_translate[:50]}...")
            
            # Create translation prompt
            translation_prompt = self._create_translation_prompt(
                source_lang, target_lang, text_to_translate
            )
            
            # Get translation using Gemini Flash
            translation_response = self.gemini_client.generate_text_flash(translation_prompt)
            
            # Parse and validate translation response
            parsed_response = self._parse_translation_response(translation_response)
            
            if parsed_response.get('success'):
                result = {
                    'success': True,
                    'translated_text': parsed_response['translated_text'],
                    'source_language': source_lang,
                    'target_language': target_lang,
                    'original_text': text_to_translate,
                    'confidence': parsed_response.get('confidence', 0.9),
                    'agent': 'translator'
                }
                
                # Add any additional notes or context
                if parsed_response.get('notes'):
                    result['notes'] = parsed_response['notes']
                
                logger.info(f"Translation successful: {source_lang} -> {target_lang}")
                return result
            else:
                return {
                    'success': False,
                    'error': parsed_response.get('error', 'Translation failed'),
                    'agent': 'translator'
                }
                
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                'success': False,
                'error': f'Translation service error: {str(e)}',
                'agent': 'translator'
            }
    
    def _normalize_language(self, language: str) -> Optional[str]:
        """Normalize language input to standard form"""
        if not language:
            return None
        
        language_key = language.lower().strip()
        return self.supported_languages.get(language_key)
    
    def _create_translation_prompt(self, source_lang: str, target_lang: str, text: str) -> str:
        """Create a translation prompt for Gemini"""
        
        # Special considerations for farming/agricultural context
        farming_context = """
        Important Context: This is part of a farmer assistance system. The text may contain:
        - Agricultural terms (crops, diseases, treatments)
        - Government scheme names and policies
        - Technical farming vocabulary
        - Local/regional farming practices
        
        Please maintain:
        - Technical accuracy for agricultural terms
        - Cultural context appropriate for farmers
        - Simple, clear language that farmers can understand
        """
        
        prompt = f"""
        You are a professional translator specializing in agricultural and farming content for Indian farmers.
        
        {farming_context}
        
        Task: Translate the following text from {source_lang} to {target_lang}.
        
        Source Language: {source_lang}
        Target Language: {target_lang}
        
        Text to translate:
        "{text}"
        
        Requirements:
        1. Provide accurate, natural translation
        2. Preserve the meaning and context
        3. Use appropriate agricultural terminology
        4. Keep the tone suitable for farmers
        5. If certain terms don't have direct translations, provide the closest equivalent with brief explanation
        
        Response format (JSON only, no markdown):
        {{
            "success": true,
            "translated_text": "your translation here",
            "confidence": 0.95
        }}
        
        If translation is not possible, respond with:
        {{
            "success": false,
            "error": "explanation of why translation failed"
        }}
        """
        
        return prompt
    
    def _parse_translation_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate the translation response from Gemini"""
        
        try:
            # Clean the response (remove any markdown formatting)
            import re
            cleaned_response = re.sub(r'^```json\s*', '', response.strip(), flags=re.MULTILINE)
            cleaned_response = re.sub(r'\s*```$', '', cleaned_response.strip(), flags=re.MULTILINE)
            cleaned_response = cleaned_response.strip()
            
            # Try to parse JSON
            import json
            parsed = json.loads(cleaned_response)
            
            # Validate required fields
            if 'success' not in parsed:
                return {
                    'success': False,
                    'error': 'Invalid response format from translation service'
                }
            
            if parsed.get('success') and 'translated_text' not in parsed:
                return {
                    'success': False,
                    'error': 'Translation response missing translated text'
                }
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse translation JSON: {e}")
            logger.error(f"Raw response: {response}")
            
            # Fallback: try to extract translation from response text
            fallback_translation = self._extract_fallback_translation(response)
            
            if fallback_translation:
                return {
                    'success': True,
                    'translated_text': fallback_translation,
                    'confidence': 0.7,
                    'notes': 'Translation extracted from unstructured response'
                }
            else:
                return {
                    'success': False,
                    'error': f'Failed to parse translation response: {str(e)}'
                }
        
        except Exception as e:
            logger.error(f"Translation response parsing failed: {e}")
            return {
                'success': False,
                'error': f'Response parsing error: {str(e)}'
            }
    
    def _extract_fallback_translation(self, response: str) -> Optional[str]:
        """Extract translation from unstructured response as fallback"""
        
        try:
            # Look for common patterns in translation responses
            import re
            
            # Pattern 1: "Translation: ..."
            pattern1 = re.search(r'Translation:\s*"([^"]+)"', response, re.IGNORECASE)
            if pattern1:
                return pattern1.group(1)
            
            # Pattern 2: Look for text in quotes
            pattern2 = re.search(r'"([^"]{10,})"', response)
            if pattern2:
                return pattern2.group(1)
            
            # Pattern 3: Simple fallback - return the response if it looks like translated text
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 10 and not line.startswith('{') and not line.startswith('```'):
                    return line
            
            return None
            
        except Exception:
            return None
    
    def get_supported_languages(self) -> Dict[str, Any]:
        """Get list of supported languages"""
        
        # Get unique language names
        unique_languages = list(set(self.supported_languages.values()))
        unique_languages.sort()
        
        return {
            'supported_languages': unique_languages,
            'total_count': len(unique_languages),
            'language_codes': {
                lang: [code for code, name in self.supported_languages.items() if name == lang]
                for lang in unique_languages
            },
            'note': 'You can use either language names (e.g., "Hindi") or language codes (e.g., "hi")'
        }
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect the language of given text"""
        
        try:
            if not text or not text.strip():
                return {
                    'success': False,
                    'error': 'Text cannot be empty for language detection'
                }
            
            detection_prompt = f"""
            Detect the language of the following text. Focus on Indian languages and common international languages.
            
            Text: "{text}"
            
            Respond with JSON only (no markdown):
            {{
                "detected_language": "Language Name",
                "confidence": 0.95,
                "language_code": "en/hi/ta/etc",
                "is_supported": true
            }}
            
            If detection is uncertain, respond with:
            {{
                "detected_language": "Unknown",
                "confidence": 0.1,
                "error": "Could not detect language reliably"
            }}
            """
            
            response = self.gemini_client.generate_text_flash(detection_prompt)
            
            # Parse response
            import json
            import re
            
            cleaned_response = re.sub(r'^```json\s*', '', response.strip(), flags=re.MULTILINE)
            cleaned_response = re.sub(r'\s*```$', '', cleaned_response.strip(), flags=re.MULTILINE)
            
            parsed = json.loads(cleaned_response.strip())
            
            # Validate if detected language is supported
            if 'detected_language' in parsed:
                detected_lang = parsed['detected_language']
                is_supported = self._normalize_language(detected_lang) is not None
                parsed['is_supported'] = is_supported
            
            parsed['success'] = True
            return parsed
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {
                'success': False,
                'error': f'Language detection error: {str(e)}'
            }