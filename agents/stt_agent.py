# agents/stt_agent.py
import logging
import base64
from typing import Dict, Any
from utils.gemini_client import GeminiClient
from vertexai.generative_models import Part

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class STTAgent:
    """
    Speech-to-Text agent using Google Gemini 2.5 Flash for transcribing audio files
    """
    
    def __init__(self):
        self.gemini_client = GeminiClient()
        
        # Language mapping for prompt generation
        self.language_names = {
            'english': 'English',
            'hindi': 'Hindi (हिन्दी)',
            'bengali': 'Bengali (বাংলা)',
            'tamil': 'Tamil (தமிழ்)',
            'telugu': 'Telugu (తెలుగు)',
            'marathi': 'Marathi (मराठी)',
            'gujarati': 'Gujarati (ગુજરાતી)',
            'kannada': 'Kannada (ಕನ್ನಡ)',
            'malayalam': 'Malayalam (മലയാളം)',
            'punjabi': 'Punjabi (ਪੰਜਾਬੀ)',
            'urdu': 'Urdu (اردو)'
        }
        
        logger.info("STTAgent initialized with Gemini")
    
    def transcribe_audio(self, audio_data: str, language: str = 'english', farm_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Transcribe audio data to text using Gemini 2.5 Flash
        
        Args:
            audio_data: Base64 encoded audio data
            language: Language for transcription (default: english)
            
        Returns:
            Dict with transcription results
        """
        
        try:
            # Check farm settings for preferred language
            transcription_language = language
            if farm_settings:
                preferred_languages = farm_settings.get('preferredLanguages', [])
                if preferred_languages:
                    transcription_language = preferred_languages[0].lower()

            # Get language name for prompt
            language_name = self.language_names.get(transcription_language.lower(), 'English')
            logger.info(f"Transcribing audio in preferred language: {language_name}")
            
            # Create transcription prompt
            prompt = f"""
            You are an expert transcription system. Please transcribe this audio recording accurately.

            Language: {language_name}
            Context: Agricultural/farming conversation

            Instructions:
            1. Transcribe exactly what is spoken word-for-word
            2. Use proper punctuation and capitalization
            3. Common agricultural terms may include: crops, diseases, fertilizers, pesticides, irrigation, harvest
            4. If unclear about a word, provide your best guess
            5. Output ONLY the transcribed text with no additional comments

            Transcribe this audio:
            """
            
            # Use the new audio analysis method
            response = self.gemini_client.analyze_audio(prompt, audio_data)
            
            if response and not response.startswith("I couldn't") and not response.startswith("I'm having trouble"):
                # Clean up the response
                transcript = response.strip()
                
                # Simple confidence estimation based on response quality
                confidence = self._estimate_confidence(transcript)
                
                logger.info(f"Transcription successful: {transcript[:50]}...")
                
                return {
                    'success': True,
                    'transcript': transcript,
                    'confidence': confidence,
                    'language': language,
                    'language_name': language_name,
                    'agent': 'stt_agent_gemini'
                }
            else:
                logger.warning(f"Transcription failed with response: {response}")
                return {
                    'success': False,
                    'error': 'Could not transcribe audio clearly',
                    'transcript': '',
                    'language': language,
                    'agent': 'stt_agent_gemini'
                }
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'transcript': '',
                'language': language,
                'agent': 'stt_agent_gemini'
            }
    
    def _estimate_confidence(self, transcript: str) -> float:
        """
        Simple confidence estimation based on transcript quality
        """
        if not transcript:
            return 0.0
        
        # Basic heuristics for confidence
        confidence = 0.8  # Base confidence
        
        # Reduce confidence for unclear markers
        unclear_count = transcript.lower().count('[unclear]')
        confidence -= (unclear_count * 0.1)
        
        # Reduce confidence for very short transcripts
        if len(transcript.split()) < 3:
            confidence -= 0.2
        
        # Reduce confidence if response looks like an error message
        error_indicators = ['sorry', 'cannot', 'unable', 'error', 'failed']
        if any(indicator in transcript.lower() for indicator in error_indicators):
            confidence -= 0.3
        
        return max(0.1, min(1.0, confidence))
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.language_names