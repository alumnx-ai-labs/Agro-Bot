# agents/manager.py
import json
import logging
import os
from typing import Dict, Any
from datetime import datetime
from utils.gemini_client import GeminiClient
from utils.firestore_client import FirestoreClient
from agents.disease_detection import DiseaseDetectionAgent
from agents.stt_agent import STTAgent
from agents.rag_agent import RAGAgent
from agents.translator_agent import TranslatorAgent
from agents.general_agent import GeneralAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManagerAgent:
    """
    The orchestrator agent that manages all other agents
    """
    
    def __init__(self):
        self.gemini_client = GeminiClient()
        self.firestore_client = FirestoreClient()
        
        # Initialize available agents
        self.disease_agent = DiseaseDetectionAgent()
        self.stt_agent = STTAgent()
        self.translator_agent = TranslatorAgent()
        self.general_agent = GeneralAgent()  # ADD THIS LINE

        # Initialize RAG agent with error handling
        try:
            self.rag_agent = RAGAgent()
            rag_available = True
        except Exception as e:
            logger.warning(f"RAG agent initialization failed: {e}")
            self.rag_agent = None
            rag_available = False

        # Agent registry for easy expansion
        self.agents = {
            'disease_detection': self.disease_agent,
            'speech_to_text': self.stt_agent,
            'translator': self.translator_agent,
            'general_farming': self.general_agent,  # ADD THIS LINE
        }
        
        if rag_available:
            self.agents['government_schemes'] = self.rag_agent
        
        available_agents = list(self.agents.keys())
        logger.info(f"ManagerAgent initialized with agents: {', '.join(available_agents)}")
    

    def fallback_classification(self, input_data: Dict[str, Any]) -> str:
        """Simple fallback classification logic"""
        
        # Check for explicit query type
        if input_data.get('queryType') == 'government_schemes':
            return 'government_schemes'
        
        # Check for image data
        if input_data.get('image_data'):
            return 'disease_detection'
        
        # Check text content for keywords
        text_content = input_data.get('text', input_data.get('content', '')).lower()
        
        scheme_keywords = ['scheme', 'subsidy', 'loan', 'pm-kisan', 'government', 'policy', 'support', 'benefit']
        disease_keywords = ['disease', 'pest', 'problem', 'sick', 'dying', 'spot', 'infection']
        
        if any(keyword in text_content for keyword in scheme_keywords):
            return 'government_schemes'
        elif any(keyword in text_content for keyword in disease_keywords):
            return 'disease_detection'
        
        return 'general_query'
    

    def process_request(self, session_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main orchestration method with translation flow"""
        
        try:
            # Add initial manager thought
            self.firestore_client.add_manager_thought(
                session_id, 
                "🤔 Analyzing your request..."
            )
            
            # Step 1: Input Processing
            processed_input, original_language = self.process_input_stage(session_id, input_data)
            
            # Step 2: Logic Processing  
            logic_result = self.process_logic_stage(session_id, processed_input)
            
            # Step 3: Output Formation
            final_response = self.process_output_stage(session_id, logic_result, original_language)
            
            # Update session status
            self.firestore_client.update_session_status(
                session_id, 
                'completed', 
                final_response['message']
            )
            
            logger.info(f"Request processed successfully for session: {session_id}")

            final_result = {
                'session_id': session_id,
                'original_language': original_language,
                'final_response': final_response,
                'status': 'success',
                'debug_info': {
                    'model_used': 'gemini-2.5-flash',
                    'processing_time': 'calculated_in_production',
                    'confidence': logic_result.get('confidence', 'unknown') if logic_result else 'unknown'
                }
            }

            # Log full response
            enable_logging = os.getenv('ENABLE_RESPONSE_LOGGING', 'true').lower() == 'true'
            self.log_full_response(session_id, final_result, enable_logging)

            return final_result
                        
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            
            # Update session with error
            self.firestore_client.update_session_status(session_id, 'error')
            self.firestore_client.add_manager_thought(
                session_id,
                f"❌ Error occurred: {str(e)}"
            )
            
            return {
                'session_id': session_id,
                'error': str(e),
                'status': 'error'
            }  


    def process_input_stage(self, session_id: str, input_data: Dict[str, Any]) -> tuple:
        """Stage 1: Input Processing"""
        
        # Get preferred language from farm settings, fallback to payload language
        farm_settings = input_data.get('farm_settings', {})
        preferred_languages = farm_settings.get('preferredLanguages', [])
        
        # Use first preferred language, or fallback to payload language
        if preferred_languages:
            original_language = preferred_languages[0].lower()
        else:
            original_language = input_data.get('language', 'english')
        processed_data = {}
        
        # Simple concatenation approach - initialize empty strings for all inputs
        image_context = ""
        audio_text = ""
        user_text = ""
        
        # Handle image input
        if input_data.get('image_data'):
            self.firestore_client.add_manager_thought(
                session_id,
                "📸 Processing image for disease detection..."
            )
            
            # Run disease detection on image
            farm_settings = input_data.get('farm_settings', {})
            disease_result = self.disease_agent.analyze(input_data, {}, farm_settings)
            processed_data['disease_detection_result'] = disease_result
            processed_data['image_data'] = input_data['image_data']
            
            # Extract disease context for concatenation
            if disease_result.get('analysis', {}).get('primary_disease', {}).get('name'):
                disease_name = disease_result['analysis']['primary_disease']['name']
                image_context = f"Disease detected in image: {disease_name}. "
        
        # Handle audio input
        if input_data.get('audio_data'):
            self.firestore_client.add_manager_thought(
                session_id,
                "🎤 Transcribing audio..."
            )
            
            # Transcribe audio with preferred language
            stt_result = self.stt_agent.transcribe_audio(
                input_data['audio_data'], 
                original_language,
                farm_settings
            )
            
            if stt_result.get('success'):
                transcript = stt_result['transcript']
                
                # Translate to English if needed
                if original_language.lower() not in ['english', 'en']:
                    self.firestore_client.add_manager_thought(
                        session_id,
                        "🌐 Translating audio to English for processing..."
                    )
                    
                    translation_result = self.translator_agent.translate(
                        original_language, 'english', transcript, farm_settings
                    )
                    
                    if translation_result.get('success'):
                        english_text = translation_result['translated_text']
                    else:
                        english_text = transcript
                else:
                    english_text = transcript
                    
                audio_text = english_text
                processed_data['original_transcript'] = transcript
            else:
                raise Exception(f"Audio transcription failed: {stt_result.get('error')}")
        
        # Handle text input
        if input_data.get('text') or input_data.get('content'):
            text_content = input_data.get('text', input_data.get('content', ''))
            
            # Translate to English if needed
            if original_language.lower() not in ['english', 'en']:
                self.firestore_client.add_manager_thought(
                    session_id,
                    "🌐 Translating text to English for processing..."
                )
                
                translation_result = self.translator_agent.translate(
                    original_language, 'english', text_content
                )
                
                if translation_result.get('success'):
                    english_text = translation_result['translated_text']
                else:
                    english_text = text_content
            else:
                english_text = text_content
                
            user_text = english_text
            processed_data['original_text'] = text_content
        
        # Simple concatenation - empty strings add nothing
        final_query = image_context + audio_text + user_text
        
        # Update processed data with concatenated query
        if final_query.strip():
            processed_data['text'] = final_query.strip()
        
        # Add metadata
        processed_data.update({
            'type': 'processed',
            'language': 'english',
            'queryType': input_data.get('queryType'),
            'farm_settings': input_data.get('farm_settings', {})
        })
        
        # Validate we have something to process
        if not self.validate_processed_input(processed_data):
            raise Exception("No valid input provided (no text, audio, or image)")
        
        return processed_data, original_language

    def process_logic_stage(self, session_id: str, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Logic Processing"""
        
        # If we already have disease detection result and no text query, return it
        if processed_input.get('disease_detection_result') and not processed_input.get('text'):
            return processed_input['disease_detection_result']
        
        # For text queries (with or without disease context), use LLM to decide agent
        text_content = processed_input.get('text', '')
        
        if text_content:
            # Use LLM to classify intent and decide which agent to use
            classification = self.classify_intent_for_agent_selection(text_content, processed_input)
            
            if classification.get('agent') == 'disease_analysis':
                self.firestore_client.add_manager_thought(
                    session_id,
                    "🔬 Providing detailed disease analysis..."
                )
                
                # If we have disease detection result, use it for detailed analysis
                if processed_input.get('disease_detection_result'):
                    disease_data = processed_input['disease_detection_result']
                    
                    # Extract possible diseases from the detection result
                    if disease_data.get('analysis', {}).get('possible_diseases'):
                        possible_diseases = disease_data['analysis']['possible_diseases']
                    elif disease_data.get('analysis', {}).get('primary_disease'):
                        # Convert primary disease to possible diseases format
                        primary = disease_data['analysis']['primary_disease']
                        possible_diseases = [primary] if primary.get('name') else []
                    else:
                        possible_diseases = []
                    
                    farm_metadata = processed_input.get('farm_settings', {})
                    
                    # Import and use disease analysis agent
                    from agents.disease_analysis_agent import DiseaseAnalysisAgent
                    analysis_agent = DiseaseAnalysisAgent()
                    return analysis_agent.analyze_disease(farm_metadata, possible_diseases)
                else:
                    # No image provided, general disease advice
                    farm_settings = processed_input.get('farm_settings', {})
                    return self.general_agent.query(text_content, farm_settings)
                    
            elif classification.get('agent') == 'government_schemes':
                if self.rag_agent:
                    self.firestore_client.add_manager_thought(
                        session_id,
                        "🏛️ Searching government schemes..."
                    )
                    
                    farm_settings = processed_input.get('farm_settings', {})
                    return self.rag_agent.query(text_content, {'farm_settings': farm_settings}, farm_settings)
                else:
                    return {
                        'type': 'error',
                        'message': 'Government schemes functionality is not available.',
                        'agent': 'manager_fallback'
                    }
            else:
                # General farming query
                self.firestore_client.add_manager_thought(
                    session_id,
                    "💬 Providing personalized farming assistance..."
                )
                
                farm_settings = processed_input.get('farm_settings', {})
                return self.general_agent.query(text_content, farm_settings)
        
        # Fallback for edge cases
        return {
            'type': 'error',
            'message': 'Unable to process the request. Please provide text or image input.',
            'agent': 'manager_fallback'
        }


    def process_output_stage(self, session_id: str, logic_result: Dict[str, Any], original_language: str) -> Dict[str, Any]:
        """Stage 3: Output Formation"""
        
        # Create response in English first
        english_response = self.synthesize_response({}, logic_result)
        
        # Get farm settings to check preferred languages
        farm_settings = logic_result.get('farm_settings', {}) if isinstance(logic_result, dict) else {}
        preferred_languages = farm_settings.get('preferredLanguages', [])
        
        # Check if English is in preferred languages
        english_preferred = any(lang.lower() in ['english', 'en'] for lang in preferred_languages)
        
        # Use preferred language for output, fallback to original language
        target_language = original_language
        if preferred_languages and not english_preferred:
            target_language = preferred_languages[0].lower()
        
        # Translate if target language is not English
        if target_language not in ['english', 'en'] and not english_preferred:
            self.firestore_client.add_manager_thought(
                session_id,
                f"🌐 Translating response to {target_language}..."
            )

            # Translate main message
            translation_result = self.translator_agent.translate(
                'english', target_language, english_response['message']
            )
            
            if translation_result.get('success'):
                # Create translated response preserving all context
                translated_response = english_response.copy()
                translated_response['message'] = translation_result['translated_text']
                translated_response['original_english'] = english_response['message']
                translated_response['language'] = target_language
                
                # Translate detailed analysis if present (for disease responses)
                if english_response.get('detailed_analysis'):
                    try:
                        # Translate key fields in detailed analysis
                        detailed = english_response['detailed_analysis'].copy()
                        
                        # Translate treatment plan if present
                        if detailed.get('treatment_plan', {}).get('immediate_actions'):
                            actions_text = '. '.join(detailed['treatment_plan']['immediate_actions'])
                            action_translation = self.translator_agent.translate('english', target_language, actions_text)
                            if action_translation.get('success'):
                                detailed['treatment_plan']['immediate_actions_translated'] = action_translation['translated_text'].split('. ')
                        
                        translated_response['detailed_analysis'] = detailed
                    except Exception as e:
                        logger.warning(f"Failed to translate detailed analysis: {e}")
                
                return translated_response
            else:
                # Fallback to English if translation fails
                english_response['translation_note'] = 'Translation failed, showing in English'
                return english_response
        
        return english_response


    def synthesize_response(self, classification: Dict, agent_response: Dict) -> Dict[str, Any]:
        """Create a unified response from agent outputs"""
        
        if not agent_response:
            return {
                'message': "I couldn't process your request properly. Please try again.",
                'type': 'error'
            }
        
        agent_type = agent_response.get('agent', agent_response.get('type', 'unknown'))
        
        # For disease detection responses
        if 'disease' in agent_type and agent_response.get('analysis'):
            analysis = agent_response['analysis']
            
            # Create farmer-friendly summary
            summary_parts = []
            
            if analysis.get('disease_name'):
                summary_parts.append(f"🔍 **Detected Issue**: {analysis['disease_name']}")
            
            if analysis.get('severity'):
                summary_parts.append(f"📊 **Severity**: {analysis['severity']}")
            
            if analysis.get('immediate_action'):
                summary_parts.append(f"⚡ **Immediate Action**: {analysis['immediate_action']}")
            
            if analysis.get('treatment_summary'):
                summary_parts.append(f"💊 **Treatment**: {analysis['treatment_summary']}")
            
            message = "\n\n".join(summary_parts) if summary_parts else agent_response.get('message', 'Analysis completed.')
            
            return {
                'message': message,
                'type': 'disease_analysis',
                'detailed_analysis': analysis,
                'confidence': agent_response.get('confidence', 'medium')
            }
        
        # For government schemes responses
        elif 'rag' in agent_type or agent_response.get('schemes'):
            message = agent_response.get('message', 'Government schemes information retrieved.')
            
            # Add scheme summary if available
            if agent_response.get('schemes'):
                scheme_count = len(agent_response['schemes'])
                message = f"Found {scheme_count} relevant scheme(s) for your query.\n\n{message}"
            
            return {
                'message': message,
                'type': 'government_schemes',
                'schemes': agent_response.get('schemes', []),
                'sources': agent_response.get('sources', []),
                'confidence': agent_response.get('confidence', 'medium')
            }
        
        # For general responses
        return {
            'message': agent_response.get('message', 'Request processed.'),
            'type': agent_response.get('type', 'general'),
        }
    

    def log_full_response(self, session_id: str, response_data: Dict[str, Any], enable_logging: bool = True):
        """Log complete response for debugging"""
        if not enable_logging:
            return
            
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'response_type': response_data.get('final_response', {}).get('type', 'unknown'),
                'full_response': response_data
            }
            
            # Log to console
            logger.info(f"FULL_RESPONSE_LOG: {json.dumps(log_entry, indent=2)}")
            
            # Save to Firestore for debugging
            self.firestore_client.db.collection('response_logs').document(session_id).set(log_entry)
            
        except Exception as e:
            logger.error(f"Failed to log response: {e}")


    def clean_json_response(self, response: str) -> str:
        """Clean JSON response by removing markdown code blocks"""
        import re
        
        # Remove ```json and ``` wrappers
        cleaned = re.sub(r'^```json\s*', '', response.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned.strip(), flags=re.MULTILINE)
        
        return cleaned.strip()
    

    def validate_processed_input(self, processed_data: Dict[str, Any]) -> bool:
        """Validate that we have some input to process"""
        return (
            processed_data.get('text') or 
            processed_data.get('disease_detection_result') or
            processed_data.get('image_data')
        )


    def classify_intent_for_agent_selection(self, text_content: str, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to decide which agent to use for text queries"""
        
        # Check for disease context from image analysis
        disease_context = ""
        if processed_input.get('disease_detection_result'):
            disease_info = processed_input['disease_detection_result']
            if disease_info.get('analysis'):
                disease_context = f"Previous disease detection found: {disease_info['analysis'].get('primary_disease', {}).get('name', 'disease detected')}"
        
        classification_prompt = f"""
        You are an AI assistant that routes farmer queries to the right specialist agent.
        
        CONTEXT:
        {disease_context}
        
        USER QUERY: {text_content}
        
        Available agents:
        1. "disease_analysis" - For disease treatment, prevention, detailed crop health analysis
        2. "government_schemes" - For government schemes, subsidies, loans, policies, financial support
        3. "general_farming" - For general farming advice, crop cultivation, best practices
        
        Rules:
        - If query mentions treatment, medicine, fungicide, pesticide, disease management → "disease_analysis"
        - If query mentions scheme, subsidy, loan, government support, PM-KISAN, etc. → "government_schemes"  
        - If there's disease context from image and query asks about treatment → "disease_analysis"
        - Everything else → "general_farming"
        
        Respond with JSON only:
        {{
            "agent": "disease_analysis|government_schemes|general_farming",
            "confidence": 0.95,
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            response = self.gemini_client.generate_text_flash(classification_prompt)
            cleaned_response = self.clean_json_response(response)
            classification = json.loads(cleaned_response)
            
            logger.info(f"Agent selection: {classification.get('agent')} (confidence: {classification.get('confidence', 'unknown')})")
            return classification
            
        except Exception as e:
            logger.error(f"Agent classification failed: {e}")
            # Fallback logic
            text_lower = text_content.lower()
            
            if any(keyword in text_lower for keyword in ['treatment', 'medicine', 'fungicide', 'pesticide', 'cure', 'spray']):
                return {'agent': 'disease_analysis', 'confidence': 0.7, 'reasoning': 'Fallback: disease treatment keywords'}
            elif any(keyword in text_lower for keyword in ['scheme', 'subsidy', 'loan', 'government', 'pm-kisan', 'support']):
                return {'agent': 'government_schemes', 'confidence': 0.7, 'reasoning': 'Fallback: scheme keywords'}
            else:
                return {'agent': 'general_farming', 'confidence': 0.5, 'reasoning': 'Fallback: default'}