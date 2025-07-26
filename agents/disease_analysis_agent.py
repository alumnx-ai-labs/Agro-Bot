# agents/disease_analysis_agent.py
import json
import logging
from typing import Dict, Any, List
from utils.gemini_client import GeminiClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiseaseAnalysisAgent:
    """
    Agent for detailed disease analysis, treatment, and prevention recommendations
    """
    
    def __init__(self):
        self.gemini_client = GeminiClient()
        logger.info("DiseaseAnalysisAgent initialized")
    
    def analyze_disease(self, farm_metadata: Dict[str, Any], possible_diseases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze farm metadata and possible diseases to provide detailed recommendations
        
        Args:
            farm_metadata: Farm information including crop, location, soil, climate etc.
            possible_diseases: List of possible diseases with confidence scores
        """
        
        if not possible_diseases:
            return {
                'type': 'error',
                'message': 'No disease information provided for analysis.',
                'agent': 'disease_analysis'
            }
        
        try:
            # Create comprehensive analysis prompt
            analysis_prompt = self.create_analysis_prompt(farm_metadata, possible_diseases)
            
            # Generate analysis using Gemini Pro for complex reasoning
            analysis_result = self.gemini_client.generate_text_pro(analysis_prompt)
            
            # Return the friendly text response directly
            return {
                'type': 'disease_analysis',
                'message': analysis_result,
                'agent': 'disease_analysis',
                'farm_context': farm_metadata
            }
                
        except Exception as e:
            logger.error(f"Disease analysis failed: {e}")
            
            return {
                'type': 'error',
                'message': f'Failed to analyze disease: {str(e)}',
                'agent': 'disease_analysis'
            }
    
    def create_analysis_prompt(self, farm_metadata: Dict[str, Any], possible_diseases: List[Dict[str, Any]]) -> str:
        """Create comprehensive analysis prompt"""
        
        # Extract farm information
        crop_type = farm_metadata.get('cropType', 'Unknown crop')
        farmer_name = farm_metadata.get('farmerName', 'Farmer')
        location = farm_metadata.get('location', 'Unknown location')
        soil_type = farm_metadata.get('soilType', 'Unknown soil')
        farm_size = farm_metadata.get('acreage', 'Unknown size')
        current_stage = farm_metadata.get('currentStage', 'Unknown stage')
        climate = farm_metadata.get('climate', 'Unknown climate')
        irrigation = farm_metadata.get('irrigationType', 'Unknown irrigation')
        current_challenges = farm_metadata.get('currentChallenges', 'None mentioned')
        
        # Format possible diseases
        diseases_text = ""
        for i, disease in enumerate(possible_diseases, 1):
            diseases_text += f"{i}. {disease.get('name', 'Unknown')} (Confidence: {disease.get('confidence', 0):.2f})\n"
        
        prompt = f"""
        Plant pathologist analyzing {crop_type} disease.
        
        Farm: {farm_size} acres, {soil_type} soil, {current_stage} stage
        Location: {location}, Climate: {climate}
        Challenges: {current_challenges}
        
        Diseases detected:
        {diseases_text}

        1. Based on the above information, determine the most likely disease affecting the crop.
        2. Urgency level (1-5 scale)
        3. Top 2 immediate actions
        4. Primary treatment (organic/chemical)
        5. Expected timeline & cost estimate
        6. Key warning signs

        Keep response under 300 words. Use simple language and Indian rupees for costs.
        """
        
        return prompt
    