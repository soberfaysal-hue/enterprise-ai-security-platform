from typing import Dict, Any, Optional
from .base import ModelAdapter


class GoogleAdapter(ModelAdapter):
    """Adapter for Google Gemini models"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro", timeout: int = 30, max_retries: int = 3):
        super().__init__(timeout, max_retries)
        self.api_key = api_key
        self.model = model
        self.model_type = "enterprise"
        self.client = None
        
        if api_key and api_key != "your-google-api-key-here":
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.client = genai
                self.generation_config = {
                    "temperature": 0.7,
                    "max_output_tokens": 1000,
                }
            except ImportError:
                self.client = None
    
    def generate(self, prompt: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate response using Google Gemini API"""
        if params is None:
            params = {}
        
        default_params = {
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 1000),
        }
        
        try:
            if self.client:
                model = self.client.GenerativeModel(self.model)
                generation_config = {
                    "temperature": default_params["temperature"],
                    "max_output_tokens": default_params["max_tokens"],
                }
                
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                response_text = ""
                if hasattr(response, 'text'):
                    response_text = response.text
                elif hasattr(response, 'parts'):
                    response_text = "".join([part.text for part in response.parts if hasattr(part, 'text')])
                
                return {
                    "response_text": response_text,
                    "model_name": self.model,
                    "model_type": self.model_type,
                    "vendor": "google",
                    "metadata": {
                        "tokens_used": len(prompt.split()) + len(response_text.split()),
                        "response_time_ms": 0,
                        "model_version": self.model
                    }
                }
            else:
                return {
                    "response_text": f"[Simulated Google Gemini response for: {prompt[:50]}...]",
                    "model_name": self.model,
                    "model_type": self.model_type,
                    "vendor": "google",
                    "metadata": {
                        "tokens_used": len(prompt.split()),
                        "response_time_ms": 0,
                        "model_version": self.model
                    }
                }
        except Exception as e:
            return {
                "response_text": f"[Error: {str(e)}]",
                "model_name": self.model,
                "model_type": self.model_type,
                "vendor": "google",
                "metadata": {
                    "error": str(e),
                    "response_time_ms": 0,
                    "model_version": self.model
                }
            }
    
    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_name": self.model,
            "model_type": self.model_type,
            "vendor": "google"
        }
