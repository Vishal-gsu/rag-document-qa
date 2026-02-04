"""
LLM Manager - Handles multiple LLM providers:
1. OpenAI API (Cloud)
2. Groq API (Cloud - FREE, Fast)
3. Ollama GPU (Local with GPU acceleration)
4. Ollama CPU (Local CPU-only)
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI
from groq import Groq


class LLMManager:
    """Manages different LLM providers."""
    
    def __init__(self):
        self.provider = "openai"
        self.openai_client = None
        self.groq_client = None
        self.available_models = {}  # Initialize before checking Ollama
        self.ollama_available = self._check_ollama()
        
    def _check_ollama(self) -> bool:
        """Check if Ollama is installed and running."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                self.available_models["ollama"] = [m["name"] for m in models]
                return True
        except Exception as e:
            print(f"Ollama check failed: {e}")
        return False
    
    def set_provider(self, provider: str, **kwargs):
        """
        Set the LLM provider.
        
        Args:
            provider: "openai", "groq", "ollama_gpu", or "ollama_cpu"
            **kwargs: Additional parameters (api_key, model, etc.)
        """
        self.provider = provider
        
        if provider == "openai":
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            self.openai_client = OpenAI(api_key=api_key)
            self.model = kwargs.get("model", "gpt-3.5-turbo")
            
        elif provider == "groq":
            api_key = kwargs.get("api_key") or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("Groq API key required")
            self.groq_client = Groq(api_key=api_key)
            self.model = kwargs.get("model", "llama-3.3-70b-versatile")
            
        elif provider in ["ollama_gpu", "ollama_cpu"]:
            if not self.ollama_available:
                raise ValueError("Ollama not available. Please install and start Ollama.")
            self.model = kwargs.get("model", "llama3.2")
            
    def generate(self, 
                messages: List[Dict],
                temperature: float = 0.7,
                max_tokens: int = 500) -> str:
        """
        Generate text using the selected provider.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if self.provider == "openai":
            return self._generate_openai(messages, temperature, max_tokens)
        elif self.provider == "groq":
            return self._generate_groq(messages, temperature, max_tokens)
        elif self.provider == "ollama_gpu":
            return self._generate_ollama(messages, temperature, max_tokens, use_gpu=True)
        elif self.provider == "ollama_cpu":
            return self._generate_ollama(messages, temperature, max_tokens, use_gpu=False)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _generate_openai(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """Generate using OpenAI API."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating with OpenAI: {str(e)}"
    
    def _generate_groq(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """Generate using Groq API (FREE & Fast)."""
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating with Groq: {str(e)}"
    
    def _generate_ollama(self, 
                        messages: List[Dict], 
                        temperature: float, 
                        max_tokens: int,
                        use_gpu: bool = True) -> str:
        """Generate using Ollama (local)."""
        try:
            import requests
            
            # Convert messages to Ollama format
            prompt = self._messages_to_prompt(messages)
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": False
            }
            
            # Add GPU/CPU preference
            if not use_gpu:
                payload["num_gpu"] = 0  # Force CPU
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=180  # Increased to 3 minutes
            )
            
            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                return f"Ollama error: {response.text}"
                
        except Exception as e:
            return f"Error generating with Ollama: {str(e)}"
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert OpenAI-style messages to a single prompt."""
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def get_available_ollama_models(self) -> List[str]:
        """Get list of available Ollama models."""
        return self.available_models.get("ollama", [])
    
    def get_provider_info(self) -> Dict:
        """Get information about current provider."""
        info = {
            "provider": self.provider,
            "model": getattr(self, "model", None),
            "ollama_available": self.ollama_available
        }
        
        if self.provider in ["ollama_gpu", "ollama_cpu"]:
            info["available_models"] = self.get_available_ollama_models()
            info["using_gpu"] = (self.provider == "ollama_gpu")
        
        return info


# Quick test function
if __name__ == "__main__":
    manager = LLMManager()
    
    print("🔍 Checking available LLM providers...\n")
    
    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        print("✓ OpenAI API: Available")
    else:
        print("✗ OpenAI API: No API key found")
    
    # Check Ollama
    if manager.ollama_available:
        print(f"✓ Ollama: Available")
        models = manager.get_available_ollama_models()
        if models:
            print(f"  Models: {', '.join(models)}")
        else:
            print("  No models installed. Run: ollama pull llama3.2")
    else:
        print("✗ Ollama: Not available")
        print("  Install: https://ollama.com/download")
