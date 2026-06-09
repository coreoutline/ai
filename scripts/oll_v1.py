import requests
import json
import numpy as np

class OllamaQuery:
    def __init__(self, model="llama2"):
        self.base_url = "http://localhost:11434/api"
        self.model = model

    def generate_with_logits(self, prompt: str, max_tokens: int = 100):
        """
        Query Ollama and extract logits from the response
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
                # Request logits in the response
                "raw": True
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/generate",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            result = response.json()
            print(result)
            
            return {
                'text': result.get('response', ''),
                'logits': result.get('logits', []),
                'top_tokens': result.get('top_tokens', [])
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error querying Ollama: {e}")
            return None

def main():
    # Initialize the querier
    ollama = OllamaQuery()
    
    # Example prompt
    prompt = "What is artificial intelligence?"
    
    # Generate response and get logits
    result = ollama.generate_with_logits(prompt)
    
    if result:
        print("Generated text:", result['text'])
        print("\nLogits sample:", result['logits'][:10] if result['logits'] else "No logits returned")
        print("\nTop tokens:", result['top_tokens'][:5] if result['top_tokens'] else "No top tokens returned")

if __name__ == "__main__":
    main()