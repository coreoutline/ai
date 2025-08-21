import requests
import json

# Test non-streaming endpoint
response = requests.post(
    "http://localhost:6000/predict",
    json={"prompt": "What is customer churn?"}
)
print("\nNon-streaming response:")
print(response.json()["response"].replace("<|endoftext|>",""))

# Test streaming endpoint
print("\nStreaming response:")
with requests.post(
    "http://localhost:6000/predict/stream",
    json={"prompt": "What is customer churn?"},
    stream=True
) as r:
    r.raise_for_status()
    
    for line in r.iter_lines():
        if line:
            try:
                data = line.decode('utf-8')
                chunk = json.loads(data)
                
                if 'token' in chunk:
                    print(chunk['token'].replace("<|endoftext|>",""), end="", flush=True)
                elif 'done' in chunk:
                    print("\nStream complete")
                    break
                elif 'error' in chunk:
                    print("\nError:", chunk['error'])
                    break
            except json.JSONDecodeError as e:
                print(f"\nError parsing chunk: {e}")
                break
                
print("\nTest complete!")