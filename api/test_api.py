import requests
import json

# Test non-streaming endpoint
response = requests.post(
    "http://localhost:6000/predict",
    json={"prompt": """What does the Weighted average actuarial assumptions consist of? Actuarial assumptions The Group’s scheme liabilities are measured using the projected unit credit method using the principal actuarial assumptions set out below: Notes: 1 Figures shown represent a weighted average assumption of the individual schemes. 2 The rate of increases in pensions in payment and deferred revaluation are dependent on the rate of inflation.|                        | 2019 % | 2018 % | 2017 % ||------------------------|--------|--------|--------|| Weighted average actuarial assumptions used at 31 March1: |          |        |        || Rate of inflation2     | 2.9    | 2.9    | 3.0    || Rate of increase in salaries | 2.7   | 2.7    | 2.6    || Discount rate          | 2.3    | 2.5    | 2.6    |The actual evidence needed: 'Discount rate', 'Rate of increase in salaries', 'Rate of inflation'"""
          }
)
print("\nNon-streaming response:")
print(response.json()["response"].replace("<|endoftext|>",""))

# Test streaming endpoint
print("\nStreaming response:")
with requests.post(
    "http://localhost:6000/predict/stream",
    json={"prompt": """What does the Weighted average actuarial assumptions consist of? Actuarial assumptions The Group’s scheme liabilities are measured using the projected unit credit method using the principal actuarial assumptions set out below:
Notes:
1 Figures shown represent a weighted average assumption of the individual schemes.
2 The rate of increases in pensions in payment and deferred revaluation are dependent on the rate of inflation.

|                        | 2019 % | 2018 % | 2017 % |
|------------------------|--------|--------|--------|
| Weighted average actuarial assumptions used at 31 March1: |          |        |        |
| Rate of inflation2     | 2.9    | 2.9    | 3.0    |
| Rate of increase in salaries | 2.7   | 2.7    | 2.6    |
| Discount rate          | 2.3    | 2.5    | 2.6    |

The actual evidence needed: 'Discount rate', 'Rate of increase in salaries', 'Rate of inflation'"""},
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