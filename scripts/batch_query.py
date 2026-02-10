## start up the server with uvicorn app:app --reload
## before running this in a seperate terminal
import requests

questions = [
    "Why are xrays needed?",
    "How do we treat diabetes?",
    "Who was prescribed antibiotics?",
]

for q in questions:
    try: 
        r = requests.post(
            "http://localhost:8000/ask",
            json={"question": q}
        )
        r.raise_for_status() #Raise error if request fails
        print(f"Q: {q}")
        print(f"A: {r.json()['answer']}")
        print()
    except requests.exceptions.RequestException as e:
        print(f"Error with question '{q}': {e}")
        print()