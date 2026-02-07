## start up the server with uvicorn app:app --reload
## before running this
import requests

questions = [
    "Who needed an xray?",
    "Which patient has diabetes?",
    "Who was prescribed antibiotics?",
]

for q in questions:
    r = requests.post(
        "http://localhost:8000/ask",
        json={"question": q}
    )
    print(q)
    print(r.json()["answer"])
    print()