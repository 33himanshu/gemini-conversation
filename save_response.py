import requests
import json

# Make the API call
response = requests.post(
    "http://localhost:8000/api/chat-to-speech",
    json={
        "messages": [
            {
                "role": "user",
                "content": "Tell me a short joke about programming"
            }
        ]
    }
)

# Save the entire response to a file
with open("response.json", "w") as f:
    json.dump(response.json(), f)

print("Response saved to response.json")