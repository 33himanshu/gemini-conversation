import requests
import json
import base64
import os

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

data = response.json()

# Print the text response
print("Text response:", data["message"])

# Save the audio to a file
audio_base64 = data["audio"]
audio_data = base64.b64decode(audio_base64)

with open("response.mp3", "wb") as f:
    f.write(audio_data)

print("Audio saved to response.mp3")

# Try to play the audio
if os.name == "nt":  # Windows
    os.system("start response.mp3")
elif os.name == "posix":  # macOS or Linux
    if os.uname().sysname == "Darwin":  # macOS
        os.system("afplay response.mp3")
    else:  # Linux
        os.system("xdg-open response.mp3")