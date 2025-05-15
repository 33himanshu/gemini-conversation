import requests
import json
import base64
import os
import datetime

# Available ElevenLabs voice IDs (premium voices)
VOICE_OPTIONS = {
    "Rachel": "21m00Tcm4TlvDq8ikWAM",  # Female, warm and natural
    "Domi": "AZnzlk1XvdvUeBnXmlld",    # Female, professional
    "Bella": "EXAVITQu4vr4xnSDxMaL",   # Female, soft and warm
    "Antoni": "ErXwobaYiN019PkySvjV",  # Male, deep and clear
    "Thomas": "GBv7mTt0atIp3Br8iCZE",  # Male, British accent
    "Josh": "TxGEqnHWrfWFTfGW9XjX"     # Male, friendly
}

# Select a voice (change this to try different voices)
selected_voice = "Antoni"  # Try "Rachel", "Antoni", "Josh", etc.
voice_id = VOICE_OPTIONS[selected_voice]

try:
    print(f"Making API request with {selected_voice} voice...")
    
    # Make the API call with custom voice settings
    response = requests.post(
        "http://localhost:8000/api/text-to-speech",  # Using the dedicated TTS endpoint
        json={
            "text": "This is a test of the ElevenLabs voice quality. How does this sound compared to the Google voice?",
            "voice_id": voice_id,
            "model_id": "eleven_multilingual_v2",  # Higher quality model
            "stability": 0.5,  # Voice stability (0.0-1.0)
            "similarity_boost": 0.75  # Voice clarity (0.0-1.0)
        }
    )
    
    # Save the audio to a file with a unique name
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"voice_test_{selected_voice}_{timestamp}.mp3"
    
    with open(filename, "wb") as f:
        f.write(response.content)
    
    print(f"\nAudio saved to {filename}")
    
    # Play the audio (Windows-specific)
    print(f"Playing {filename}...")
    os.system(f"start {filename}")
    
except Exception as e:
    print(f"Error: {str(e)}")