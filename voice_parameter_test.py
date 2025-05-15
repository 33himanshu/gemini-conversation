import requests
import json
import base64
import os
import datetime

# Voice ID for Antoni (male voice)
voice_id = "ErXwobaYiN019PkySvjV"

# Test different voice parameters
test_settings = [
    {"stability": 0.3, "similarity_boost": 0.8, "name": "clear_voice"},
    {"stability": 0.7, "similarity_boost": 0.7, "name": "stable_voice"},
    {"stability": 0.5, "similarity_boost": 0.9, "name": "natural_voice"}
]

for setting in test_settings:
    try:
        print(f"Testing voice with {setting['name']} settings...")
        
        # Make the API call with custom voice settings
        response = requests.post(
            "http://localhost:8000/api/text-to-speech",
            json={
                "text": "This is a test of the ElevenLabs voice with custom parameters. How does this sound?",
                "voice_id": voice_id,
                "model_id": "eleven_multilingual_v2",
                "stability": setting["stability"],
                "similarity_boost": setting["similarity_boost"]
            }
        )
        
        # Save the audio to a file
        filename = f"voice_test_{setting['name']}.mp3"
        
        with open(filename, "wb") as f:
            f.write(response.content)
        
        print(f"Audio saved to {filename}")
        
        # Play the audio (Windows-specific)
        os.system(f"start {filename}")
        
        # Wait for user input before continuing to next test
        input("Press Enter to continue to the next voice test...")
        
    except Exception as e:
        print(f"Error: {str(e)}")