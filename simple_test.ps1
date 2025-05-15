# Simple PowerShell command to test the Edge TTS endpoint

$body = @{
    messages = @(
        @{
            role    = "user"
            content = "Tell me a short joke about programming"
        }
    )
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:8000/api/chat-to-speech" -Method Post -Body $body -ContentType "application/json"

# Display the text response
Write-Host "Response:" $response.message

# Save the audio to a file
$audioBytes = [Convert]::FromBase64String($response.audio)
[System.IO.File]::WriteAllBytes("response.mp3", $audioBytes)

# Play the audio
Start-Process "response.mp3"