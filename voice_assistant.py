import whisper
import pyaudio
import wave
import os
import pyttsx3
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── 1. SETUP ──────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load Whisper model — downloads once, cached forever
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("✅ Whisper ready!\n")

# ── 2. TEXT TO SPEECH ENGINE ──────────────────────
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 175)      # speaking speed
tts_engine.setProperty('volume', 0.9)    # volume

def speak(text):
    print(f"Aria: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

# ── 3. RECORD AUDIO FROM MICROPHONE ───────────────
def record_audio(filename="input.wav", duration=5):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    print(f"🎤 Recording for {duration} seconds... SPEAK NOW!")

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save audio file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("✅ Recording complete!")
    return filename

# ── 4. TRANSCRIBE AUDIO → TEXT ────────────────────
def transcribe(audio_file):
    print("🔄 Transcribing...")
    result = whisper_model.transcribe(audio_file)
    text = result["text"].strip()
    print(f"📝 You said: {text}")
    return text

# ── 5. GET AI RESPONSE ────────────────────────────
conversation_history = [
    {
        "role": "system",
        "content": "You are Aria, a helpful voice AI assistant. Keep responses short and conversational — max 2-3 sentences. You are being used as a voice assistant so avoid bullet points and long lists."
    }
]

def get_ai_response(user_text):
    conversation_history.append({
        "role": "user",
        "content": user_text
    })

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=conversation_history,
        max_tokens=150,
        temperature=0.7
    )

    ai_reply = response.choices[0].message.content

    conversation_history.append({
        "role": "assistant",
        "content": ai_reply
    })

    return ai_reply

# ── 6. FULL VOICE PIPELINE ────────────────────────
def voice_pipeline():
    print("=" * 50)
    print("     AriaAI — Voice Assistant")
    print("=" * 50)
    speak("Hello! I am Aria, your AI voice assistant. How can I help you?")

    while True:
        print("\nPress ENTER to speak or type 'quit' to exit...")
        user_input = input()

        if user_input.lower() == "quit":
            speak("Goodbye! Have a great day!")
            break

        # Record → Transcribe → AI → Speak
        audio_file = record_audio(duration=5)
        user_text = transcribe(audio_file)

        if user_text == "":
            speak("I didn't catch that. Please try again.")
            continue

        ai_response = get_ai_response(user_text)
        speak(ai_response)

        # Clean up audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)

# ── 7. RUN ────────────────────────────────────────
voice_pipeline()
