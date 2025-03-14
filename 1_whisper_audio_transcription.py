import whisper
import os
import json

# Whisper-Modell laden 
model = whisper.load_model("base")
list = []
for i in range(35):
    
    video_path = f"./assets/videos/video_{i}.mp4"

    # Überprüfen, ob die Datei existiert
    if not os.path.isfile(video_path):
        print(f"Datei {video_path} nicht gefunden!")
        continue

    
    result = model.transcribe(video_path)
    text = result["text"]
    list.append(text) 

# Transkripte speichern
with open("./assets/audio_transcripts.json", "w", encoding="utf-8") as f:
    json.dump(list, f, ensure_ascii=False, indent=2)

    