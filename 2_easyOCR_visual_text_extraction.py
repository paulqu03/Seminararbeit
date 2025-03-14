import os
import cv2
import math
import json
import easyocr
import re
import multiprocessing as mp
from tqdm import tqdm

def clean_text(s):
    """
    Erlaubt nur Buchstaben, Ziffern, Leerzeichen und .,?!  
    Rest wird gelöscht.
    Mehrfache Leerzeichen -> 1 Leerzeichen. 
    """
    s = re.sub(r"[^A-Za-z0-9\s\.,\?\!]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def process_video(video_path, languages=('de', 'en'), use_gpu=True):
    """
    Liest ein Frame pro Sekunde des Videos,
    Entfernt unerwünschte Sonderzeichen,
    Lässt doppelte Texte innerhalb eines Videos weg,
    Schreibt neue Texte (mit Sekunde) in .txt-Datei,
    Gibt am Ende einen einzigen langen String mit allen unique-Texten zurück.
    """
    if not os.path.isfile(video_path):
        print(f"Datei {video_path} nicht gefunden!")
        return ""

    base_name = os.path.splitext(video_path)[0]
    txt_path = base_name + ".txt"

    # EasyOCR initialisieren
    reader = easyocr.Reader(list(languages), gpu=use_gpu)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Konnte {video_path} nicht öffnen.")
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = int(math.floor(frame_count / fps)) if fps else 0

    print(f"\nStarte OCR für {video_path}")
    print(f" - Ausgabe: {txt_path}")
    print(f" - Frames: {frame_count}, FPS: {fps}, Sek: ~{duration_seconds}")

    # Falls ein Text länger als ein Frame zu sehen ist
    seen_texts = set()          
    unique_texts_for_this_vid = []

    # Textextraktion
    with open(txt_path, "w", encoding="utf-8") as f:
        for second in range(duration_seconds):
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            success, frame = cap.read()
            if not success:
                break

            result = reader.readtext(frame, detail=0)  

            for raw_text in result:
                text = clean_text(raw_text)
                if not text:
                    continue
                if text in seen_texts:
                    continue

                line = f"[Sek {second:3d}] {text}\n"
                f.write(line)

                seen_texts.add(text)
                unique_texts_for_this_vid.append(text)

    cap.release()

    # Zusammenführen
    combined_text = " ".join(unique_texts_for_this_vid)
    return combined_text

def main():
    videos = [f"./assets/videos/video_{i}_h264.mp4" for i in range(35)]

    # Anzahl Prozesse
    num_processes = 4

    # Hier speichern wir pro Video genau EINE Zeichenkette
    all_texts = []

    with mp.Pool(processes=num_processes) as pool:
        # tqdm zeigt an, wie viele Videos fertig sind (oder auch nicht)
        for combined_text in tqdm(pool.imap(process_video, videos), total=len(videos), desc="Verarbeite Videos"):
            all_texts.append(combined_text)

    # Speichern in JSON
    json_path = "./assets/visual_texts.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(all_texts, jf, ensure_ascii=False, indent=2)

    print(f"\nDone")

if __name__ == "__main__":
    main()
