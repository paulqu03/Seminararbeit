import csv
import json
import os

# manuelle Zuordnung: Videoindex -> Label 
index_to_label = {
    0: "Food and Cooking",
    1: "Design and Industry",
    2: "Travel and Nature",
    3: "Sports and Games",
    4: "Music and Instruments",
    5: "Music and Instruments",
    6: "Travel and Nature",
    7: "Design and Industry",
    8: "Travel and Nature",
    9: "Travel and Nature",
    10: "Travel and Nature",
    11: "Design and Industry",
    12: "Sports and Games",
    13: "Sports and Games",
    14: "Movies and Trailer",
    15: "Movies and Trailer",
    16: "Movies and Trailer",
    17: "Design and Industry",
    18: "Travel and Nature",
    19: "Music and Instruments",
    20: "Movies and Trailer",
    21: "Music and Instruments",
    22: "Travel and Nature",
    23: "Travel and Nature",
    24: "Music and Instruments",
    25: "Movies and Trailer",
    26: "Music and Instruments",
    27: "Travel and Nature",
    28: "Politics",
    29: "Travel and Nature",
    30: "Travel and Nature",
    31: "Travel and Nature",
    32: "Design and Industry",
    33: "Food and Cooking",
    34: "Travel and Nature",
}

# Visuelle Texte laaden und zusammenführen
# Aufgrund von Laufzeitproblemen ist der Teil etwas aufwändiger, da sonst das zusammenfassen prinzipiell schon erledigt wäre 
# (visual_texts.json müsste sonst nur eingelesen werden)
visual_text_dict = {}
for i in range(35):
    txt_file = f"./assets/videos/video_{i}_h264.txt"
    if not os.path.isfile(txt_file):
        print(f"WARNUNG: {txt_file} existiert nicht - überspringe.")
        continue

    # Lies alle Zeilen aus dem Textfile
    with open(txt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Entferne das "[Sek   x]" und fasse alle Zeilen zu einem String zusammen
    extracted_texts = []
    for line in lines:
        # Falls die Zeile nicht leer ist
        if line.strip():
            parts = line.split("]", 1)  
            if len(parts) == 2:
                extracted_texts.append(parts[1].strip())

    # Alle Zeilen zusammenfügen
    combined_text = " ".join(extracted_texts)
    visual_text_dict[i] = combined_text


# Audio Transkripte laden
with open("./assets/audio_transcripts.json", "r", encoding="utf-8") as f:
    transcripts = json.load(f)

# Visuelle Texte und Audio Transkripte in einer CSV Datei zusammenfassen
all_data = []
for i in range(35):
    video_name = f"video_{i}_h264.mp4"
    audio_text = transcripts[i]
    visual_text = visual_text_dict.get(i, "")
    label = index_to_label[i]
    all_data.append([video_name, audio_text, visual_text, label])

# Schreiben in CSV: Videoname, Audiotext, VisuellerText, Label
with open("./assets/videos.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["VideoName", "AudioText", "VisualText", "Label"])
    writer.writerows(all_data)