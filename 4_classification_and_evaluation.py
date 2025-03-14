import os
import csv
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

import shap
import matplotlib.pyplot as plt

import plotly.figure_factory as ff


# -------------------------------
# Daten einlesen und vorbereiten
# -------------------------------
df = pd.read_csv("./assets/videos.csv")  # Spalten: VideoName, AudioText, VisualText, Label

# Zusätzliche Spalte "FullText" = Audio + Visueller Text
df["FullText"] = df["AudioText"].fillna("") + " " + df["VisualText"].fillna("")

# Stopwords englisch + deutsch herunterladen (einmalig)
# nltk.download('punkt')
# nltk.download('stopwords')

stop_words_en = set(stopwords.words("english"))
stop_words_de = set(stopwords.words("german"))
stop_words = stop_words_en.union(stop_words_de)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    # Nur alphanumerische Tokens ohne Stopwords
    filtered = [t for t in tokens if t.isalnum() and t not in stop_words]
    return " ".join(filtered)

df["FullTextPrepared"] = df["FullText"].apply(preprocess_text)

# Features (Rohtext + vorbereiteter Text)
X_raw = df["FullText"]
X_prepared = df["FullTextPrepared"]
y = df["Label"]

# Eindeutige Labels
candidate_labels = sorted(df["Label"].unique()) 



# Hilfsfunktion um die Confusion Matrix mit Plotly zu visualisieren
def plot_confusion_matrix_plotly(cm, labels, out_file, title="Confusion Matrix"):
    """
    Plottet eine Confusion Matrix (2D-Array cm) mit Plotly und speichert sie als PNG.
    labels: Liste der Label-Strings in passender Reihenfolge.
    out_file: Pfad zur Ausgabedatei (z. B. "./assets/diagrams/nb_raw_cm.png").
    """
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True
    )
    fig.update_layout(
        title=title,
        xaxis=dict(title='Predicted label'),
        yaxis=dict(title='True label')
    )
    # y-Achse invertieren, damit die erste Zeile oben steht (optional):
    fig['layout']['yaxis']['autorange'] = "reversed"

    # Ordner anlegen, falls nicht vorhanden
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # Speichern als PNG
    fig.write_image(out_file)
    print(f"Gespeichert: {out_file}")




'''
# -------------------------------------------------------------
# Naive Bayes Klassifikation (Leave-one-out Crossvalidation)
# -------------------------------------------------------------
def naive_bayes_loo(X, y, labels, description="NaiveBayes"):
    """
    Führt Leave-One-Out-CV für Naive Bayes durch.
    X: Liste/Serie mit Texten
    y: Liste/Serie mit Labels
    """
    loo = LeaveOneOut()
    vectorizer = TfidfVectorizer()
    clf = MultinomialNB()

    y_true_all, y_pred_all = [], []

    X_array = np.array(X)
    y_array = np.array(y)

    for train_index, test_index in loo.split(X_array):
        X_train, X_test = X_array[train_index], X_array[test_index]
        y_train, y_test = y_array[train_index], y_array[test_index]

        # Train
        X_train_vec = vectorizer.fit_transform(X_train)
        clf.fit(X_train_vec, y_train)

        # Test
        X_test_vec = vectorizer.transform(X_test)
        y_pred = clf.predict(X_test_vec)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    # Confusion Matrix + Classification Report
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    cr = classification_report(y_true_all, y_pred_all, labels=labels)

    print(f"\n=== {description} ===")
    print("Labels:", labels)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", cr)

    return cm, cr, y_true_all, y_pred_all

# Naive Bayes (Raw)
cm_nb_raw, cr_nb_raw, y_true_nb_raw, y_pred_nb_raw = naive_bayes_loo(X_raw, y, candidate_labels, "NaiveBayes (Raw)")
plot_confusion_matrix_plotly(cm_nb_raw, candidate_labels, "./assets/diagrams/nb_raw_cm.png",
                             title="Naive Bayes (Raw)")

# Naive Bayes (Prepared)
cm_nb_prep, cr_nb_prep, y_true_nb_prep, y_pred_nb_prep = naive_bayes_loo(X_prepared, y, candidate_labels, "NaiveBayes (Prepared)")
plot_confusion_matrix_plotly(cm_nb_prep, candidate_labels, "./assets/diagrams/nb_prep_cm.png",
                             title="Naive Bayes (Prepared)")
'''





# ---------------------------
# Zero-Shot Klassifikation
# ---------------------------
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Leere Texte behandeln
all_labels_for_cm = candidate_labels + ["NoText"]

video_names = df["VideoName"].tolist()

def zero_shot_classification(X, y, description="Zero-Shot"):
    y_true_all, y_pred_all = [], []
    preds = []

    for vid, text, label_true in zip(video_names, X, y):
        if not text.strip():
            predicted_label = "NoText"
            y_true_all.append(label_true)
            y_pred_all.append(predicted_label)
            continue

        result = zero_shot_classifier(text, candidate_labels=candidate_labels, multi_label=False)
        predicted_label = result["labels"][0]
        preds.append((vid, label_true, predicted_label))

        y_true_all.append(label_true)
        y_pred_all.append(predicted_label)

    # Für einzelne Vorhersagen
    df_zsl_results = pd.DataFrame(preds, columns=["VideoName", "y_true", "y_pred"])
    df_zsl_results["correct"] = df_zsl_results["y_true"] == df_zsl_results["y_pred"]

    # Performance
    cm = confusion_matrix(y_true_all, y_pred_all, labels=all_labels_for_cm)
    cr = classification_report(y_true_all, y_pred_all, labels=all_labels_for_cm)

    print(f"\n=== {description} ===")
    print("Labels:", all_labels_for_cm)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", cr)
    print("\n=== Einzelne Vorhersagen ===\n", df_zsl_results)

    return cm, cr, y_true_all, y_pred_all

# Zero-Shot (Raw)
cm_zs_raw, cr_zs_raw, y_true_zs_raw, y_pred_zs_raw = zero_shot_classification(X_raw, y, "Zero-Shot (Raw)")
plot_confusion_matrix_plotly(cm_zs_raw, all_labels_for_cm, "./assets/diagrams/zs_raw_cm.png",
                             title="Zero-Shot (Raw)")

# Zero-Shot (Prepared)
cm_zs_prep, cr_zs_prep, y_true_zs_prep, y_pred_zs_prep = zero_shot_classification(X_prepared, y, "Zero-Shot (Prepared)")
plot_confusion_matrix_plotly(cm_zs_prep, all_labels_for_cm, "./assets/diagrams/zs_prep_cm.png",
                             title="Zero-Shot (Prepared)")




'''
# --------------------------------------------------------------------------------
# SHAP-Analyse von Naive Bayes
#---------------------------------------------------------------------------------
# Ziel: Jeweils ein Bar-Plot mit den SHAP-Werten für das Label "Travel and Nature"

print("\n=== SHAP-Analyse: Naive Bayes (Label='Travel and Nature') ===")

# TF-IDF-Vektorisierung des gesamten Datensatzes
vectorizer_nb = TfidfVectorizer()
X_full_csr = vectorizer_nb.fit_transform(X_prepared)  
X_full = X_full_csr.toarray()                         

# Naive Bayes trainieren
clf_nb = MultinomialNB()
clf_nb.fit(X_full, y)

# Gibt predict_proba für jedes Sample zurück
def nb_predict_proba_numeric(X_num):
    return clf_nb.predict_proba(X_num)

# Hintergrundsets (Background) für KernelExplainer 
idx_background = np.random.choice(len(X_full), size=min(10, len(X_full)), replace=False)
X_background = X_full[idx_background]

# SHAP-Explainer anlegen
explainer_nb = shap.KernelExplainer(
    nb_predict_proba_numeric,
    X_background
)


mask_label = (y == "Travel and Nature")
X_label = X_full[mask_label]

# Alle Texte von Travel and Nature verwenden
X_label_sample = X_label[:13]

# SHAP-Werte berechnen
shap_values_nb = explainer_nb.shap_values(X_label_sample)

classes_nb = clf_nb.classes_
label_of_interest = "Travel and Nature"
if label_of_interest in classes_nb:
    idx_interest = list(classes_nb).index(label_of_interest)
else:
    idx_interest = 0

shap_class = shap_values_nb[idx_interest]  
mean_shap = np.mean(shap_class, axis=0)

# Sortierung nach absoluter Wichtigkeit
feature_names = vectorizer_nb.get_feature_names_out()
idx_sorted = np.argsort(np.abs(mean_shap))[::-1]

top_n = 15
top_features = feature_names[idx_sorted][:top_n]
top_shap_values = mean_shap[idx_sorted][:top_n]

# Bar-Plot
plt.figure(figsize=(8, 6))
colors = ["red" if val < 0 else "cornflowerblue" for val in top_shap_values]
plt.barh(range(len(top_features)), np.abs(top_shap_values), color=colors)
plt.yticks(range(len(top_features)), top_features)
plt.gca().invert_yaxis()  # wichtig, damit das größte oben ist
plt.title("SHAP (Naive Bayes) - Label='Travel and Nature'")
plt.xlabel("Mean(|SHAP Value|)")
out_shap_nb = "./assets/diagrams/shap_nb_travel_and_nature_bar.png"
os.makedirs(os.path.dirname(out_shap_nb), exist_ok=True)
plt.tight_layout()
plt.savefig(out_shap_nb)
plt.close()
print("\nDone")

'''


'''
# -----------------------------------------------------------------------------------------------------
# SHAP-Analyse für BART (Label "Design and Industry")
# Der Text eines richtig klassifizierten Videos wird hierbei genutzt, um die SHAP Analyse durchzuführen
# -----------------------------------------------------------------------------------------------------
print("\n=== SHAP-Analyse: BART (Label='Design and Industry') ===")


# Modell & Tokenizer laden (MNLI-Variante von BART)
model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

entailment_id = 2


# Der Text, welcher für die Analyse betrachtet wird
# Der Text gehört immer zu einem richtig klassifizierten Video des Labels
text = "atmosphere sense mood feeling get enter space certain texture particular object softness morning light change whole perspective right tools interior designer shape atmosphere space key explore used create hybrid scenes adapt mood time day requires delicate balance physical elements space form composition objects interior design light right combination generates sense comfort vibria exists create beautiful atmospheres rely dialogue light material elements direct gaze highlight certain aspects space defining mood subtle gradients tone colour soft accentuated light support professionals visualize project adapt design digitally using unique combinations lighting effects shape perfect atmosphere given situation design people feeling good put people centre everything generating sense beauty harmony improves quality life"


# Custom-Prediction-Funktion:
# Liefert "Entailment"-Wahrscheinlichkeit für das Label "Design and Industry"
def predict_design_industry_proba(texts):
    """
    Für jeden Text in 'texts' berechnet diese Funktion die Probability,
    dass der Text "über Design oder Industry" ist (Entailment).
    Return-Shape: (batch_size, 1)
    """
    hypothesis = "This text is about Design or Industry."
    encoded = tokenizer(
        [(premise, hypothesis) for premise in texts],  # list of (premise, hypothesis)
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits  
        probs = torch.softmax(logits, dim=1)
    entailment_probs = probs[:, entailment_id].unsqueeze(1)  
    return entailment_probs.cpu().numpy()


# SHAP-Explainer erzeugen (Token-Level)
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(
    predict_design_industry_proba,
    masker=masker,
    algorithm="permutation",
    output_names=["P(Design and Industry)"]
)


# SHAP-Werte berechnen
shap_values = explainer([text])
for i, token in enumerate(shap_values[0].feature_names):
    if not token.strip():
        shap_values[0].feature_names[i] = f"[EMPTY_{i}]"


# SHAP-Barplot
shap.plots.bar(shap_values[0], max_display=10)
# Diese Art von Plots muss ich leider manuell abspeichern

'''


'''
# Top-Tokens Diagramm

vals = shap_values[0].values
names = shap_values[0].feature_names

# Evtl. dimensionale Form anpassen
if len(vals.shape) > 1 and vals.shape[1] == 1:
    vals = vals.ravel()  # => 1D

# Leere Tokens entfernen
non_empty = []
for n, v in zip(names, vals):
    if n.strip(): 
        non_empty.append((n, v))
    else:
        pass

if len(non_empty) == 0:
    print("Alle Tokens sind leer")
    exit()

clean_names, clean_vals = zip(*non_empty)
clean_names = list(clean_names)
clean_vals = np.array(clean_vals)

# Nach absoluter SHAP-Größe sortieren
idx_sorted = np.argsort(np.abs(clean_vals))[::-1]
top_n = 15  
top_idx = idx_sorted[:top_n]
top_shap = clean_vals[top_idx]
top_tokens = [clean_names[i] for i in top_idx]

# Barplot
plt.figure(figsize=(8, 6))
colors = ["red" if v >= 0 else "cornflowerblue" for v in top_shap]
plt.barh(range(len(top_shap)), np.abs(top_shap), color=colors)
plt.yticks(range(len(top_shap)), top_tokens)
plt.gca().invert_yaxis()
plt.xlabel("|SHAP Value|")
plt.title("Top Tokens (PermutationExplainer)")
plt.tight_layout()
plt.savefig("./assets/diagrams/top_tokens.png", dpi=150)
plt.close()

print("\nDone")
'''

