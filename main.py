import tkinter as tk
from tkinter import messagebox, scrolledtext
import pandas as pd
import numpy as np
import os
import traceback
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    log_loss
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_NAME = "sample2000binary.csv"
FILE_PATH = os.path.join(BASE_DIR, FILE_NAME)

vectorizer = None
model = None
veri_yuklendi = False
full_classification_report = "Metrikler hesaplanırken bir hata oluştu veya veri yüklenemedi."

etiket_gosterim_map = {'true': "Olumlu", 'false': "Olumsuz", 'neutr': "Nötr"}
renk_map = {'true': "green", 'false': "red", 'neutr': "gray"}

KULLAN_SMOTE = False

def turkcelestir_report(report_str):
    replacements = {
        "precision": "kesinlik ",
        "recall": "duyarlılık",
        "f1-score": "f1-skoru ",
        "support": "destek   ",
        "macro avg": "makro ort",
        "weighted avg": "ağır. ort"
    }
    for old, new in replacements.items():
        report_str = report_str.replace(old, new)
    return report_str

def generate_metrics_report(y_true, y_pred, y_proba, classes, label_map, title):
    report = f"--- {title.upper()} VERİ SETİ ANALİZİ ---\n\n"
    gui_labels = [label_map.get(c, c) for c in classes]
    
    acc = accuracy_score(y_true, y_pred)
    report += f"** Genel Performans **\nDoğruluk (Accuracy): {acc:.4f}\nHata Oranı: {1-acc:.4f}\n\n"
    
    sk_report = classification_report(y_true, y_pred, target_names=gui_labels, zero_division=0, labels=classes)
    report += "** Detaylı Sınıflandırma Raporu **\n" + turkcelestir_report(sk_report) + "\n\n"
    
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    report += "** Karmaşıklık Matrisi (Gerçek \ Tahmin) **\n"
    header = "          | " + " | ".join(f"{label_map.get(c,c):<8}" for c in classes) + "\n"
    report += header + "-" * len(header) + "\n"
    for i, label in enumerate(classes):
        row = f"{label_map.get(label,label):<10} | " + " | ".join(f"{cm[i, j]:<8}" for j in range(len(classes))) + "\n"
        report += row
    
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    report += f"\n** İstatistiksel Metrikler **\nCohen Kappa: {kappa:.4f}\nMCC (Matthews Kor.): {mcc:.4f}\n"
    
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro', labels=classes)
        report += f"Makro ROC AUC: {auc:.4f}\nLog Loss: {log_loss(y_true, y_proba, labels=classes):.4f}\n"
    except:
        report += "Olasılık bazlı metrikler hesaplanamadı.\n"
    
    return report

try:
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Hata: {FILE_NAME} dosyası eksik!")

    df = pd.read_csv(FILE_PATH, header=0, sep=';', encoding='utf-8').dropna(subset=['Label', 'Text'])
    X = df['Text'].astype(str)
    y = df['Label'].astype(str).str.strip().str.lower()

    valid_mask = y.isin(etiket_gosterim_map.keys())
    X, y = X[valid_mask], y[valid_mask]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    train_rep = generate_metrics_report(y_train, model.predict(X_train), model.predict_proba(X_train), model.classes_, etiket_gosterim_map, "Eğitim")
    test_rep = generate_metrics_report(y_test, model.predict(X_test), model.predict_proba(X_test), model.classes_, etiket_gosterim_map, "Test")
    
    full_classification_report = train_rep + "\n" + "="*70 + "\n\n" + test_rep
    veri_yuklendi = True

except Exception as e:
    full_classification_report = f"Kritik Hata: {str(e)}\n{traceback.format_exc()}"

def siniflandir_yorum():
    if not veri_yuklendi:
        messagebox.showerror("Hata", "Model yüklenemedi.")
        return
    metin = text_yorum.get("1.0", tk.END).strip()
    if not metin: return
    
    tfidf = vectorizer.transform([metin])
    tahmin = model.predict(tfidf)[0]
    olasılıklar = model.predict_proba(tfidf)[0]
    
    sonuc_metni = f"Tahmin: {etiket_gosterim_map.get(tahmin, tahmin)}\n"
    detay = ", ".join([f"{etiket_gosterim_map.get(c, c)}: {p:.2%}" for c, p in zip(model.classes_, olasılıklar)])
    label_sonuc.config(text=sonuc_metni + "Olasılıklar: " + detay, fg=renk_map.get(tahmin, "black"))

root = tk.Tk()
root.title("NLP Duygu Analizi Projesi")
root.geometry("850x950")

tk.Label(root, text="Sınıflandırılacak Yorum:", font=("Arial", 10, "bold")).pack(pady=5)
text_yorum = scrolledtext.ScrolledText(root, height=5, width=90)
text_yorum.pack(padx=10, pady=5)

tk.Button(root, text="ANALİZ ET", command=siniflandir_yorum, bg="#2E86C1", fg="white", font=("Arial", 10, "bold"), height=2, width=20).pack(pady=10)

label_sonuc = tk.Label(root, text="Sonuç burada görünecek...", font=("Arial", 12, "bold"), relief=tk.GROOVE, height=3, width=80)
label_sonuc.pack(pady=10)

tk.Label(root, text="Model Performans Metrikleri (Eğitim vs Test):", font=("Arial", 10, "bold underline")).pack(pady=5)
display = scrolledtext.ScrolledText(root, height=35, width=100, font=("Courier New", 9))
display.insert(tk.INSERT, full_classification_report)
display.config(state=tk.DISABLED)
display.pack(padx=10, pady=5)

root.mainloop()
