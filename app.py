import streamlit as st
import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df_raw = pd.read_csv("datapariwisata_baru.csv")
df_preprocessed = pd.read_csv("preprocessing-makamgusdur.csv")
df_tfidf = pd.read_csv("hasil_tfidf_lengkap.csv")
df_akhir = pd.read_csv("hasil_akhir.csv")

st.title("Analisis Sentimen Ulasan Google Maps Terhadap Wisata Religi di Jombang Menggunakan Metode Naive Bayes")

# Tampilkan dataset mentah
st.header("1. Hasil Crawling Google Maps")
st.dataframe(df_raw.head())

# Tampilkan hasil preprocessing
st.header("2. Hasil Preprocessing")
st.dataframe(df_preprocessed[['snippet', 'text_clean', 'case_folding', 'tokenize', 'stemming']].head())

# Tampilkan hasil TF-IDF
st.header("3. Hasil TF-IDF")
st.dataframe(df_tfidf.iloc[:, -10:].head())  # Menampilkan 10 kolom terakhir (fitur tf-idf)

# sentimen
if 'label' in df_akhir.columns:
    st.header("4. Hasil Sentimen Ulasan")

    label_counts = df_akhir['label'].value_counts()
    st.write("Jumlah masing-masing sentimen:")
    st.bar_chart(label_counts)

    for label, count in label_counts.items():
        st.write(f"- **{label.capitalize()}**: {count} ulasan")
else:
    st.warning("Kolom 'label' tidak ditemukan pada hasil_akhir.csv.")

# 5. Tampilkan Snippet dan Label
st.header("5. Daftar Ulasan dan Label Sentimen")
if all(col in df_akhir.columns for col in ['snippet', 'label']):
    st.dataframe(df_akhir[['snippet', 'label']].head(20))  # Tampilkan 20 data pertama
else:
    st.warning("Kolom 'snippet' dan/atau 'label' tidak ditemukan.")
