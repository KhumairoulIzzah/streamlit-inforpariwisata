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

# Cek jika ada label untuk klasifikasi
if 'label' in df_akhir.columns:
    st.header("4. Pelatihan dan Evaluasi Model Naive Bayes")

    X = df_akhir.drop(columns=["date", "snippet"], errors='ignore')
    y = df_akhir['label']

    # Encode label jika perlu
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Akurasi
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Akurasi")
    st.write(f"{accuracy:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    st.pyplot(fig)

    # Simpan model
    joblib.dump(model, "naive_bayes_model.joblib")
    st.success("Model disimpan sebagai naive_bayes_model.joblib")

else:
    st.warning("Kolom 'label' tidak ditemukan pada dataset TF-IDF. Tidak bisa melakukan pelatihan model.")
