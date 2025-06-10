import streamlit as st
import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Load data
df_raw = pd.read_csv("datapariwisata_baru.csv")
df_preprocessed = pd.read_csv("preprocessing-makamgusdur.csv")
df_tfidf = pd.read_csv("hasil_tfidf_lengkap.csv")

st.title("Klasifikasi Informasi Pariwisata dengan Naive Bayes")

# Tampilkan dataset mentah
st.header("1. Dataset Mentah")
st.dataframe(df_raw.head())

# Tampilkan hasil preprocessing
st.header("2. Hasil Preprocessing")
st.dataframe(df_preprocessed[['snippet', 'text_clean', 'case_folding', 'tokenize', 'stemming']].head())

# Tampilkan hasil TF-IDF
st.header("3. Hasil TF-IDF")
st.dataframe(df_tfidf.iloc[:, -10:].head())  # Menampilkan 10 kolom terakhir (fitur tf-idf)

# Cek jika ada label untuk klasifikasi
if 'label' in df_tfidf.columns:
    st.header("4. Pelatihan dan Evaluasi Model Naive Bayes")

    X = df_tfidf.drop(columns=["date", "snippet", "text_clean", "case_folding", "tokenize", "stemming", "label"], errors='ignore')
    y = df_tfidf['label']

    # Encode label jika perlu
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Simpan model
    joblib.dump(model, "naive_bayes_model.joblib")
    st.success("Model disimpan sebagai naive_bayes_model.joblib")
else:
    st.warning("Kolom 'label' tidak ditemukan pada dataset TF-IDF. Tidak bisa melakukan pelatihan model.")
