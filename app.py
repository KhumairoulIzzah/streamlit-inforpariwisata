import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("datapariwisata_baru.csv")
    return df

# Preprocessing sederhana (contoh bisa disesuaikan)
def preprocess(text):
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Train model
def train_model(X, y):
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, report, accuracy

# Save model to json
def save_model_json(model, vectorizer):
    joblib.dump(model, "model_nb.joblib")
    joblib.dump(vectorizer, "vectorizer_nb.joblib")

# UI
st.title("Analisis Sentimen Ulasan Google Maps Terhadap Wisata Religi di Jombang Menggunakan Metode Naive Bayes")

df = load_data()
st.subheader("1. Dataset Asli")
st.write(df.head())

st.subheader("2. Preprocessing")
df['processed_text'] = df['Deskripsi'].astype(str).apply(preprocess)
st.write(df[['Deskripsi', 'processed_text']].head())

st.subheader("3. Training dan Evaluasi Naive Bayes")

if st.button("Train Model"):
    X = df['processed_text']
    y = df['Label']
    
    model, vectorizer, report, accuracy = train_model(X, y)
    
    st.success(f"Akurasi: {accuracy:.2f}")
    st.text("Classification Report:")
    st.json(report)

    save_model_json(model, vectorizer)
    st.success("Model berhasil disimpan sebagai `model_nb.joblib` dan `vectorizer_nb.joblib`.")