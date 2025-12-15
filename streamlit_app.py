import streamlit as st
import librosa
import numpy as np
import pickle
import whisper
import os
import tempfile

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="Voice Watch – Audio Phishing Detection",
    layout="centered"
)

# -------------------- LOAD MODELS --------------------

@st.cache_resource
def load_models():
    model = pickle.load(open("model/vishing_model.pkl", "rb"))
    whisper_model = whisper.load_model("base", device="cpu")
    return model, whisper_model

model, whisper_model = load_models()

# -------------------- CONSTANTS --------------------

SUSPICIOUS_KEYWORDS = [
    "otp", "urgent", "bank", "verify", "account",
    "security", "immediately", "blocked", "fraud"
]

# -------------------- FEATURE EXTRACTION --------------------

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    return np.hstack((mfcc, chroma, zcr))

# -------------------- WHISPER (STREAMLIT SAFE) --------------------

def transcribe_audio(file_path):
    """
    Streamlit-safe Whisper transcription (NO ffmpeg)
    """
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        audio = audio.astype(np.float32)

        result = whisper_model.transcribe(
            audio,
            fp16=False,
            verbose=False
        )
        return result.get("text", "").lower()

    except Exception as e:
        st.error(f"Whisper error: {e}")
        return ""

# -------------------- KEYWORD DETECTION --------------------

def detect_keywords(transcript):
    return [w.upper() for w in SUSPICIOUS_KEYWORDS if w in transcript]

# -------------------- UI --------------------

st.title("🛡️ Voice Watch")
st.caption("AI-Powered Audio Phishing (Vishing) Detection")

uploaded_files = st.file_uploader(
    "Upload audio files (WAV / MP3)",
    type=["wav", "mp3"],
    accept_multiple_files=True
)

if uploaded_files:
    st.divider()
    st.subheader("🔍 Analysis Results")

    phishing_count = 0

    for uploaded_file in uploaded_files:

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

        st.audio(uploaded_file)

        # ---------- ML ----------
        features = extract_features(audio_path)
        prediction = model.predict([features])[0]
        confidence = model.predict_proba([features])[0].max()

        label = "Phishing" if prediction == 1 else "Legitimate"

        if prediction == 1:
            phishing_count += 1

        st.markdown(
            f"### {'⚠️' if prediction==1 else '✅'} **{label}**"
        )

        st.progress(float(confidence))
        st.caption(f"Confidence: **{confidence*100:.2f}%**")

        # ---------- WHISPER ----------
        transcript = transcribe_audio(audio_path)
        keywords = detect_keywords(transcript)

        # ---------- HIGHLIGHT ----------
        highlighted = transcript
        for kw in keywords:
            highlighted = highlighted.replace(
                kw.lower(),
                f"**:red[{kw.lower()}]**"
            )

        with st.expander("📝 Transcript"):
            st.markdown(highlighted if highlighted else "_No speech detected_")

        if keywords:
            st.error(f"🚨 Suspicious keywords detected: {', '.join(keywords)}")
        else:
            st.success("✅ No suspicious keywords detected")

        st.divider()

    # ---------- SUMMARY ----------
    st.subheader("📊 Batch Summary")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Files", len(uploaded_files))
    col2.metric("Phishing", phishing_count)
    col3.metric("Legitimate", len(uploaded_files) - phishing_count)
