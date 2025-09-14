from flask import Flask, request, jsonify, render_template
import tempfile, os
import speech_recognition as sr
import numpy as np
import librosa
import joblib
from tensorflow import keras
from pydub import AudioSegment

from scorer import RobertaAnchorScorer  # your scorer class

app = Flask(__name__)

# Load models
scorer = RobertaAnchorScorer("../data/processed/lexicon_for_sentences.csv")
emotion_model = keras.models.load_model("../models/speech_emotion_recognition_model.h5")
label_encoder = joblib.load("../models/speech_label_encoder.pkl")

def preprocess_audio(file_path, max_len=174):
    y, sr_rate = librosa.load(file_path, sr=None, res_type="kaiser_fast")
    mfccs = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=40)

    if mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfccs = mfccs[:, :max_len]

    return np.expand_dims(mfccs, axis=0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze_audio", methods=["POST"])
def analyze_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    # Save uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_file.save(tmp.name)
        path = tmp.name

    # Convert to WAV (if not already WAV)
    audio = AudioSegment.from_file(path)
    audio.export(path, format="wav")

    # Step 1: Transcription
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio_data = r.record(source)
        try:
            transcription = r.recognize_google(audio_data)
        except:
            transcription = ""

    # Step 2: Audio emotion
    X_new = preprocess_audio(path)
    y_pred = emotion_model.predict(X_new)
    pred_class = int(np.argmax(y_pred, axis=1)[0])
    pred_label = str(label_encoder.inverse_transform([pred_class])[0])

    # Step 3: Lexicon scoring
    state_scores = scorer.score_sentence(transcription)

    # Convert all NumPy floats to Python floats for JSON
    state_percentages = {
        "hypo": float(state_scores["percentages"]["hypo"]),
        "hyper": float(state_scores["percentages"]["hyper"]),
        "flow": float(state_scores["percentages"]["flow"])
    }

    matched_words = []
    for m in state_scores["matched"]:
        matched_words.append({
            "token": str(m["token"]),
            "matched_anchor": str(m["matched_anchor"]),
            "similarity": float(m["similarity"])
        })


    dominant_state = str(state_scores["dominant"])

    os.remove(path)

    return jsonify({
        "transcription": transcription,
        "emotion": pred_label,
        "matched_words": matched_words,
        "state_percentages": state_percentages,
        "dominant_state": dominant_state
    })

if __name__ == "__main__":
    app.run(debug=True)
