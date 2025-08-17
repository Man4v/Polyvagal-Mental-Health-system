from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

lexicon_df = pd.read_csv("../data/processed/dataset_with_weights.csv")
lexicon = {
    row.word.lower():{"state":row.state, "weight": row.weight}
    for _, row in lexicon_df.iterrows()
}

@app.route("/words", methods=["GET"])
def get_words():
    """Return all words from lexicon so frontend can display dynamically"""
    hypo_words = ["numb", "frozen", "empty"]
    hyper_words = ["anxious", "restless", "angry"]
    flow_words  = ["calm", "safe", "relaxed"]
    demo_words = hypo_words + hyper_words + flow_words
    return jsonify(demo_words)

@app.route("/predict", methods=["POST"])
def predict():
    """Calculate state percentages from user selections"""
    data = request.json  # Expect list of {"word":..., "intensity":...}
    scores = {"hypo": 0, "hyper": 0, "flow": 0}

    for item in data:
        w = item["word"].lower()
        inten = float(item["intensity"])
        if w in lexicon:
            state = lexicon[w]["state"]
            weight = lexicon[w]["weight"]
            scores[state] += inten * weight

    total = sum(scores.values())
    percentages = {s: (v/total*100) if total else 0 for s, v in scores.items()}
    return jsonify(percentages)

if __name__ == "__main__":
    app.run(debug=True)