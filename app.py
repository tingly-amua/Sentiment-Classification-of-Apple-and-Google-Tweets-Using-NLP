import pickle
from flask import Flask, request, jsonify

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return "Tweet Sentiment Classifier is live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")

    # Example: your model might require vectorization
    prediction = model.predict([text])[0]

    return jsonify({"text": text, "sentiment": prediction})

if __name__ == "__main__":
    app.run(debug=True)