
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Last inn modellen som er trena med ni kjenneteikn
with open("models/model.pkl", "rb") as file:
    model = joblib.load(file)

selected_features = [
    "stalk_root",
    "odor",
    "bruises",
    "spore_print_color",
    "stalk_surface_below",
    "cap_surface",
    "cap_shape",
    "ring_number",
    "gill_size"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        values = {
            feature: request.form[feature]
            for feature in selected_features
        }

        # Lag ei rad med same kolonnenamn som under treninga
        X_new = pd.DataFrame([values], columns=selected_features)

        result = model.predict(X_new)[0]

        if result == "p":
            prediction = "Giftig (p)"
        elif result == "e":
            prediction = "Etande (e)"
        else:
            prediction = f"Ukjend klasse: {result}"

    return render_template(
        "index.html",
        features=selected_features,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)