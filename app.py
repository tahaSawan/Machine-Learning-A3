
# web_app/app.py (same as before - just ensuring you have the working version)
from flask import Flask, render_template, request
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Load model
repo_id = "tahasawan/linear-regression-numpy"
model_path = hf_hub_download(repo_id=repo_id, filename="linear_regression_weights.joblib")
scaler_path = hf_hub_download(repo_id=repo_id, filename="scaler.joblib")
weights = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    
    if request.method == "POST":
        try:
            input_str = request.form.get("features", "").strip()
            features = [float(x.strip()) for x in input_str.split(",") if x.strip()]
            
            if len(features) != 8:
                error = "Please enter exactly 8 numbers"
            else:
                features_array = np.array(features).reshape(1, -1)
                scaled_features = scaler.transform(features_array)
                coeffs = weights[:-1].flatten()
                intercept = weights[-1].item()
                prediction = round(float(np.dot(scaled_features, coeffs) + intercept), 4)
                
        except ValueError:
            error = "Invalid input - use numbers like: 1,2,3,4,5,6,7,8"
        except Exception as e:
            error = f"Error: {str(e)}"
    
    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
