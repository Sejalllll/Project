from flask import Flask, render_template, request
import joblib
from utils import preprocess_image

app = Flask(__name__)
model = joblib.load('model/waste_classifier.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    moisture = float(request.form['moisture'])
    temp = float(request.form['temperature'])
    gas = float(request.form['gas'])

    img_feat = preprocess_image(image)
    sensor_feat = [moisture, temp, gas]

    final_input = img_feat + sensor_feat
    prediction = model.predict([final_input])[0]
    confidence = round(max(model.predict_proba([final_input])[0]) * 100, 2)

    label = "♻️ Recyclable" if prediction == 1 else "🚯 Non-Recyclable"
    return render_template('index.html', prediction=label, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
