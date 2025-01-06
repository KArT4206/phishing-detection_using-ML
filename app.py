from flask import Flask, request, render_template
import model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    result = model.predict_url(url)
    prediction = "Phishing" if result == 1 else "Legitimate"
    return render_template('index.html', prediction=prediction, url=url)

if __name__ == '__main__':
    app.run(debug=True)
