from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response
import sys
app = Flask(__name__)
CORS(app, resources={r"*":{"origins":"*"}})

@app.get("/")
def index_get():
    return render_template('base.html')
@app.route('/predict', methods=['POST'])
def predict():
        text = request.get_json().get("message")
        response = get_response(text)
        message = {"answer": response}
        response = jsonify(message);
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
if __name__ == "__main__":
    app.run(debug="True")
    



