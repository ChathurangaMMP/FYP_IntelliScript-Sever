from flask import Flask, request, jsonify
from main import *

app = Flask(__name__)


@app.route("/generate_text", methods=["POST"])
def generate_text():
    prompt = request.json["prompt"]
    response = response_generation(prompt)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Accessible from laptops
