from flask import Flask, request, jsonify
import pandas as pd
import pickle
from model import get_similar_colleges, recommend_improvements, load_data,file_path  # Importing functions


app = Flask(__name__)

df = load_data(file_path)


with open("cosine_similarity.pkl", "rb") as f:
    cosin_sim = pickle.load(f)

@app.route('/similar_colleges', methods=['GET'])
def similar_colleges():
   
    institute_id = request.args.get("institute_id")
    if not institute_id:
        return jsonify({"error": "Institute ID is required"}), 400
    
    try:
        similar_colleges = get_similar_colleges(institute_id)
        return jsonify({"similar_colleges": similar_colleges.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommend_improvements', methods=['GET'])
def recommend():
   
    institute_id = request.args.get("institute_id")
    if not institute_id:
        return jsonify({"error": "Institute ID is required"}), 400

    try:
        recommendations = recommend_improvements(institute_id)
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
