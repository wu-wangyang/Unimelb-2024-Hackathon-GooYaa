from flask import Flask, request, jsonify, render_template
import pandas as pd
from recommender import recommend_companies

app = Flask(__name__)

# Load the company data
data_path = './data/Nature_2022-2024_PublicDataset_20240718_v4.xlsx'
scores_ranks_df = pd.read_excel(data_path, sheet_name='Scores and ranks')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the product name input from the user's request
    user_input = request.json.get('input', '').strip()

    if not user_input:
        return jsonify({"error": "Product name cannot be empty"}), 400

    try:
        # Get the recommendations from the recommender function
        recommendations = recommend_companies(user_input, scores_ranks_df)
        return jsonify(recommendations)
    except Exception as e:
        # Return a proper error message if something goes wrong
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)