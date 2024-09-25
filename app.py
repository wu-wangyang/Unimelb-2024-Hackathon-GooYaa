from flask import Flask, request, jsonify
import pandas as pd
from recommender import recommend_companies

app = Flask(__name__)

# Load the company data
data_path = './data/Nature_2022-2024_PublicDataset_20240718_v4.xlsx'
scores_ranks_df = pd.read_excel(data_path, sheet_name='Scores and ranks')


@app.route('/')
def home():
    return "Welcome to the Product Recommendation API!"


@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the product name input from the user's request
    user_input = request.json.get('input', '')

    # Get the product name input from the user's request
    recommendations = recommend_companies(user_input, scores_ranks_df)

    # Return the recommendation results in JSON format
    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)
