from flask import Flask, request, jsonify
import pandas as pd
from recommender import recommend_companies

app = Flask(__name__)

# 读取企业数据
data_path = './data/Nature_2022-2024_PublicDataset_20240718_v4.xlsx'
scores_ranks_df = pd.read_excel(data_path, sheet_name='Scores and ranks')


@app.route('/')
def home():
    return "Welcome to the Product Recommendation API!"


@app.route('/recommend', methods=['POST'])
def recommend():
    # 从请求中获取用户输入的行业
    user_input = request.json.get('industry', '')

    # 调用推荐函数，获取推荐结果
    recommendations = recommend_companies(user_input, scores_ranks_df)

    # 返回JSON格式的推荐结果
    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)
