import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载产品-行业映射表
product_industry_df = pd.read_csv('./data/product_industry.csv')

# 加载预训练的BERT模型和Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)

# 将行业标签定义为列表（从数据集的 Industry (Grouped) 字段中提取的行业列表）
industries = ['Household Products', 'Electronics', 'Telecommunications', 'Food', 'Apparel', 'Automotive', 'Healthcare',
              'Financials', 'Energy', 'Technology']


def classify_product(product_name):
    """
    使用BERT模型对产品进行分类，预测其所属行业。
    """
    inputs = tokenizer(product_name, return_tensors="pt", padding="max_length", truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # 返回预测的行业名称
    return industries[predicted_class]


def map_product_to_industry(product_name):
    """
    从产品-行业映射表中查找产品对应的行业。
    """
    match = product_industry_df[product_industry_df['product'].str.contains(product_name, case=False, na=False)]
    if not match.empty:
        return match['industry'].values[0]
    return None


def recommend_companies(user_input, df, top_n=5):
    """
    根据用户输入的产品名称，推荐对应行业的公司。

    Args:
    user_input (str): 用户输入的产品名称
    df (DataFrame): 含有企业评分和排名的数据表
    top_n (int): 推荐的公司数量，默认是5

    Returns:
    list: 推荐的公司列表
    """
    # 首先尝试从映射表中查找产品对应的行业
    industry = map_product_to_industry(user_input)

    # 如果在映射表中找不到行业，使用BERT进行分类
    if industry is None:
        industry = classify_product(user_input)

    # 根据行业筛选公司并推荐
    filtered_df = df[df['Industry_Grouped'].str.contains(industry, case=False, na=False)]

    if filtered_df.empty:
        return {"message": f"No companies found for the industry: {industry}."}

    # 按照总分排序并返回前N个公司
    top_companies = filtered_df.nlargest(top_n, 'Total Score \n(out of 100)')
    recommendations = top_companies[['Company Name', 'Total Score \n(out of 100)', 'Total Rank']].to_dict(
        orient='records')

    return recommendations
