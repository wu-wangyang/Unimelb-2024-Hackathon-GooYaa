import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# from app import scores_ranks_df
data_path = './data/Nature_2022-2024_PublicDataset_20240718_v4.xlsx'
scores_ranks_df = pd.read_excel(data_path, sheet_name='Scores and ranks')

# Load the product-industry mapping table
product_industry_df = pd.read_csv('./data/dataset.csv')


# Extract the unique list of industries from the company score dataset
def get_unique_industries(df):
    return df['Industry_Grouped'].dropna().unique().tolist()


industries = get_unique_industries(scores_ranks_df)

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(industries))


def classify_product(product_name):
    """
    Classify the product using the BERT model to predict its associated industry.
    """
    inputs = tokenizer(product_name, return_tensors="pt", padding="max_length", truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Return the predicted industry name
    return industries[predicted_class]


def map_product_to_industry(product_name):
    """
    Find the industry corresponding to the product from the product-industry mapping table.
    """
    match = product_industry_df[product_industry_df['Product Name'].str.contains(product_name, case=False, na=False)]
    if not match.empty:
        return match['Industry'].values[0]
    return None


def recommend_companies(user_input, df, top_n=5):
    """
    Recommend companies based on the product name provided by the user.

    Args:
    user_input (str): The product name provided by the user
    df (DataFrame): The dataset containing company scores and rankings
    top_n (int): The number of companies to recommend, default is 5

    Returns:
    list: A list of recommended companies
    """
    # First try to find the corresponding industry from the mapping table
    industry = map_product_to_industry(user_input)

    # If the industry is not found in the mapping table, classify it using BERT
    if industry is None:
        industry = classify_product(user_input)

    # Filter the companies based on the identified industry and recommend them
    filtered_df = df[df['Industry_Grouped'].str.contains(industry, case=False, na=False)]

    if filtered_df.empty:
        return {"message": f"No companies found for the industry: {industry}."}

    # Sort by total score and return the top N companies
    top_companies = filtered_df.nlargest(top_n, 'Total Score \n(out of 100)')
    recommendations = top_companies[['Company Name', 'Total Score \n(out of 100)', 'Total Rank']].to_dict(
        orient='records')

    return recommendations
