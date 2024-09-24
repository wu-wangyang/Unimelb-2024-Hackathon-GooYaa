def recommend_companies(industry, df, top_n=5):
    """
    根据行业名称推荐评分最高的公司。

    Args:
    industry (str): 用户输入的行业名称
    df (DataFrame): 含有企业评分和排名的数据表
    top_n (int): 推荐的公司数量，默认是5

    Returns:
    list: 推荐的公司列表
    """
    # 根据行业筛选出相关的公司
    filtered_df = df[df['Industry_Disaggregated'].str.contains(industry, case=False, na=False)]

    # 如果没有找到匹配的行业，返回空列表
    if filtered_df.empty:
        return {"message": "No companies found for the specified industry."}

    # 按照总分降序排序，并获取前N个公司
    top_companies = filtered_df.nlargest(top_n, 'Total Score \n(out of 100)')

    # 构建推荐结果
    recommendations = top_companies[['Company Name', 'Total Score \n(out of 100)', 'Total Rank']].to_dict(
        orient='records')

    return recommendations
