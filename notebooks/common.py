import pandas as pd
import numpy
from IPython.display import HTML, display
from sklearn import feature_selection
from scipy import stats


def print_title(title):
    display(HTML("<h4>{}</h4>".format(title)))


def get_highest_correlation_dim(dim_correlations):
    """
    从 dim_correlations 中提取出每行数据最相关的维度是哪个，以及它们的相关系数

    参数：
    dim_correlations : DataFrame
        calc_dim_correlations 函数的输出结果，指数的各种维度的相关系数
        index: 是指数名字
        columns: 维度1, 维度2, ... , '最早数据日期'

    返回：
    DataFrame :
        index: 是指数名字
        columns: '最相关维度', '相关系数', '最早数据日期'
    """
    value_cols = dim_correlations.columns.drop("最早数据日期")

    # 创建结果DataFrame
    result = pd.DataFrame(index=dim_correlations.index)

    # 找出每行绝对值最大的列名
    result["最相关维度"] = dim_correlations[value_cols].abs().idxmax(axis=1)

    # 获取对应的原始值（非绝对值）
    # result["列值"] = dim_correlations.lookup(result.index, result["最相关列名"])
    result["相关系数"] = result.apply(
        lambda row: dim_correlations.loc[row.name, row["最相关维度"]], axis=1
    )

    # 添加最早数据日期
    result["最早数据日期"] = dim_correlations["最早数据日期"]

    return result


def calc_highest_correlation_dim_latest_percentile(
    data_dir,
    highest_correlation_dim,
    from_date: str,
    date_col_name: str,
    is_xls: bool = True,
):
    """
    计算每个指数/股票最新值在历史数据中的百分位

    参数：
    dim_df : DataFrame
        get_highest_correlation_dim 函数的输出结果
        index: 是指数名字
        columns: '最相关维度', '相关系数', '最早数据日期'

    返回：
    DataFrame : 每个指数最新值的百分位（0-100），列名，最相关维度, 相关系数, 最早数据日期, 最新百分位

    逻辑：
    迭代每一行，用 pd.read_excel加载xls数据文件，比如 ../data-指数-260114/{index-name}.xls
        读取 DataFrame 中的第一行数据（跳过标题行）中的 dim_df 的 '最相关维度' 所指定的列名的值
        然后计算这个数据在这列中的 percentile(百分位) ，小数点后两位
    然后输出一个 pd.DateFrame：
        index: 依然是指数名字
        columns: 最相关维度, 相关系数, 最早数据日期, 最新百分位
    """

    result_df = highest_correlation_dim[
        ["最相关维度", "相关系数", "最早数据日期"]
    ].copy()
    # 初始化最新百分位列
    result_df["最新维度值"] = None
    result_df["最新百分位%"] = None
    result_df["最新数据日期"] = None

    for idx, row in highest_correlation_dim.iterrows():
        # 读取指数历史数据
        index_name = idx
        history_df = None
        if is_xls:
            history_df = pd.read_excel(f"../{data_dir}/{index_name}.xls")
        else:
            history_df = pd.read_csv(f"../{data_dir}/{index_name}.csv")

        # 只取大于某日期的数据
        history_df = history_df.loc[lambda d: d[date_col_name] >= from_date, :]

        # 获取目标维度列
        target_col = row["最相关维度"]
        latest_value = history_df[target_col][0]

        # 计算百分位
        historical_values = history_df[target_col].dropna()

        result_df.at[idx, "最新数据日期"] = history_df[date_col_name][0]
        result_df.at[idx, "最新维度值"] = latest_value
        result_df.at[idx, "最新百分位%"] = (
            historical_values <= latest_value
        ).mean() * 100
        # 下面两种算法结果是一样的
        # result_df.at[idx, "最新百分位2%"] = (historical_values <= latest_value).sum() / len(historical_values) * 100
        # result_df.at[idx, "最新百分位3%"] = stats.percentileofscore(historical_values, latest_value, kind="weak")

    return result_df
