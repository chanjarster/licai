import pandas as pd
import numpy
from IPython.display import HTML, display
from sklearn import feature_selection
from scipy import stats


def calc_dim_correlations(data_dir, indicators, from_date):
    """
    计算每个指数的各个维度上和价格的相关系数

    :param data_dir: 数据目录
    :param indicators: 指数名字（就是文件名去掉xls后缀）
    :param from_date: 取哪一天之后（含）的数据

    Return: DataFrame,
        index: 指数名字
        columns: 这些维度和价格的相关系数, "PE_ETF加权","PE_市值加权","PE_等权","PB_ETF加权","PB_市值加权","PB_等权","股息收益率 %","ROE %"
    """

    # 考察的维度
    dim_cols = [
        "PE_ETF加权",
        "PE_市值加权",
        "PE_等权",
        "PB_ETF加权",
        "PB_市值加权",
        "PB_等权",
        "股息收益率 %",
        "ROE %",
    ]

    correlation_coefficients_result = []

    for indicator in indicators:
        df = pd.read_excel(f"../{data_dir}/{indicator}.xls")
        # 只取大于某日期的数据
        df = df.loc[lambda d: d["日期"] >= from_date, :]

        # 取特征
        features = df[dim_cols]
        # 取y值
        price = df["收盘价"].values

        correlation_coefficients = feature_selection.r_regression(features, price)
        # 最后一行的日期
        correlation_coefficients = numpy.append(
            correlation_coefficients, df.tail(1)["日期"]
        )
        correlation_coefficients_result.append(correlation_coefficients)

    dim_cols.append("最早数据日期")

    result = pd.DataFrame(
        data=correlation_coefficients_result, columns=dim_cols, index=indicators
    )
    return result
