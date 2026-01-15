import pandas as pd
import numpy
from IPython.display import HTML, display
from sklearn import feature_selection
from scipy import stats


def calc_dim_correlations(data_dir, stock_list, from_date):
    """
    计算每个股票的各个维度上和价格的相关系数

    :param data_dir: 数据目录
    :param stock_list: 股票名字（就是文件名去掉xls后缀）
    :param from_date: 取哪一天之后（含）的数据

    Return: DataFrame,
        index: 股票名字
        columns: 这些维度和价格的相关系数, "滚动市盈率 - pe_ttm","滚动扣非市盈率 - koufei_pe_ttm","滚动市净率 - pb_ttm","动态市销率 - ps_ttm","动态股息收益率 % - dividend_yield","ROE %"
    """

    # 考察的维度
    dim_cols = [
        "滚动市盈率 - pe_ttm",
        "滚动扣非市盈率 - koufei_pe_ttm",
        "滚动市净率 - pb_ttm",
        "动态市销率 - ps_ttm",
        "动态股息收益率 % - dividend_yield",
        "ROE %",
    ]

    date_column = "日期 - date"
    price_column = "收盘价(前复权) - close price"

    correlation_coefficients_result = []

    for stock in stock_list:
        df = pd.read_csv(f"../{data_dir}/{stock}.csv")
        # 只取大于某日期的数据
        df = df.loc[lambda d: d[date_column] >= from_date, :]

        # 取特征
        features = df[dim_cols]
        # 取y值
        price = df[price_column].values

        correlation_coefficients = feature_selection.r_regression(features, price)
        # 最后一行的日期
        correlation_coefficients = numpy.append(
            correlation_coefficients, df.tail(1)[date_column]
        )
        correlation_coefficients_result.append(correlation_coefficients)

    dim_cols.append("最早数据日期")
    result = pd.DataFrame(
        data=correlation_coefficients_result, columns=dim_cols, index=stock_list
    )
    return result
