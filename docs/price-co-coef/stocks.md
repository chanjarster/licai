# 股票的PE/PB的相关性分析

取各个股票的股价和相关指标之间的 [相关系数][1]：

|                    | 滚动市盈率 pe_ttm | 滚动扣非市盈率 koufei_pe_ttm | 滚动市净率 pb_ttm | 动态市销率 ps_ttm | 动态股息收益率 % - dividend_yield | ROE %   | 最早数据日期
|--------------------|-------------------|--------------------------|-----------------|------------------|--------------------------------|---------|------------
| sz002389_航天彩虹   | 0.4404           | 0.4094                    | 0.4893          | 0.2679           | -0.6834                       | -0.2970 | 2012-01-04
| sh601127_赛力斯    | -0.2387           | -0.2058                  | <font color="red">0.8879</font>           | 0.9721           | -0.5977                       | -0.3056 | 2016-06-15
| sz300750_宁德时代   | 0.5871           | 0.5440                    | <font color="red">0.8214</font>           | 0.5507          | -0.0141                       | 0.4145   | 2018-06-11

[1]: https://chanjarster.github.io/ai-learn/#/ai_basics/statistics?id=%e6%95%b0%e5%ad%97%e7%89%b9%e5%be%81