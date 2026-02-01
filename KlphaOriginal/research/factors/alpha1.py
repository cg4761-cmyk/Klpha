"""
Alpha#1 因子实现
公式: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)

解析：
1. returns < 0: 判断收益率是否为负
2. stddev(returns, 20): 计算20日收益率的标准差
3. close: 收盘价
4. (returns < 0) ? stddev(returns, 20) : close: 如果收益率为负，用20日标准差，否则用收盘价
5. SignedPower(..., 2.): 对上述结果取平方（带符号的幂）
6. Ts_ArgMax(..., 5): 时间序列ArgMax，找到过去5天内最大值的索引位置（距离当前的天数）
7. rank(...): 横截面排名（0-1分位数）
8. - 0.5: 减去0.5，将排名从[0,1]转换为[-0.5, 0.5]
"""
import numpy as np
import pandas as pd
from research.factors.base import BaseFactor
from typing import Dict


def signed_power(x: pd.DataFrame, power: float) -> pd.DataFrame:
    """
    带符号的幂运算
    保留原始符号，然后取绝对值幂
    
    Parameters:
    -----------
    x : pd.DataFrame
        输入数据
    power : float
        幂次
    
    Returns:
    --------
    pd.DataFrame
        结果：sign(x) * |x|^power
    """
    # 获取符号
    sign = np.sign(x)
    # 计算绝对值幂
    abs_power = np.abs(x) ** power
    # 应用符号
    result = sign * abs_power
    return pd.DataFrame(result, index=x.index, columns=x.columns)


def ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    时间序列 ArgMax（向量化优化版本）
    找到过去 window 天内最大值的索引位置（距离当前的天数）
    
    例如：如果过去5天的值为 [1, 3, 2, 5, 4]，最大值是5（索引3），
    则返回 3（表示最大值出现在3天前）
    
    Parameters:
    -----------
    x : pd.DataFrame
        输入数据，行=日期，列=股票代码
    window : int
        滚动窗口大小
    
    Returns:
    --------
    pd.DataFrame
        每个位置的值表示：过去window天内最大值出现在多少天前
        值越大，表示最大值出现得越早
    """
    result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    
    # 使用滚动窗口的 apply 方法，对每列分别处理
    for col in x.columns:
        series = x[col]
        
        def argmax_func(window_data):
            """计算窗口内最大值的位置（距离窗口末尾的天数）"""
            if len(window_data) == 0 or window_data.isna().all():
                return np.nan
            
            # 找到最大值的索引（在窗口内的位置）
            max_idx = window_data.values.argmax()
            
            # 计算距离窗口末尾的天数（0表示最大值在最后一天，window-1表示在第一天）
            days_ago = len(window_data) - 1 - max_idx
            
            return float(days_ago)
        
        # 使用滚动窗口
        result[col] = series.rolling(window=window, min_periods=1).apply(
            argmax_func, raw=False
        )
    
    return result


class Alpha1Factor(BaseFactor):
    """
    Alpha#1 因子
    
    公式: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
    
    因子逻辑：
    - 当收益率为负时，使用20日收益率标准差（衡量波动性）
    - 当收益率为正时，使用收盘价
    - 对结果取平方（带符号）
    - 找到过去5天内最大值的索引位置
    - 进行横截面排名并减去0.5
    """
    
    def __init__(self, stddev_period: int = 20, argmax_window: int = 5, power: float = 2.0):
        """
        Parameters:
        -----------
        stddev_period : int
            计算标准差的时间窗口（默认20天）
        argmax_window : int
            ArgMax的时间窗口（默认5天）
        power : float
            SignedPower的幂次（默认2.0，即平方）
        """
        super().__init__(
            name="alpha1",
            params={
                "stddev_period": stddev_period,
                "argmax_window": argmax_window,
                "power": power
            }
        )
        self.stddev_period = stddev_period
        self.argmax_window = argmax_window
        self.power = power
    
    def calculate(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算 Alpha#1 因子值
        
        Parameters:
        -----------
        data : dict
            数据字典，必须包含 'close'
        
        Returns:
        --------
        pd.DataFrame
            因子值，行=日期，列=股票代码
        """
        if "close" not in data:
            raise ValueError("数据中必须包含 'close' 价格数据")
        
        close_df = data['close']
        
        # ========== 1. 计算收益率 ==========
        returns = close_df.pct_change()
        
        # ========== 2. 计算20日收益率标准差 ==========
        stddev_returns = returns.rolling(window=self.stddev_period, min_periods=1).std()
        
        # ========== 3. 条件选择：如果收益率为负，用标准差，否则用收盘价 ==========
        # 创建条件掩码
        negative_returns_mask = returns < 0
        
        # 使用 np.where 进行向量化条件选择
        conditional_value = pd.DataFrame(
            np.where(negative_returns_mask.values, stddev_returns.values, close_df.values),
            index=close_df.index,
            columns=close_df.columns
        )
        
        # ========== 4. SignedPower：带符号的幂运算 ==========
        signed_power_result = signed_power(conditional_value, self.power)
        
        # ========== 5. Ts_ArgMax：时间序列ArgMax ==========
        argmax_result = ts_argmax(signed_power_result, self.argmax_window)
        
        # ========== 6. 打破平局：添加基于股票代码的微小确定性扰动 ==========
        # Ts_ArgMax 只返回 0-4 的整数，导致很多股票有相同的值
        # 为了能够正确分成10组，添加一个基于股票代码的微小扰动来打破平局
        # 扰动是确定性的（基于列名哈希），所以不会影响因子的稳定性
        argmax_with_tiebreak = argmax_result.copy()
        
        # 为每个股票代码生成一个确定性的微小扰动值
        # 使用列名的哈希值来确保每个股票都有不同的扰动
        tiebreak_values = {}
        for col in argmax_result.columns:
            # 使用列名的哈希值生成一个在 [0, 1e-10) 范围内的扰动
            hash_val = hash(str(col)) % 1000000  # 取模确保值不会太大
            tiebreak_values[col] = hash_val * 1e-12
        
        # 将扰动添加到整个列（对每一行都应用相同的扰动）
        for col in argmax_result.columns:
            argmax_with_tiebreak[col] = argmax_result[col] + tiebreak_values[col]
        
        # ========== 7. 横截面排名（0-1分位数） ==========
        # 按行（日期）进行排名，转换为0-1分位数
        # 使用 method='first' 来进一步确保唯一排名
        rank_result = argmax_with_tiebreak.rank(axis=1, pct=True, method='first')
        
        # ========== 8. 减去0.5 ==========
        factor_df = rank_result - 0.5
        
        # 验证和清洗
        factor_df = self.validate(factor_df)
        
        return factor_df

