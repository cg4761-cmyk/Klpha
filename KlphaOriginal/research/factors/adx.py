"""
ADX (Average Directional Index) 趋势因子
基于趋势强度和方向性的横截面因子
"""
import numpy as np
import pandas as pd
from research.factors.base import BaseFactor


def wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder EMA 平滑处理
    平滑版本的 Wilder EMA，trend carry + 原始NaN对齐
    """
    values = series.values.copy()
    original_nan_mask = np.isnan(values)  # 记录原始NaN位置
    result = np.full_like(values, np.nan, dtype=np.float64)
    valid_values = []
    
    for i, val in enumerate(values):
        if not np.isnan(val):
            valid_values.append(val)
            if len(valid_values) == period:
                result[i] = np.mean(valid_values)
                
                # 从这里开始，所有位置都用平滑填充（包括NaN）
                for j in range(i + 1, len(values)):
                    if np.isnan(values[j]):
                        result[j] = result[j - 1]  # 平滑延续
                    else:
                        result[j] = (result[j - 1] * (period - 1) + values[j]) / period
                break
    
    # 最后对齐：只有原始数据中的NaN位置才设为NaN
    result[original_nan_mask] = np.nan
    
    return pd.Series(result, index=series.index)


class ADXFactor(BaseFactor):
    """
    ADX 趋势因子
    
    计算步骤：
    1. 计算上升动量 UpMove 和下降动量 DownMove
    2. 计算 +DM 和 -DM（趋向移动）
    3. 计算真实波幅 TR（True Range）
    4. 使用 Wilder EMA 平滑处理：ATR、+DM、-DM
    5. 计算 +DI 和 -DI（Directional Indicator）
    6. 计算 DX（Directional Index）
    7. 计算 ADX（Average Directional Index）
    8. 筛选 ADX ∈ [25, 75]：过滤弱趋势或异常强趋势
    9. 趋势方向：+DI > -DI 为上涨趋势（正值），否则为下跌（负值）
    10. 计算最终带符号的因子值：方向 × 趋势强度
    """
    
    def __init__(self, period: int = 14, min_adx: float = 25, max_adx: float = 75):
        """
        Parameters:
        -----------
        period : int
            ADX 计算周期（默认14天）
        min_adx : float
            最小 ADX 值（默认25，过滤弱趋势）
        max_adx : float
            最大 ADX 值（默认75，过滤异常强趋势）
        """
        super().__init__(
            name="adx",
            params={"period": period, "min_adx": min_adx, "max_adx": max_adx}
        )
        self.period = period
        self.min_adx = min_adx
        self.max_adx = max_adx
    
    def calculate(self, data: dict) -> pd.DataFrame:
        """
        计算 ADX 因子
        
        Parameters:
        -----------
        data : dict
            数据字典，必须包含 'close', 'high', 'low'
        
        Returns:
        --------
        pd.DataFrame
            因子值，行=日期，列=股票代码
        """
        if "close" not in data or "high" not in data or "low" not in data:
            raise ValueError("数据中必须包含 'close', 'high', 'low' 价格数据")
        
        close_df = data['close']
        high_df = data['high']
        low_df = data['low']
        
        # 获取所有的股票代码
        symbols = close_df.columns
        
        # 创建空的 DataFrame 用于存放最终的 ADX 因子值
        factor_df = pd.DataFrame(index=close_df.index, columns=symbols, dtype=np.float64)
        
        # 针对每一个股票分别计算 ADX
        for sym in symbols:
            close = close_df[sym]
            high = high_df[sym]
            low = low_df[sym]
            
            # 1. 计算上升动量 UpMove 和下降动量 DownMove
            up_move = high.diff()                   # 当前最高价 - 前一最高价
            down_move = -low.diff()                 # 前一最低价 - 当前最低价（加负号使其为正）
            
            # 2. 计算 +DM 和 -DM（趋向移动）
            plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move         # 若 UpMove > DownMove 且 UpMove > 0，取 UpMove；否则为 0
            minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move    # 同理，计算 DownMove
            
            # 3. 计算真实波幅 TR（True Range）：三种方式中取最大值
            tr1 = high - low                              # 当前最高价 - 当前最低价
            tr2 = (high - close.shift()).abs()            # 当前最高价 - 昨日收盘价
            tr3 = (low - close.shift()).abs()             # 当前最低价 - 昨日收盘价
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)  # 取三者中最大值作为 TR
            
            # 4. 使用 Wilder EMA 平滑处理：ATR、+DM、-DM
            atr = wilder_ema(tr, self.period)                      # 平滑后的真实波幅
            plus_dm_smoothed = wilder_ema(plus_dm, self.period)    # 平滑后的 +DM
            minus_dm_smoothed = wilder_ema(minus_dm, self.period)   # 平滑后的 -DM
            
            # 5. 计算 +DI 和 -DI（Directional Indicator）
            plus_di = 100 * plus_dm_smoothed / atr            # +DI = 100 * S(+DM) / ATR
            minus_di = 100 * minus_dm_smoothed / atr          # -DI = 100 * S(-DM) / ATR
            
            # 6. 计算 DX（Directional Index）
            # 避免除零：当 plus_di + minus_di 为 0 时，DX 设为 NaN
            di_sum = plus_di + minus_di
            dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)   # DX = 100 * |+DI - -DI| / (+DI + -DI)
            dx = dx.replace([np.inf, -np.inf], np.nan)
        
            # 7. 计算 ADX（Average Directional Index）：DX 的平滑均值
            adx = wilder_ema(dx, self.period)
            
            # 8. 筛选 ADX ∈ [min_adx, max_adx]：过滤弱趋势或异常强趋势
            adx_filtered = adx.where((adx >= self.min_adx) & (adx <= self.max_adx), np.nan)
            
            # 9. 趋势方向：+DI > -DI 为上涨趋势（正值），否则为下跌（负值）
            direction = np.sign(plus_di - minus_di)
            
            # 10. 计算最终带符号的因子值：方向 × 趋势强度
            factor = adx_filtered * direction
            
            # 存入结果矩阵
            factor_df[sym] = factor
        
        # 验证和清洗
        factor_df = self.validate(factor_df)
        
        return factor_df

