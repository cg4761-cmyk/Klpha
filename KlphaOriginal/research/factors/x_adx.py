"""
X-ADX (X - Average Directional Index) 趋势因子
基于趋势强度和方向性的横截面因子（完全向量化版本）
"""
import numpy as np
import pandas as pd
from research.factors.base import BaseFactor


def wilder_ema_vectorized(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    向量化 Wilder EMA 平滑处理
    
    使用 pandas ewm 近似 Wilder's Smoothing:
    Wilder EMA ≈ ewm(alpha=1/period, adjust=False)
    
    Parameters:
    -----------
    df : pd.DataFrame
        输入数据，行=日期，列=股票代码
    period : int
        平滑周期
    
    Returns:
    --------
    pd.DataFrame
        平滑后的数据，保持原始 NaN 位置
    """
    # Wilder's Smoothing: alpha = 1/period
    alpha = 1.0 / period
    
    # 使用 ewm 进行向量化平滑
    # adjust=False 确保使用递归公式：EMA_t = alpha * X_t + (1-alpha) * EMA_{t-1}
    smoothed = df.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    
    return smoothed


class XADXFactor(BaseFactor):
    """
    X-ADX 趋势因子（改良版 ADX，完全向量化）
    
    计算步骤（完全向量化）：
    1. 计算上升动量 UpMove 和下降动量 DownMove（向量化）
    2. 计算 +DM 和 -DM（趋向移动，向量化）
    3. 计算真实波幅 TR（True Range，向量化）
    4. 使用向量化 Wilder EMA 平滑处理：ATR、+DM、-DM
    5. 计算 +DI 和 -DI（Directional Indicator，向量化）
    6. 计算 DX（Directional Index，向量化）
    7. 计算 X-ADX（Average Directional Index，向量化）
    8. 使用 Sigmoid 软门控平滑加权（替代硬阈值）
    9. 趋势方向：+DI > -DI 为上涨趋势（正值），否则为下跌（负值）
    10. 计算最终带符号的因子值：方向 × 软加权趋势强度
    """
    
    def __init__(self, period: int = 14, min_adx: float = 25, max_adx: float = 75, sigmoid_scale: float = 5.0):
        """
        Parameters:
        -----------
        period : int
            X-ADX 计算周期（默认14天）
        min_adx : float
            Sigmoid 阈值（默认25），ADX 低于此值会被平滑衰减
        max_adx : float
            最大 X-ADX 值（默认75，用于参考，不用于硬截断）
        sigmoid_scale : float
            Sigmoid 函数的尺度参数（默认5.0），控制软门控的陡峭程度
            较小的值 = 更陡峭的过渡，较大的值 = 更平滑的过渡
        """
        super().__init__(
            name="x-adx",
            params={"period": period, "min_adx": min_adx, "max_adx": max_adx, "sigmoid_scale": sigmoid_scale}
        )
        self.period = period
        self.min_adx = min_adx
        self.max_adx = max_adx
        self.sigmoid_scale = sigmoid_scale
    
    def _sigmoid_weight(self, adx: pd.DataFrame) -> pd.DataFrame:
        """
        Sigmoid 软门控函数：平滑地加权 ADX 值
        
        使用 Sigmoid 函数替代硬阈值，避免信号突变：
        weight = 1 / (1 + exp(-(adx - threshold) / scale))
        
        当 ADX 从 15 上升到 30 时，权重从 ~0.12 平滑过渡到 ~0.88，
        而不是从 0 突然跳到 1（硬阈值的情况）。
        
        这减少了信号"whipsaws"（频繁反转），提高了策略稳定性。
        
        Parameters:
        -----------
        adx : pd.DataFrame
            ADX 值，行=日期，列=股票代码
        
        Returns:
        --------
        pd.DataFrame
            Sigmoid 权重，范围 [0, 1]
        """
        # Sigmoid 函数：weight = 1 / (1 + exp(-(x - threshold) / scale))
        # 当 adx = threshold 时，weight = 0.5
        # 当 adx >> threshold 时，weight → 1
        # 当 adx << threshold 时，weight → 0
        
        # 计算 (adx - threshold) / scale
        normalized = (adx - self.min_adx) / self.sigmoid_scale
        
        # 应用 Sigmoid 函数
        weight = 1.0 / (1.0 + np.exp(-normalized))
        
        # 保持原始 NaN 位置
        weight = weight.where(~adx.isna())
        
        return weight
    
    def calculate(self, data: dict) -> pd.DataFrame:
        """
        计算 X-ADX 因子（完全向量化版本）
        
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
        
        # ========== 1. 计算上升动量 UpMove 和下降动量 DownMove（向量化）==========
        up_move = high_df.diff()                    # 当前最高价 - 前一最高价（所有股票同时计算）
        down_move = -low_df.diff()                  # 前一最低价 - 当前最低价（加负号使其为正）
        
        # ========== 2. 计算 +DM 和 -DM（趋向移动，向量化）==========
        # 使用向量化条件选择
        # 条件：UpMove > DownMove 且 UpMove > 0 时，取 UpMove；否则为 0
        plus_dm_mask = (up_move > down_move) & (up_move > 0)
        plus_dm = pd.DataFrame(
            np.where(plus_dm_mask.values, up_move.values, 0.0),
            index=close_df.index,
            columns=close_df.columns
        )
        
        # 条件：DownMove > UpMove 且 DownMove > 0 时，取 DownMove；否则为 0
        minus_dm_mask = (down_move > up_move) & (down_move > 0)
        minus_dm = pd.DataFrame(
            np.where(minus_dm_mask.values, down_move.values, 0.0),
            index=close_df.index,
            columns=close_df.columns
        )
        
        # ========== 3. 计算真实波幅 TR（True Range，向量化）==========
        # 三种方式中取最大值，使用 np.maximum 进行向量化
        tr1 = high_df - low_df                              # 当前最高价 - 当前最低价
        tr2 = (high_df - close_df.shift(1)).abs()           # 当前最高价 - 昨日收盘价
        tr3 = (low_df - close_df.shift(1)).abs()            # 当前最低价 - 昨日收盘价
        
        # 使用 np.maximum 进行向量化取最大值
        tr = pd.DataFrame(
            np.maximum(np.maximum(tr1.values, tr2.values), tr3.values),
            index=close_df.index,
            columns=close_df.columns
        )
        
        # ========== 4. 使用向量化 Wilder EMA 平滑处理：ATR、+DM、-DM ==========
        atr = wilder_ema_vectorized(tr, self.period)                      # 平滑后的真实波幅
        plus_dm_smoothed = wilder_ema_vectorized(plus_dm, self.period)    # 平滑后的 +DM
        minus_dm_smoothed = wilder_ema_vectorized(minus_dm, self.period)  # 平滑后的 -DM
        
        # ========== 5. 计算 +DI 和 -DI（Directional Indicator，向量化）==========
        # 避免除零：使用 np.where 处理
        di_denominator = np.where(atr.values > 0, atr.values, np.nan)
        atr_safe = pd.DataFrame(di_denominator, index=atr.index, columns=atr.columns)
        
        plus_di = 100.0 * plus_dm_smoothed / atr_safe            # +DI = 100 * S(+DM) / ATR
        minus_di = 100.0 * minus_dm_smoothed / atr_safe          # -DI = 100 * S(-DM) / ATR
        
        # ========== 6. 计算 DX（Directional Index，向量化）==========
        # DX = 100 * |+DI - -DI| / (+DI + -DI)
        di_sum = plus_di + minus_di
        
        # 避免除零：使用 np.where
        dx_numerator = (plus_di - minus_di).abs()
        dx_denominator = np.where(di_sum.values > 0, di_sum.values, np.nan)
        di_sum_safe = pd.DataFrame(dx_denominator, index=di_sum.index, columns=di_sum.columns)
        
        dx = 100.0 * dx_numerator / di_sum_safe
        dx = dx.replace([np.inf, -np.inf], np.nan)
        
        # ========== 7. 计算 X-ADX（Average Directional Index，向量化）==========
        adx = wilder_ema_vectorized(dx, self.period)
        
        # ========== 8. 使用 Sigmoid 软门控平滑加权（替代硬阈值）==========
        # 使用 Sigmoid 函数平滑地加权 ADX，而不是硬截断
        # 这避免了信号突变，减少了"whipsaws"
        sigmoid_weight = self._sigmoid_weight(adx)
        adx_weighted = adx * sigmoid_weight
        
        # ========== 9. 趋势方向（向量化）==========
        # +DI > -DI 为上涨趋势（正值），否则为下跌（负值）
        direction = np.sign(plus_di - minus_di)
        
        # ========== 10. 计算最终带符号的因子值（向量化）==========
        # 最终因子 = 软加权 ADX × 方向
        factor_df = adx_weighted * direction
        
        # 验证和清洗
        factor_df = self.validate(factor_df)
        
        return factor_df

