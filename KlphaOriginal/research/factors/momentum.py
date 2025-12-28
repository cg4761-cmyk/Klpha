"""
动量因子示例
作为因子实现的模板
"""
import pandas as pd
from typing import Dict
from .base import BaseFactor


class MomentumFactor(BaseFactor):
    """动量因子 - 过去N日收益率"""
    
    def __init__(self, lookback: int = 20):
        """
        Parameters:
        -----------
        lookback : int
            回看窗口（交易日数）
        """
        super().__init__(
            name="momentum",
            params={"lookback": lookback}
        )
        self.lookback = lookback
    
    def calculate(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算动量因子（过去N日收益率）
        
        Parameters:
        -----------
        data : dict
            数据字典，必须包含 "close" 键
        
        Returns:
        --------
        pd.DataFrame
            因子值，行=日期，列=股票代码
        """
        if "close" not in data:
            raise ValueError("数据中必须包含 'close' 价格数据")
        
        close_df = data["close"]
        
        # 计算过去N日收益率
        returns = close_df.pct_change(self.lookback)
        
        # 验证和清洗
        factor_values = self.validate(returns)
        
        return factor_values

