"""
因子基类
定义因子计算的标准接口
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Optional


class BaseFactor(ABC):
    """因子基类"""
    
    def __init__(self, name: str, params: Optional[Dict] = None):
        """
        Parameters:
        -----------
        name : str
            因子名称
        params : dict
            因子参数
        """
        self.name = name
        self.params = params or {}
    
    @abstractmethod
    def calculate(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算因子值（子类必须实现）
        
        Parameters:
        -----------
        data : dict
            数据字典，包含 {"close": DataFrame, "open": DataFrame, ...}
            每个 DataFrame：行=日期，列=股票代码
        
        Returns:
        --------
        pd.DataFrame
            因子值，行=日期，列=股票代码
        """
        raise NotImplementedError
    
    def validate(self, factor_values: pd.DataFrame) -> pd.DataFrame:
        """
        验证和清洗因子值
        
        Parameters:
        -----------
        factor_values : pd.DataFrame
            原始因子值
        
        Returns:
        --------
        pd.DataFrame
            清洗后的因子值
        """
        result = factor_values.copy()
        
        # 处理无穷大和无穷小
        result = result.replace([np.inf, -np.inf], np.nan)
        
        return result
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"

