"""
横截面信号处理工具
提供横截面标准化、排名、仓位调整等功能
"""
import numpy as np
import pandas as pd
from typing import Optional
from numba import njit, prange


@njit(parallel=True)
def cs_booksize_nb(
    data: np.ndarray, 
    size: float = 1.0, 
    upper_bound: float = 0.1, 
    lower_bound: float = 0.0
) -> np.ndarray:
    """
    横截面仓位标准化（只做多策略）
    确保每个时间点的总仓位大小固定为1.0，并控制单只股票上限
    所有负值仓位将被设为0（只做多）
    
    Parameters:
    -----------
    data : np.ndarray
        2D数组，行=时间，列=股票，值=原始仓位
    size : float
        目标总仓位大小（归一化，1.0表示100%）
    upper_bound : float
        单只股票最大仓位（按size的比例）
    lower_bound : float
        单只股票最小仓位阈值（低于此值设为0）
    
    Returns:
    --------
    np.ndarray
        标准化后的仓位数组（只包含非负值，总和为size）
    """
    result = data.copy()
    
    for i in prange(data.shape[0]):
        tmp = result[i, :].copy()
        
        # 只做多：将所有负值设为0
        tmp[tmp < 0] = 0.0
        
        # 去除低于阈值的仓位
        tmp[tmp < lower_bound] = 0.0
        
        # 计算多仓总和
        positive_sum = np.nansum(tmp)
        
        # 标准化多仓，使总仓位为size
        if positive_sum > 0:
            tmp = tmp * (size / positive_sum)
        else:
            # 如果没有正仓位，全部设为0
            tmp[:] = 0.0
            result[i, :] = tmp
            continue
        
        # 应用上限约束（单只股票最大仓位）
        # 迭代方式：先应用上限，然后重新归一化，直到收敛
        max_iter = 10
        for _ in range(max_iter):
            if not np.any(tmp > size * upper_bound):
                break
            
            # 将超过上限的仓位设为上限
            tmp[tmp > size * upper_bound] = size * upper_bound
            
            # 计算已分配的仓位总和
            allocated = np.nansum(tmp[tmp >= size * upper_bound])
            remaining = size - allocated
            
            # 计算未达到上限的仓位总和
            below_bound_sum = np.nansum(tmp[tmp < size * upper_bound])
            
            if below_bound_sum > 0 and remaining > 0:
                # 将剩余仓位按比例分配给未达到上限的股票
                tmp[tmp < size * upper_bound] = tmp[tmp < size * upper_bound] * (remaining / below_bound_sum)
            elif remaining > 0:
                # 如果所有股票都达到上限，但还有剩余仓位，说明上限设置不合理
                # 这种情况下，我们按比例增加所有仓位（虽然会超过上限，但至少保证总和为size）
                current_sum = np.nansum(tmp)
                if current_sum > 0:
                    tmp = tmp * (size / current_sum)
                break
            
            # 最终归一化，确保总和为size（处理浮点误差）
            current_sum = np.nansum(tmp)
            if current_sum > 0:
                tmp = tmp * (size / current_sum)
        
        result[i, :] = tmp
    
    return result


def normalize_positions(
    positions: pd.DataFrame,
    booksize: float = 1.0,
    upper_bound: float = 0.1,
    lower_bound: float = 0.0
) -> pd.DataFrame:
    """
    横截面仓位标准化（DataFrame接口，只做多策略）
    
    注意：此函数实现只做多策略，所有负值仓位将被设为0。
    总仓位固定为 booksize（默认1.0，即100%）。
    
    Parameters:
    -----------
    positions : pd.DataFrame
        仓位数据，行=日期，列=股票代码（负值将被设为0）
    booksize : float
        目标总仓位大小（默认1.0，即100%）
    upper_bound : float
        单只股票最大仓位（按booksize的比例）
    lower_bound : float
        单只股票最小仓位阈值（低于此值设为0）
    
    Returns:
    --------
    pd.DataFrame
        标准化后的仓位（只包含非负值，每行总和为booksize）
    """
    values = positions.values.copy()
    normalized = cs_booksize_nb(values, booksize, upper_bound, lower_bound)
    
    return pd.DataFrame(
        normalized,
        index=positions.index,
        columns=positions.columns
    )


def cross_section_zscore(factor_values: pd.DataFrame) -> pd.DataFrame:
    """
    横截面 Z-score 标准化
    每个时间点，对所有股票的因子值进行标准化
    
    Parameters:
    -----------
    factor_values : pd.DataFrame
        因子值，行=日期，列=股票代码
    
    Returns:
    --------
    pd.DataFrame
        标准化后的因子值
    """
    # 按行（日期）标准化
    zscore = factor_values.sub(factor_values.mean(axis=1), axis=0) \
                          .div(factor_values.std(axis=1), axis=0)
    return zscore


def cross_section_rank(
    factor_values: pd.DataFrame,
    method: str = 'percentile'
) -> pd.DataFrame:
    """
    横截面排名
    
    Parameters:
    -----------
    factor_values : pd.DataFrame
        因子值，行=日期，列=股票代码
    method : str
        'percentile' (0-1分位数) 或 'ordinal' (1-N排名)
    
    Returns:
    --------
    pd.DataFrame
        排名结果
    """
    if method == 'percentile':
        return factor_values.rank(axis=1, pct=True)
    else:
        return factor_values.rank(axis=1)


def cross_section_linear(factor_values: pd.DataFrame) -> pd.DataFrame:
    """
    横截面线性标准化（Min-Max 标准化）
    将每个时间点的因子值标准化到 [0, 1] 区间
    
    Parameters:
    -----------
    factor_values : pd.DataFrame
        因子值，行=日期，列=股票代码
    
    Returns:
    --------
    pd.DataFrame
        标准化后的因子值（0-1之间）
    """
    result = factor_values.copy()
    
    for date in factor_values.index:
        row = factor_values.loc[date].dropna()
        if len(row) == 0:
            continue
        
        min_val = row.min()
        max_val = row.max()
        
        if max_val != min_val:
            normalized = (row - min_val) / (max_val - min_val)
        else:
            # 如果所有值相同，设为0.5
            normalized = pd.Series(0.5, index=row.index)
        
        result.loc[date, normalized.index] = normalized
    
    return result


def cross_section_log(factor_values: pd.DataFrame) -> pd.DataFrame:
    """
    横截面对数标准化
    适用于正偏分布的因子值（如市值、成交量）
    
    Parameters:
    -----------
    factor_values : pd.DataFrame
        因子值，行=日期，列=股票代码
    
    Returns:
    --------
    pd.DataFrame
        标准化后的因子值
    """
    result = factor_values.copy()
    
    for date in factor_values.index:
        row = factor_values.loc[date].dropna()
        if len(row) == 0:
            continue
        
        # 先平移确保所有值为正
        min_val = row.min()
        if min_val <= 0:
            row = row - min_val + 1
        
        # 取对数（使用 log1p 更稳定）
        log_row = np.log1p(row)
        
        # 再标准化（Z-score）
        mean_log = log_row.mean()
        std_log = log_row.std()
        
        if std_log > 0:
            normalized = (log_row - mean_log) / std_log
        else:
            normalized = pd.Series(0.0, index=row.index)
        
        result.loc[date, normalized.index] = normalized
    
    return result


def cross_section_sin(factor_values: pd.DataFrame) -> pd.DataFrame:
    """
    横截面正弦标准化
    将因子值映射到正弦函数，适用于周期性因子或需要非线性变换
    
    Parameters:
    -----------
    factor_values : pd.DataFrame
        因子值，行=日期，列=股票代码
    
    Returns:
    --------
    pd.DataFrame
        标准化后的因子值（-1 到 1 之间）
    """
    result = factor_values.copy()
    
    for date in factor_values.index:
        row = factor_values.loc[date].dropna()
        if len(row) == 0:
            continue
        
        # 先线性标准化到 [0, 1]
        min_val = row.min()
        max_val = row.max()
        
        if max_val != min_val:
            normalized = (row - min_val) / (max_val - min_val)
        else:
            normalized = pd.Series(0.5, index=row.index)
        
        # 映射到 [-π, π] 然后取 sin
        sin_values = np.sin(normalized * 2 * np.pi - np.pi)
        result.loc[date, sin_values.index] = sin_values
    
    return result


def cross_section_tanh(factor_values: pd.DataFrame) -> pd.DataFrame:
    """
    横截面 Tanh 标准化
    使用双曲正切函数进行标准化，结果在 [-1, 1] 之间
    
    Parameters:
    -----------
    factor_values : pd.DataFrame
        因子值，行=日期，列=股票代码
    
    Returns:
    --------
    pd.DataFrame
        标准化后的因子值（-1 到 1 之间）
    """
    result = factor_values.copy()
    
    for date in factor_values.index:
        row = factor_values.loc[date].dropna()
        if len(row) == 0:
            continue
        
        # 先 Z-score 标准化
        mean_val = row.mean()
        std_val = row.std()
        
        if std_val > 0:
            zscore = (row - mean_val) / std_val
            # 应用 tanh（可以缩放以控制范围）
            tanh_values = np.tanh(zscore)
        else:
            tanh_values = pd.Series(0.0, index=row.index)
        
        result.loc[date, tanh_values.index] = tanh_values
    
    return result


def cross_section_sigmoid(factor_values: pd.DataFrame) -> pd.DataFrame:
    """
    横截面 Sigmoid 标准化
    使用 sigmoid 函数进行标准化，结果在 [0, 1] 之间
    
    Parameters:
    -----------
    factor_values : pd.DataFrame
        因子值，行=日期，列=股票代码
    
    Returns:
    --------
    pd.DataFrame
        标准化后的因子值（0 到 1 之间）
    """
    result = factor_values.copy()
    
    for date in factor_values.index:
        row = factor_values.loc[date].dropna()
        if len(row) == 0:
            continue
        
        # 先 Z-score 标准化
        mean_val = row.mean()
        std_val = row.std()
        
        if std_val > 0:
            zscore = (row - mean_val) / std_val
            # 应用 sigmoid: 1 / (1 + exp(-x))
            sigmoid_values = 1 / (1 + np.exp(-zscore))
        else:
            sigmoid_values = pd.Series(0.5, index=row.index)
        
        result.loc[date, sigmoid_values.index] = sigmoid_values
    
    return result


def cross_section_robust(factor_values: pd.DataFrame) -> pd.DataFrame:
    """
    横截面稳健标准化（Robust Scaling）
    使用中位数和 IQR（四分位距）进行标准化，对异常值更稳健
    
    Parameters:
    -----------
    factor_values : pd.DataFrame
        因子值，行=日期，列=股票代码
    
    Returns:
    --------
    pd.DataFrame
        标准化后的因子值
    """
    result = factor_values.copy()
    
    for date in factor_values.index:
        row = factor_values.loc[date].dropna()
        if len(row) == 0:
            continue
        
        median_val = row.median()
        q75 = row.quantile(0.75)
        q25 = row.quantile(0.25)
        iqr = q75 - q25
        
        if iqr > 0:
            # 使用中位数和 IQR 标准化
            normalized = (row - median_val) / iqr
        else:
            # 如果 IQR 为 0，使用中位数标准化
            if median_val != 0:
                normalized = (row - median_val) / abs(median_val)
            else:
                normalized = pd.Series(0.0, index=row.index)
        
        result.loc[date, normalized.index] = normalized
    
    return result


def normalize_factor(
    factor_values: pd.DataFrame,
    method: str = 'zscore'
) -> pd.DataFrame:
    """
    统一的因子标准化函数，支持多种标准化方法
    
    Parameters:
    -----------
    factor_values : pd.DataFrame
        因子值，行=日期，列=股票代码
    method : str
        标准化方法，可选：
        - 'zscore': Z-score 标准化（默认）
        - 'rank': 排名标准化（0-1分位数）
        - 'linear': 线性标准化（Min-Max，0-1）
        - 'log': 对数标准化
        - 'sin': 正弦标准化
        - 'tanh': Tanh 标准化
        - 'sigmoid': Sigmoid 标准化
        - 'robust': 稳健标准化（中位数和IQR）
    
    Returns:
    --------
    pd.DataFrame
        标准化后的因子值
    """
    method_map = {
        'zscore': cross_section_zscore,
        'rank': lambda x: cross_section_rank(x, method='percentile'),
        'linear': cross_section_linear,
        'log': cross_section_log,
        'sin': cross_section_sin,
        'tanh': cross_section_tanh,
        'sigmoid': cross_section_sigmoid,
        'robust': cross_section_robust
    }
    
    if method not in method_map:
        raise ValueError(
            f"未知的标准化方法: {method}. "
            f"可选方法: {list(method_map.keys())}"
        )
    
    return method_map[method](factor_values)


def winsorize(
    factor_values: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99
) -> pd.DataFrame:
    """
    去极值处理（Winsorize）
    将超出分位数的值截断到分位数
    
    Parameters:
    -----------
    factor_values : pd.DataFrame
        因子值
    lower : float
        下分位数（如0.01表示1%分位数）
    upper : float
        上分位数（如0.99表示99%分位数）
    
    Returns:
    --------
    pd.DataFrame
        去极值后的因子值
    """
    result = factor_values.copy()
    
    for date in factor_values.index:
        row = factor_values.loc[date]
        lower_bound = row.quantile(lower)
        upper_bound = row.quantile(upper)
        
        result.loc[date] = row.clip(lower=lower_bound, upper=upper_bound)
    
    return result


def fill_na_with_cross_section(
    factor_values: pd.DataFrame,
    method: str = 'median'
) -> pd.DataFrame:
    """
    用横截面统计量填充缺失值
    
    Parameters:
    -----------
    factor_values : pd.DataFrame
        因子值
    method : str
        'median', 'mean', 或 'zero'
    
    Returns:
    --------
    pd.DataFrame
        填充后的因子值
    """
    result = factor_values.copy()
    
    for date in factor_values.index:
        row = factor_values.loc[date]
        
        if method == 'median':
            fill_value = row.median()
        elif method == 'mean':
            fill_value = row.mean()
        else:
            fill_value = 0.0
        
        result.loc[date] = row.fillna(fill_value)
    
    return result

