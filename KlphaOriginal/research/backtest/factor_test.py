"""
因子测试框架
提供 IC 分析、分层回测等功能
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def calculate_ic(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = 'pearson'
) -> pd.Series:
    """
    计算信息系数（IC）
    
    Parameters:
    -----------
    factor_values : pd.DataFrame
        因子值，行=日期，列=股票代码
    forward_returns : pd.DataFrame
        未来收益，行=日期，列=股票代码
    method : str
        'pearson' (皮尔逊相关系数) 或 'spearman' (斯皮尔曼相关系数)
    
    Returns:
    --------
    pd.Series
        每日 IC 值，索引=日期
    """
    # 对齐日期和股票
    common_dates = factor_values.index.intersection(forward_returns.index)
    common_symbols = factor_values.columns.intersection(forward_returns.columns)
    
    factor_aligned = factor_values.loc[common_dates, common_symbols]
    returns_aligned = forward_returns.loc[common_dates, common_symbols]
    
    ic_series = pd.Series(index=common_dates, dtype=float)
    
    for date in common_dates:
        factor_row = factor_aligned.loc[date]
        return_row = returns_aligned.loc[date]
        
        # 去除缺失值
        valid_mask = ~(factor_row.isna() | return_row.isna())
        if valid_mask.sum() < 2:
            ic_series[date] = np.nan
            continue
        
        factor_valid = factor_row[valid_mask]
        return_valid = return_row[valid_mask]
        
        if method == 'pearson':
            ic = factor_valid.corr(return_valid)
        else:  # spearman
            ic = factor_valid.corr(return_valid, method='spearman')
        
        ic_series[date] = ic
    
    return ic_series


def calculate_ir(ic_series: pd.Series) -> float:
    """
    计算信息比率（IR = IC均值 / IC标准差）
    
    Parameters:
    -----------
    ic_series : pd.Series
        IC 序列
    
    Returns:
    --------
    float
        信息比率
    """
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    
    if ic_std == 0:
        return 0.0
    
    return ic_mean / ic_std


def factor_returns_by_quantile(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 5
) -> pd.DataFrame:
    """
    按因子值分位数计算各组收益
    
    Parameters:
    -----------
    factor_values : pd.DataFrame
        因子值
    forward_returns : pd.DataFrame
        未来收益
    n_quantiles : int
        分位数数量（如5表示分成5组）
    
    Returns:
    --------
    pd.DataFrame
        各组收益，行=日期，列=分位数组（1到n_quantiles）
    """
    # 对齐数据
    common_dates = factor_values.index.intersection(forward_returns.index)
    common_symbols = factor_values.columns.intersection(forward_returns.columns)
    
    factor_aligned = factor_values.loc[common_dates, common_symbols]
    returns_aligned = forward_returns.loc[common_dates, common_symbols]
    
    # 按日期分组计算
    quantile_returns = pd.DataFrame(index=common_dates, columns=range(1, n_quantiles + 1))
    
    for date in common_dates:
        factor_row = factor_aligned.loc[date]
        return_row = returns_aligned.loc[date]
        
        # 去除缺失值
        valid_mask = ~(factor_row.isna() | return_row.isna())
        if valid_mask.sum() < n_quantiles:
            quantile_returns.loc[date] = np.nan
            continue
        
        factor_valid = factor_row[valid_mask]
        return_valid = return_row[valid_mask]
        
        # 分位数分组
        quantiles = pd.qcut(factor_valid, q=n_quantiles, labels=False, duplicates='drop') + 1
        
        # 计算每组平均收益
        for q in range(1, n_quantiles + 1):
            mask = quantiles == q
            if mask.sum() > 0:
                quantile_returns.loc[date, q] = return_valid[mask].mean()
            else:
                quantile_returns.loc[date, q] = np.nan
    
    return quantile_returns


def factor_summary(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 5
) -> Dict:
    """
    因子统计摘要
    
    Parameters:
    -----------
    factor_values : pd.DataFrame
        因子值
    forward_returns : pd.DataFrame
        未来收益
    n_quantiles : int
        分位数数量
    
    Returns:
    --------
    dict
        因子统计摘要
    """
    # 计算 IC
    ic_series = calculate_ic(factor_values, forward_returns)
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ir = calculate_ir(ic_series)
    ic_positive_ratio = (ic_series > 0).sum() / len(ic_series)
    
    # 计算分位数收益
    quantile_returns = factor_returns_by_quantile(factor_values, forward_returns, n_quantiles)
    quantile_annual_returns = quantile_returns.mean() * 252
    
    # 计算多空收益（最高分位 - 最低分位）
    long_short_return = quantile_annual_returns.iloc[-1] - quantile_annual_returns.iloc[0]
    
    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ir': ir,
        'ic_positive_ratio': ic_positive_ratio,
        'quantile_annual_returns': quantile_annual_returns.to_dict(),
        'long_short_return': long_short_return,
        'n_observations': len(ic_series)
    }


def plot_ic_distribution(ic_series: pd.Series, save_path: Optional[str] = None):
    """
    绘制 IC 分布图
    
    Parameters:
    -----------
    ic_series : pd.Series
        IC 序列
    save_path : str, optional
        保存路径
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # IC 分布直方图
        axes[0].hist(ic_series.dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(ic_series.mean(), color='r', linestyle='--', label=f'Mean: {ic_series.mean():.4f}')
        axes[0].set_xlabel('IC')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('IC Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # IC 时间序列
        axes[1].plot(ic_series.index, ic_series.values, alpha=0.6)
        axes[1].axhline(0, color='r', linestyle='--', alpha=0.5)
        axes[1].axhline(ic_series.mean(), color='g', linestyle='--', 
                       label=f'Mean: {ic_series.mean():.4f}')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('IC')
        axes[1].set_title('IC Time Series')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("matplotlib 未安装，无法绘制图表")


def plot_quantile_returns(quantile_returns: pd.DataFrame, save_path: Optional[str] = None):
    """
    绘制分位数收益图
    
    Parameters:
    -----------
    quantile_returns : pd.DataFrame
        分位数收益数据
    save_path : str, optional
        保存路径
    """
    try:
        import matplotlib.pyplot as plt
        
        # 计算年化收益
        annual_returns = quantile_returns.mean() * 252
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(annual_returns.index, annual_returns.values, alpha=0.7, edgecolor='black')
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Annualized Return')
        ax.set_title('Factor Returns by Quantile')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("matplotlib 未安装，无法绘制图表")

