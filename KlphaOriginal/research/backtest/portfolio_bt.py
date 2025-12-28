"""
组合回测引擎
参考加密货币回测代码，适配美股市场
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from numba import njit
from pathlib import Path
import yaml


@njit
def diff_nb(data: np.ndarray, window: int) -> np.ndarray:
    """计算差分（用于TVR计算）"""
    result = np.zeros_like(data)
    for i in range(window, data.shape[0]):
        result[i] = data[i] - data[i - window]
    return result


def rolling_sum_nb(data: np.ndarray, window: int, minp: int = 1) -> np.ndarray:
    """
    滚动求和（用于计算持仓）
    使用纯 Python 实现，避免 Numba 类型推断问题
    """
    result = np.zeros_like(data)
    n_rows = data.shape[0]
    
    for i in range(n_rows):
        start = max(0, i - window + 1)
        # 使用 numpy 的 nansum，更高效
        result[i] = np.nansum(data[start:i+1], axis=0)
    
    return result


def simulate_portfolio(
    alpha: pd.DataFrame,
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    costs: Dict,
    booksize: float = 1.0,
    upper_bound: float = 0.1,
    forward_window: int = 1,
    hold_period: int = 1
) -> pd.DataFrame:
    """
    模拟组合回测（参考 simulate2，适配美股）
    
    Parameters:
    -----------
    alpha : pd.DataFrame
        仓位信号，行=日期，列=股票代码，值=仓位权重
    prices : pd.DataFrame
        价格数据（用于计算收益），行=日期，列=股票代码
    volume : pd.DataFrame
        成交量数据（用于流动性计算），行=日期，列=股票代码
    costs : dict
        交易成本配置，包含 'total_cost' 等
    booksize : float
        组合规模（归一化）
    upper_bound : float
        单只股票最大仓位
    forward_window : int
        前瞻窗口（避免未来信息泄露，已弃用，实际使用 hold_period）
    hold_period : int
        持仓天数（买入后持有多少天，1表示每天调仓，5表示持仓5天）
        收益计算将使用此参数，确保评估的收益与实际持仓期一致
    
    Returns:
    --------
    pd.DataFrame
        回测结果，包含：
        - ret: 每日收益
        - ret_net: 扣除成本后的收益
        - tvr: 换手率
        - dailypnl: 每日盈亏
        - long_size: 多仓规模
        - short_size: 空仓规模
    """
    # 确保索引对齐
    common_dates = alpha.index.intersection(prices.index).intersection(volume.index)
    alpha = alpha.loc[common_dates]
    prices = prices.loc[common_dates]
    volume = volume.loc[common_dates]
    
    # 计算未来收益（使用 hold_period，因为这是实际持有天数）
    # 避免未来信息泄露：在 T 日只能看到 T 日及之前的数据
    # 用未来 hold_period 天的收益来评估这次持仓
    forward_returns = prices.pct_change(hold_period).shift(-hold_period)
    
    # 对齐日期
    common_dates = alpha.index.intersection(forward_returns.index)
    alpha = alpha.loc[common_dates]
    forward_returns = forward_returns.loc[common_dates]
    volume = volume.loc[common_dates]
    
    # 转换为 numpy 数组
    alpha_values = alpha.values.copy()
    returns = forward_returns.values
    volume_values = volume.values
    
    # 处理缺失值
    alpha_values = np.where(np.isnan(alpha_values), 0, alpha_values)
    returns = np.where(np.isnan(returns), 0, returns)
    
    # 实现固定持仓期逻辑
    # 如果 hold_period > 1，只在特定日期调仓，其他日期保持仓位不变
    n_dates = len(common_dates)
    actual_positions = np.zeros_like(alpha_values)
    
    if hold_period > 1:
        # 固定持仓期模式：每 hold_period 天调仓一次
        for i in range(n_dates):
            # 判断是否是调仓日（每 hold_period 天调一次）
            if i % hold_period == 0:
                # 调仓日：使用新的仓位信号
                actual_positions[i] = alpha_values[i]
            else:
                # 非调仓日：保持上一次的仓位
                if i > 0:
                    actual_positions[i] = actual_positions[i-1]
                else:
                    actual_positions[i] = alpha_values[i]
    else:
        # hold_period = 1：每天调仓（原有逻辑）
        actual_positions = alpha_values
    
    # 分离多空仓位（使用实际持仓）
    long_alpha = actual_positions.copy()
    short_alpha = actual_positions.copy()
    long_alpha[long_alpha <= 0] = 0
    short_alpha[short_alpha >= 0] = 0
    
    # 计算多空规模
    long_size = np.nansum(long_alpha, axis=1)
    short_size = np.nansum(np.abs(short_alpha), axis=1)
    
    # 计算 TVR（换手率）- 基于实际仓位变化
    # 使用 hold_period 作为窗口，因为这是实际持仓期
    window = hold_period
    if hold_period > 1:
        # 计算仓位变化（只在调仓日有变化）
        position_diff = np.zeros_like(actual_positions)
        for i in range(1, n_dates):
            if i % hold_period == 0:
                # 调仓日：计算仓位变化
                position_diff[i] = actual_positions[i] - actual_positions[i-1]
        alpha_diff = position_diff
    else:
        # 每天调仓：使用原有逻辑
        alpha_diff = diff_nb(alpha_values, window)
    
    tvr = np.nansum(np.abs(alpha_diff), axis=1) / (booksize * window)
    
    # 计算流动性指标（基于成交量）
    dollar_volume = prices.loc[common_dates].values * volume_values
    liq = np.abs(alpha_diff) / (dollar_volume + 1e-10)  # 避免除零
    liq = np.nan_to_num(liq, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
    # 流动性指标（90分位数）
    liq_1 = np.nanpercentile(liq, 90, axis=1) * 10
    
    # 计算持仓（用于计算收益）- 使用实际持仓
    position = rolling_sum_nb(actual_positions, window, minp=1)
    position_long = rolling_sum_nb(long_alpha, window, minp=1)
    
    # 计算每日盈亏（使用实际持仓）
    pnl_long = returns * long_alpha
    pnl_short = returns * short_alpha
    
    # 计算每日总盈亏
    daily_pnl = np.nansum(pnl_long + pnl_short, axis=1)
    
    # 计算每日收益
    # window = hold_period，确保收益计算与实际持仓期一致
    daily_ret = daily_pnl / (booksize * window)
    
    # 计算交易成本
    total_cost = costs.get('total_cost', 0.0015)
    cost_pnl = -tvr * booksize * window * total_cost
    
    # 扣除成本后的收益
    daily_ret_net = daily_ret + (cost_pnl / (booksize * window))
    
    # 构建结果 DataFrame
    results = pd.DataFrame({
        'ret': daily_ret,
        'ret_net': daily_ret_net,
        'tvr': tvr,
        'dailypnl': daily_pnl,
        'dailypnl_net': daily_pnl + cost_pnl,
        'long_size': long_size,
        'short_size': short_size,
        'liq': liq_1
    }, index=common_dates)
    
    return results


def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> Dict:
    """
    计算性能指标
    
    Parameters:
    -----------
    returns : pd.Series
        每日收益序列
    risk_free_rate : float
        无风险利率（年化）
    
    Returns:
    --------
    dict
        性能指标字典
    """
    # 年化收益
    annual_ret = returns.mean() * 252
    
    # 年化波动率
    annual_vol = returns.std() * np.sqrt(252)
    
    # 夏普比率
    sharpe = (annual_ret - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    
    # 累计收益
    cumulative_ret = (1 + returns).prod() - 1
    
    # 最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    mdd = drawdown.min()
    
    # 最大回撤持续时间
    mddd = _calculate_mddd(returns.values)
    
    # 胜率
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    # 盈亏比
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    profit_loss_ratio = (
        positive_returns.mean() / abs(negative_returns.mean())
        if len(negative_returns) > 0 and negative_returns.mean() != 0
        else 0
    )
    
    return {
        'annual_return': annual_ret,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'cumulative_return': cumulative_ret,
        'max_drawdown': mdd,
        'max_drawdown_duration': mddd,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'total_trades': len(returns),
        'positive_trades': (returns > 0).sum(),
        'negative_trades': (returns < 0).sum()
    }


@njit
def _calculate_mddd(ret: np.ndarray) -> int:
    """计算最大回撤持续时间"""
    returns = np.cumsum(ret) + 1
    mddd = 0
    max_value = 1.0
    max_idx = 0
    
    for i in range(returns.shape[0]):
        value = returns[i]
        if value > max_value:
            max_value = value
            max_idx = i
        else:
            tmp_ddd = i - max_idx
            if tmp_ddd > mddd:
                mddd = tmp_ddd
    
    return mddd


def load_backtest_config(config_path: Optional[Path] = None) -> Dict:
    """加载回测配置"""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "backtest.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config.get('backtest', {})


def load_costs_config(config_path: Optional[Path] = None) -> Dict:
    """加载交易成本配置"""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "costs.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config.get('costs', {})

