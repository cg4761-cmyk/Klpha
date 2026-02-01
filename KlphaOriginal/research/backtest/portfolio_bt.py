"""
组合回测引擎
采用 Mark-to-Market (逐日盯市) 算法，适用于美股 Long-Only 策略
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from numba import njit
from pathlib import Path
import yaml
from scipy import stats
from research.backtest.signals import cs_booksize_nb


@njit
def _calculate_mddd(ret: np.ndarray) -> int:
    """
    计算最大回撤持续时间（天数）
    
    Parameters:
    -----------
    ret : np.ndarray
        单日收益率序列
    
    Returns:
    --------
    int
        最大回撤持续时间（天数）
    """
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


def simulate_portfolio(
    alpha: pd.DataFrame,
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    costs: Dict,
    booksize: float = 1.0,
    upper_bound: float = 0.1,
    window_size: int = 1
) -> pd.DataFrame:
    """
    Long-Only 组合回测引擎 (Mark-to-Market with Rolling Average)
    
    采用逐日盯市算法 + 滚动平均平滑信号：
    - T 日收盘产生信号 -> T 日收盘/T+1 开盘买入 -> 享受 T+1 的涨跌幅
    - 每日 PnL = 昨日持仓 * 今日收益率
    - 使用滚动平均平滑信号，降低换手率（每天调整约 1/window_size 的仓位）
    - 完全向量化实现，无 Python 循环
    
    Parameters:
    -----------
    alpha : pd.DataFrame
        仓位信号，行=日期，列=股票代码，值=仓位权重（负值将被设为0）
    prices : pd.DataFrame
        价格数据（用于计算收益），行=日期，列=股票代码
    volume : pd.DataFrame
        成交量数据（用于流动性计算），行=日期，列=股票代码（可选）
    costs : dict
        交易成本配置，包含 'total_cost' 等
    booksize : float
        组合规模（归一化，默认1.0表示100%）
    upper_bound : float
        单只股票最大仓位（按booksize的比例）
    window_size : int
        滚动窗口大小（>1时使用滚动平均平滑信号，降低换手率）
        例如 window_size=5 表示使用过去5天的平均信号，每天约调整 1/5 的仓位
    
    Returns:
    --------
    pd.DataFrame
        回测结果，包含：
        - ret: 每日收益率（费前）
        - ret_net: 每日收益率（费后）
        - tvr: 每日换手率
        - dailypnl: 每日盈亏（绝对金额）
        - dailypnl_net: 每日盈亏（扣除成本后）
        - long_size: 每日多头总仓位
    """
    # ========== 1. 数据对齐 ==========
    common_dates = alpha.index.intersection(prices.index)
    common_symbols = alpha.columns.intersection(prices.columns)
    
    if len(common_dates) == 0 or len(common_symbols) == 0:
        raise ValueError("alpha 和 prices 没有共同的日期或股票代码")
    
    alpha = alpha.loc[common_dates, common_symbols]
    prices = prices.loc[common_dates, common_symbols]
    
    # ========== 2. 计算每日收益率（Mark-to-Market 核心）==========
    # 使用 pct_change() 计算单日收益率，不预测未来
    daily_returns = prices.pct_change().fillna(0.0)
    
    # ========== 3. 输入清理：Long-Only 约束 ==========
    # 填充 NaN 为 0
    alpha_clean = alpha.fillna(0.0)
    
    # 强制做多：将所有负值信号设为 0（在滚动之前处理）
    alpha_clean = alpha_clean.clip(lower=0.0)
    
    # ========== 4. 滚动平均平滑信号（关键：降低换手率）==========
    # 如果 window_size > 1，使用滚动平均来平滑信号
    # 这模拟了"分批建仓"的效果：每天只调整约 1/window_size 的仓位
    # 数学上，换手率会降低到约 1/window_size
    if window_size > 1:
        # 向量化滚动平均（完全无循环）
        # 注意：必须在 DataFrame 上操作，不能是 numpy 数组
        alpha_smoothed = alpha_clean.rolling(window=window_size, min_periods=1).mean()
    else:
        alpha_smoothed = alpha_clean
    
    # ========== 5. 横截面仓位标准化 ==========
    # 关键：对平滑后的信号进行归一化，确保每天总仓位 = booksize
    # 转换为 numpy 数组进行归一化
    alpha_values = alpha_smoothed.values
    
    # 调用 cs_booksize_nb 进行横截面归一化
    actual_positions = cs_booksize_nb(
        alpha_values,
        size=booksize,
        upper_bound=upper_bound,
        lower_bound=0.0
    )
    
    # 转换回 DataFrame 以便后续向量化操作
    pos_df = pd.DataFrame(
        actual_positions,
        index=common_dates,
        columns=common_symbols
    )
    
    # ========== 6. 计算每日盈亏 (Mark-to-Market PnL) ==========
    # 核心逻辑：今天的 PnL = 昨天的持仓 * 今天的收益率
    # 将持仓向后移动一天：T 日的持仓享受 T+1 日的收益
    pos_shifted = pos_df.shift(1).fillna(0.0)  # 第一天为0（建仓前）
    
    # 确保索引对齐（避免索引不匹配问题）
    common_dates_pnl = pos_shifted.index.intersection(daily_returns.index)
    pos_shifted = pos_shifted.loc[common_dates_pnl]
    daily_returns_aligned = daily_returns.loc[common_dates_pnl]
    
    # 向量化计算每日个股盈亏（矩阵乘法）
    pnl_matrix = pos_shifted * daily_returns_aligned
    
    # 汇总得到每日总盈亏（向量化求和）
    daily_pnl = pnl_matrix.sum(axis=1).values
    
    # ========== 7. 计算换手率与交易成本 ==========
    # 换手 = abs(今天目标持仓 - 昨天目标持仓)
    # 使用向量化操作计算仓位变化
    pos_diff = pos_df.diff()
    # 第一天视为从0建仓，所以第一天的换手 = 第一天的持仓
    pos_diff.iloc[0] = pos_df.iloc[0]
    
    # 对齐换手率计算的索引（与 PnL 计算保持一致）
    pos_diff_aligned = pos_diff.loc[common_dates_pnl]
    
    # 每日总换手金额（向量化求和）
    daily_turnover = pos_diff_aligned.abs().sum(axis=1).values
    
    # 计算交易成本（买卖双边）
    cost_rate = costs.get('total_cost', 0.0015)  # 默认万分之15（含滑点）
    daily_cost = daily_turnover * cost_rate
    
    # ========== 8. 计算每日收益率 ==========
    # 费前收益率 = 每日盈亏 / 组合规模
    daily_ret = daily_pnl / booksize
    
    # 费后收益率 = (每日盈亏 - 交易成本) / 组合规模
    daily_ret_net = (daily_pnl - daily_cost) / booksize
    
    # ========== 9. 计算每日多头总仓位（用于验证）==========
    # 对齐仓位计算的索引
    long_size = pos_df.loc[common_dates_pnl].sum(axis=1).values
    
    # ========== 10. 构建结果 DataFrame ==========
    # 所有结果使用相同的索引（对齐后的日期）
    results = pd.DataFrame({
        'ret': daily_ret,              # 每日收益率（费前）
        'ret_net': daily_ret_net,      # 每日收益率（费后）
        'tvr': daily_turnover / booksize,  # 每日换手率
        'dailypnl': daily_pnl,         # 每日盈亏（绝对金额）
        'dailypnl_net': daily_pnl - daily_cost,  # 每日盈亏（扣除成本后）
        'long_size': long_size         # 每日多头总仓位
    }, index=common_dates_pnl)
    
    return results


def create_synthetic_benchmark(prices: pd.DataFrame) -> pd.Series:
    """
    创建合成基准（等权重组合），消除幸存者偏差
    
    基于当前股票池构建等权重指数，确保基准和策略使用完全相同的股票池。
    这样可以公平地比较选股能力，而不会受到股票池与真实指数成分股差异的影响。
    
    Parameters:
    -----------
    prices : pd.DataFrame
        价格数据，行=日期，列=股票代码
    
    Returns:
    --------
    pd.Series
        合成基准的日收益率序列，索引为日期
        计算方式：每日所有股票收益率的平均值（等权重组合）
    """
    # 计算每只股票的单日收益率
    daily_returns = prices.pct_change()
    
    # 计算每日所有股票收益率的平均值（等权重组合）
    # axis=1 表示按行（日期）计算，即每日所有股票的平均收益率
    synthetic_benchmark = daily_returns.mean(axis=1)
    
    # 删除第一行（NaN，因为 pct_change 的第一行是 NaN）
    synthetic_benchmark = synthetic_benchmark.dropna()
    
    synthetic_benchmark.name = 'Synthetic_Benchmark'
    
    return synthetic_benchmark


def calculate_metrics(
    returns: pd.Series, 
    risk_free_rate: float = 0.0,
    benchmark_returns: Optional[pd.Series] = None
) -> Dict:
    """
    计算性能指标（基于每日收益率）
    
    Parameters:
    -----------
    returns : pd.Series
        每日收益率序列（已经是单日收益率）
    risk_free_rate : float
        无风险利率（年化）
    benchmark_returns : pd.Series, optional
        基准收益率序列（用于计算 Alpha、Beta 等 CAPM 指标）
        如果提供，将计算 Alpha、Beta、R-Squared 等指标
    
    Returns:
    --------
    dict
        性能指标字典，包含：
        - annual_return_arithmetic: 算术年化收益率（简单平均）
        - annual_return_geometric: 几何年化收益率（复利年化）
        - annual_volatility: 年化波动率
        - sharpe_ratio: 夏普比率（基于几何年化收益率）
        - cumulative_return: 累计收益率
        - max_drawdown: 最大回撤
        - max_drawdown_duration: 最大回撤持续时间（天数）
        - win_rate: 胜率
        - profit_loss_ratio: 盈亏比
        - total_trades: 总交易天数
        - positive_trades: 盈利天数
        - negative_trades: 亏损天数
        - alpha: Alpha（年化），如果提供了 benchmark_returns
        - beta: Beta，如果提供了 benchmark_returns
        - r_squared: R-Squared（拟合优度），如果提供了 benchmark_returns
    """
    # 算术年化收益（基于252个交易日，简单平均）
    annual_ret_arithmetic = returns.mean() * 252
    
    # 几何年化收益（复利年化，更准确反映实际收益）
    # 使用对数形式避免数值溢出
    n_days = len(returns)
    if n_days > 0:
        # 方法1：对数形式（更稳定）
        log_returns = np.log(1 + returns)
        annual_ret_geometric = np.exp(log_returns.sum() * 252 / n_days) - 1
    else:
        annual_ret_geometric = 0.0
    
    # 年化波动率（基于252个交易日）
    annual_vol = returns.std() * np.sqrt(252)
    
    # 夏普比率（基于几何年化收益率，更准确）
    sharpe = (annual_ret_geometric - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    
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
    
    # ========== CAPM 指标计算（Alpha & Beta）==========
    if benchmark_returns is not None:
        # 1. 基于索引对齐（取交集）
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) == 0:
            # 如果没有共同索引，返回 NaN
            alpha = np.nan
            beta = np.nan
            r_squared = np.nan
        else:
            returns_aligned = returns.loc[common_index]
            benchmark_aligned = benchmark_returns.loc[common_index]
            
            # 2. 扣除无风险利率（将年化无风险利率转换为日度）
            # 公式：daily_rf = (1 + annual_rf)^(1/252) - 1
            daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
            
            # 计算超额收益
            excess_returns = returns_aligned - daily_rf
            excess_benchmark = benchmark_aligned - daily_rf
            
            # 3. 使用线性回归计算 Beta 和 Alpha
            # 模型：excess_returns = alpha + beta * excess_benchmark + epsilon
            # scipy.stats.linregress 返回：slope, intercept, rvalue, pvalue, stderr
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                excess_benchmark.values,
                excess_returns.values
            )
            
            # Beta = 斜率
            beta = slope
            
            # Alpha = 截距（日度），需要年化
            daily_alpha = intercept
            alpha = daily_alpha * 252
            
            # R-Squared = r_value^2
            r_squared = r_value ** 2
    else:
        # 如果未提供基准收益率，返回 NaN
        alpha = np.nan
        beta = np.nan
        r_squared = np.nan
    
    return {
        'annual_return_arithmetic': annual_ret_arithmetic,
        'annual_return_geometric': annual_ret_geometric,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'cumulative_return': cumulative_ret,
        'max_drawdown': mdd,
        'max_drawdown_duration': mddd,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'total_trades': len(returns),
        'positive_trades': (returns > 0).sum(),
        'negative_trades': (returns < 0).sum(),
        'alpha': alpha,
        'beta': beta,
        'r_squared': r_squared
    }


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