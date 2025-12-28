"""
回测模块
"""
from .portfolio_bt import (
    simulate_portfolio,
    calculate_metrics,
    load_backtest_config,
    load_costs_config
)
from .factor_test import (
    calculate_ic,
    calculate_ir,
    factor_summary,
    factor_returns_by_quantile,
    plot_ic_distribution,
    plot_quantile_returns
)
from .signals import (
    normalize_positions,
    cross_section_zscore,
    cross_section_rank,
    cross_section_linear,
    cross_section_log,
    cross_section_sin,
    cross_section_tanh,
    cross_section_sigmoid,
    cross_section_robust,
    normalize_factor,
    winsorize,
    fill_na_with_cross_section
)

__all__ = [
    'simulate_portfolio',
    'calculate_metrics',
    'load_backtest_config',
    'load_costs_config',
    'calculate_ic',
    'calculate_ir',
    'factor_summary',
    'factor_returns_by_quantile',
    'plot_ic_distribution',
    'plot_quantile_returns',
    'normalize_positions',
    'cross_section_zscore',
    'cross_section_rank',
    'cross_section_linear',
    'cross_section_log',
    'cross_section_sin',
    'cross_section_tanh',
    'cross_section_sigmoid',
    'cross_section_robust',
    'normalize_factor',
    'winsorize',
    'fill_na_with_cross_section'
]

