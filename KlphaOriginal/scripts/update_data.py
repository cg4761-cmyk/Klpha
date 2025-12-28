"""
下载美股历史数据并保存为专业量化格式
格式：data.pkl，包含字典 {"close": DataFrame, "open": DataFrame, ...}
每个 DataFrame：行=日期，列=股票代码
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
import json
import yaml
from datetime import datetime
from typing import List, Dict, Optional
import time
import pickle
import argparse


class BacktestDataDownloader:
    """回测数据下载器 - 生成专业量化格式"""
    
    def __init__(self, data_dir: Path = None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.raw_dir = self.data_dir / "raw" / "daily"
        self.processed_dir = self.data_dir / "processed"
        self.metadata_dir = self.data_dir / "metadata"
        
        for dir_path in [self.raw_dir, self.processed_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_sp500_symbols(self) -> List[str]:
        """获取 SP500 成分股列表"""
        # 方法1: 尝试从 Wikipedia 获取（带 User-Agent）
        try:
            import urllib.request
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            # 添加 User-Agent 避免 403 错误
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                html = response.read()
            tables = pd.read_html(html)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            # 处理特殊符号（如 BRK.B -> BRK-B）
            symbols = [s.replace('.', '-') for s in symbols]
            print(f"成功获取 SP500 列表: {len(symbols)} 只股票")
            # 保存到缓存
            cache_file = self.metadata_dir / "sp500_symbols.json"
            with open(cache_file, 'w') as f:
                json.dump({"symbols": symbols, "update_date": datetime.now().isoformat()}, f)
            return symbols
        except Exception as e:
            print(f"从 Wikipedia 获取 SP500 列表失败: {e}")
        
        # 方法2: 尝试使用本地缓存的 SP500 列表（如果存在）
        cache_file = self.metadata_dir / "sp500_symbols.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    symbols = cached_data.get('symbols', [])
                    if symbols:
                        print(f"使用缓存的 SP500 列表: {len(symbols)} 只股票")
                        return symbols
            except Exception as e:
                print(f"读取缓存失败: {e}")
        
        # 方法3: 使用扩展的默认列表（包含更多主要股票）
        print("使用扩展的默认股票列表")
        default_symbols = [
            # 科技股
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC',
            # 金融股
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW',
            # 消费股
            'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW',
            # 医疗股
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR',
            # 工业股
            'BA', 'CAT', 'GE', 'HON', 'UPS', 'RTX',
            # 能源股
            'XOM', 'CVX', 'COP', 'SLB',
            # 其他
            'V', 'MA', 'DIS', 'VZ', 'T', 'CMCSA', 'CSCO', 'ORCL', 'IBM', 'ADBE'
        ]
        return default_symbols
    
    def download_symbol(self, symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """下载单个股票的历史数据"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return None
            
            # 重命名列名为小写
            df.columns = [col.lower() for col in df.columns]
            df.index.name = 'date'
            
            return df
        except Exception as e:
            print(f"下载 {symbol} 失败: {e}")
            return None
    
    def download_universe(self, symbols: List[str], start_date: str, end_date: str = None):
        """批量下载股票池数据（保存原始 CSV）"""
        print(f"开始下载 {len(symbols)} 只股票的数据...")
        print(f"日期范围: {start_date} 到 {end_date or '最新'}")
        print()
        
        downloaded = []
        failed = []
        
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] 下载 {symbol}...", end=" ", flush=True)
            
            df = self.download_symbol(symbol, start_date, end_date)
            
            if df is not None and not df.empty:
                # 保存原始数据
                file_path = self.raw_dir / f"{symbol}.csv"
                df.to_csv(file_path)
                downloaded.append(symbol)
                print(f"✓ ({len(df)} 条记录)")
            else:
                failed.append(symbol)
                print("✗")
            
            # 避免请求过快
            time.sleep(0.1)
        
        # 保存元数据
        metadata = {
            "download_date": datetime.now().isoformat(),
            "start_date": start_date,
            "end_date": end_date,
            "total_symbols": len(symbols),
            "downloaded": downloaded,
            "failed": failed
        }
        
        metadata_file = self.metadata_dir / "download_log.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        symbols_file = self.metadata_dir / "symbols.json"
        with open(symbols_file, 'w', encoding='utf-8') as f:
            json.dump({"symbols": downloaded}, f, indent=2)
        
        print(f"\n下载完成!")
        print(f"成功: {len(downloaded)} 只")
        if failed:
            print(f"失败: {len(failed)} 只")
            print(f"失败的股票: {failed}")
        
        return downloaded
    
    def create_quant_format(self, symbols: List[str] = None):
        """
        创建专业量化格式：data.pkl
        格式：{"close": DataFrame, "open": DataFrame, ...}
        每个 DataFrame：行=日期，列=股票代码
        """
        print("\n正在创建量化格式数据...")
        
        # 如果没有提供 symbols，从元数据读取
        if symbols is None:
            symbols_file = self.metadata_dir / "symbols.json"
            if symbols_file.exists():
                with open(symbols_file, 'r') as f:
                    symbols = json.load(f)["symbols"]
            else:
                # 从 raw 目录读取所有 CSV
                symbols = [f.stem for f in self.raw_dir.glob("*.csv")]
        
        if not symbols:
            print("错误: 没有找到股票数据")
            return None
        
        print(f"处理 {len(symbols)} 只股票...")
        
        # 初始化字典，存储每个字段的宽格式 DataFrame
        data_dict = {
            "close": None,
            "open": None,
            "high": None,
            "low": None,
            "volume": None
        }
        
        # 收集所有数据
        all_dataframes = {key: [] for key in data_dict.keys()}
        all_dates = set()
        
        for symbol in symbols:
            file_path = self.raw_dir / f"{symbol}.csv"
            if not file_path.exists():
                print(f"警告: {symbol}.csv 不存在，跳过")
                continue
            
            try:
                df = pd.read_csv(file_path, index_col='date', parse_dates=True)
                
                # 确保列名是小写
                df.columns = [col.lower() for col in df.columns]
                
                # 收集每个字段的数据
                for field in data_dict.keys():
                    if field in df.columns:
                        # 创建单列 DataFrame，列名是股票代码
                        field_df = pd.DataFrame({symbol: df[field]})
                        all_dataframes[field].append(field_df)
                
                all_dates.update(df.index)
                
            except Exception as e:
                print(f"处理 {symbol} 失败: {e}")
                continue
        
        # 合并每个字段的数据
        print("正在合并数据...")
        for field in data_dict.keys():
            if all_dataframes[field]:
                # 按日期索引合并所有股票
                combined = pd.concat(all_dataframes[field], axis=1)
                # 按日期排序
                combined = combined.sort_index()
                # 确保日期索引名称
                combined.index.name = 'date'
                data_dict[field] = combined
                print(f"  {field}: {combined.shape[0]} 行 × {combined.shape[1]} 列")
        
        # 保存为 pickle
        output_file = self.processed_dir / "data.pkl"
        print(f"\n保存数据到: {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 打印数据统计
        if data_dict["close"] is not None:
            close_df = data_dict["close"]
            print(f"\n数据统计:")
            print(f"  日期范围: {close_df.index.min()} 到 {close_df.index.max()}")
            print(f"  交易日数: {len(close_df)}")
            print(f"  股票数量: {len(close_df.columns)}")
            missing_pct = close_df.isna().sum().sum() / close_df.size * 100
            print(f"  缺失值比例: {missing_pct:.2f}%")
            
            # 显示前几行几列
            print(f"\n数据预览 (前5行 × 前5列):")
            print(close_df.iloc[:5, :5])
        
        print(f"\n✓ 数据已保存为专业量化格式: {output_file}")
        print("\n使用方式:")
        print('  import pickle')
        print('  with open("data/processed/data.pkl", "rb") as f:')
        print('      data = pickle.load(f)')
        print('  close_df = data["close"].copy()  # 行=日期，列=股票代码')
        
        return data_dict


def load_config():
    """从配置文件加载参数"""
    config_path = Path(__file__).parent.parent / "configs" / "universe.yaml"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('universe', {})
        except Exception as e:
            print(f"读取配置文件失败: {e}")
            return {}
    return {}


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='下载美股回测数据')
    parser.add_argument('--start-date', type=str, default=None,
                        help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='结束日期 (YYYY-MM-DD)，不指定则下载到最新')
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='自定义股票代码列表，如: --symbols AAPL MSFT GOOGL')
    parser.add_argument('--sp500', action='store_true',
                        help='使用 SP500 成分股')
    
    args = parser.parse_args()
    
    # 加载配置文件
    config = load_config()
    
    downloader = BacktestDataDownloader()
    
    # 获取股票池（优先级：命令行 > 配置文件 > 默认）
    if args.symbols:
        symbols = args.symbols
    elif args.sp500:
        symbols = downloader.get_sp500_symbols()
    elif config.get('use_sp500'):
        symbols = downloader.get_sp500_symbols()
    elif config.get('custom_symbols'):
        symbols = config['custom_symbols']
    else:
        # 默认使用 SP500
        symbols = downloader.get_sp500_symbols()
    
    # 获取日期范围（优先级：命令行 > 配置文件 > 默认）
    start_date = args.start_date or config.get('start_date', '2015-01-01')
    end_date = args.end_date if args.end_date else config.get('end_date')
    
    print("=" * 60)
    print("美股回测数据下载器")
    print("=" * 60)
    print(f"配置:")
    print(f"  股票数量: {len(symbols)}")
    print(f"  开始日期: {start_date}")
    print(f"  结束日期: {end_date or '最新'}")
    print("=" * 60)
    print()
    
    # 下载数据
    downloaded_symbols = downloader.download_universe(symbols, start_date, end_date)
    
    # 创建专业量化格式
    data_dict = downloader.create_quant_format(downloaded_symbols)
    
    print("\n" + "=" * 60)
    print("✓ 数据准备完成！可以开始回测了。")
    print("=" * 60)


if __name__ == "__main__":
    main()

