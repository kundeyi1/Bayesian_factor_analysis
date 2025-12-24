import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# --- 页面配置 ---
st.set_page_config(page_title="贝叶斯胜率测算工具", layout="wide")

# 设置中文字体 (尝试适配不同系统)
import platform
system_name = platform.system()
if system_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif system_name == "Darwin": # Mac
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else: # Linux/Cloud
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 备选，可能不支持中文
plt.rcParams['axes.unicode_minus'] = False

# --- 核心逻辑类 (经过改造以适配 Streamlit) ---
class BayesianWinRateModel:
    def __init__(self, price_df, factor_df, benchmark_df, config):
        self.df_price = price_df.copy()
        self.df_factors_raw = factor_df.copy()
        self.df_bench = benchmark_df.copy() if benchmark_df is not None else None
        self.config = config
        
        # 解包配置
        self.stock_code = config['STOCK_CODE']
        self.start_date = pd.to_datetime(config['START_DATE'])
        self.end_date = pd.to_datetime(config['END_DATE'])
        self.data_freq = config['DATA_FREQ']
        self.holding_period = config['HOLDING_PERIOD']
        self.win_threshold = config['WIN_THRESHOLD']
        self.stats_period = config['STATS_PERIOD']
        self.feature1_threshold = config['FEATURE1_THRESHOLD']
        self.feature1_mode = config['FEATURE1_MODE']
        self.feature2_threshold = config['FEATURE2_THRESHOLD']
        self.feature2_mode = config['FEATURE2_MODE']
        self.feature1_name = config['FEATURE1_NAME']
        self.feature2_name = config['FEATURE2_NAME']
        
        # 变换参数
        self.f1_trans = config.get('F1_TRANS', '原始值')
        self.f1_lag = config.get('F1_LAG', 1)
        self.f2_trans = config.get('F2_TRANS', '原始值')
        self.f2_lag = config.get('F2_LAG', 1)

    def _apply_transform(self, series, trans_type, lag_num):
        """辅助函数：应用变换"""
        if trans_type == '原始值':
            return series, ""
        
        suffix = ""
        result = series.copy()
        
        # 确定周期
        periods_yoy = 12
        if self.data_freq == 'Q': periods_yoy = 4
        elif self.data_freq == 'M': periods_yoy = 12
        elif self.data_freq == 'W': periods_yoy = 52
        elif self.data_freq == 'D': periods_yoy = 252
        
        if trans_type == '同比':
            result = series.pct_change(periods=periods_yoy)
            suffix = "_YoY"
        elif trans_type == '环比':
            result = series.pct_change(periods=1)
            suffix = "_MoM"
        elif trans_type == '滞后':
            result = series.shift(lag_num)
            suffix = f"_Lag{lag_num}"
            
        return result, suffix

    def process_data(self):
        """处理数据：清洗、合并、重采样"""
        with st.spinner('正在处理数据...'):
            # 1. 处理股价数据
            # 识别日期列
            date_col = next((c for c in self.df_price.columns if 'date' in c.lower() or '日期' in c or 'time' in c.lower()), self.df_price.columns[0])
            self.df_price.rename(columns={date_col: '日期'}, inplace=True)
            self.df_price['日期'] = pd.to_datetime(self.df_price['日期'])

            # 筛选股票代码
            code_col = next((c for c in self.df_price.columns if 'code' in c.lower() or 'symbol' in c.lower() or '代码' in c), None)
            if code_col:
                self.df_price[code_col] = self.df_price[code_col].astype(str)
                # 尝试多种匹配方式
                filtered_df = self.df_price[self.df_price[code_col] == str(self.stock_code)].copy()
                if len(filtered_df) == 0 and '.' in self.stock_code:
                    short_code = self.stock_code.split('.')[0]
                    filtered_df = self.df_price[self.df_price[code_col] == short_code].copy()
                if len(filtered_df) == 0:
                     # 尝试去前导零
                    short_code = self.stock_code.split('.')[0] if '.' in self.stock_code else self.stock_code
                    no_zero_code = str(int(short_code)) if short_code.isdigit() else short_code
                    filtered_df = self.df_price[self.df_price[code_col] == no_zero_code].copy()
                
                if len(filtered_df) > 0:
                    self.df_price = filtered_df
                else:
                    st.warning(f"未在数据中找到代码 {self.stock_code}，将使用全部数据。")

            # 识别收盘价
            close_col = next((c for c in self.df_price.columns if 'close' in c.lower() or '收盘' in c or 'price' in c.lower()), None)
            if not close_col:
                st.error("未找到收盘价列！")
                return False
            
            # 计算收益率
            if 'pct_chg' in self.df_price.columns:
                self.df_price['return'] = self.df_price['pct_chg'] / 100.0
            else:
                self.df_price['return'] = self.df_price[close_col].pct_change()

            # 2. 处理基准数据
            if self.df_bench is not None:
                bench_date_col = next((c for c in self.df_bench.columns if 'date' in c.lower() or '日期' in c), self.df_bench.columns[0])
                self.df_bench.rename(columns={bench_date_col: '日期'}, inplace=True)
                self.df_bench['日期'] = pd.to_datetime(self.df_bench['日期'])
                
                bench_close_col = next((c for c in self.df_bench.columns if 'close' in c.lower() or '收盘' in c or 'price' in c.lower()), None)
                if bench_close_col:
                    if 'pct_chg' in self.df_bench.columns:
                        self.df_bench['bench_return'] = self.df_bench['pct_chg'] / 100.0
                    else:
                        self.df_bench['bench_return'] = self.df_bench[bench_close_col].pct_change()
                    
                    self.df_price = pd.merge(self.df_price, self.df_bench[['日期', 'bench_return']], on='日期', how='left')
                    self.df_price['bench_return'] = self.df_price['bench_return'].fillna(0)
                    
                    # 计算超额收益率 (几何超额: (1+Rs)/(1+Rb) - 1)
                    self.df_price['超额收益率'] = (1 + self.df_price['return']) / (1 + self.df_price['bench_return']) - 1
                else:
                    self.df_price['超额收益率'] = self.df_price['return']
            else:
                self.df_price['超额收益率'] = self.df_price['return']

            # 3. 处理因子数据
            # 清理列名
            self.df_factors_raw.columns = [str(c).strip() for c in self.df_factors_raw.columns]
            factor_date_col = self.df_factors_raw.columns[0]
            self.df_factors_raw.rename(columns={factor_date_col: '日期'}, inplace=True)
            self.df_factors_raw['日期'] = pd.to_datetime(self.df_factors_raw['日期'], errors='coerce')
            self.df_factors_raw = self.df_factors_raw.dropna(subset=['日期'])
            
            if self.feature1_name not in self.df_factors_raw.columns or self.feature2_name not in self.df_factors_raw.columns:
                st.error(f"因子文件中未找到指定的列名: {self.feature1_name} 或 {self.feature2_name}")
                st.write("文件中的列名:", self.df_factors_raw.columns.tolist())
                return False

            # 使用显式赋值构建 df_features，避免 rename 在列名相同时的冲突
            self.df_features = pd.DataFrame()
            self.df_features['日期'] = self.df_factors_raw['日期']
            
            # 应用变换
            f1_series, f1_suffix = self._apply_transform(self.df_factors_raw[self.feature1_name], self.f1_trans, self.f1_lag)
            self.df_features['feature1'] = f1_series
            self.feature1_name += f1_suffix # 更新名称用于显示
            
            f2_series, f2_suffix = self._apply_transform(self.df_factors_raw[self.feature2_name], self.f2_trans, self.f2_lag)
            self.df_features['feature2'] = f2_series
            self.feature2_name += f2_suffix # 更新名称用于显示

            # 4. 重采样 (如果需要)
            resample_rule = None
            if self.data_freq == 'W':
                resample_rule = 'W-FRI'
            elif self.data_freq == 'M':
                resample_rule = 'ME'
            elif self.data_freq == 'Q':
                resample_rule = 'QE'
            
            if resample_rule:
                self.df_price.set_index('日期', inplace=True)
                
                # 构建基准净值序列 (用于重采样)
                if 'bench_return' in self.df_price.columns:
                    self.df_price['bench_index'] = (1 + self.df_price['bench_return'].fillna(0)).cumprod()
                
                resample_dict = {
                    close_col: 'last'
                }
                if 'bench_index' in self.df_price.columns:
                    resample_dict['bench_index'] = 'last'
                
                df_resampled = self.df_price.resample(resample_rule).agg(resample_dict)
                
                # 重新计算收益率
                df_resampled['return'] = df_resampled[close_col].pct_change()
                
                if 'bench_index' in df_resampled.columns:
                    df_resampled['bench_return'] = df_resampled['bench_index'].pct_change()
                    df_resampled['超额收益率'] = (1 + df_resampled['return'].fillna(0)) / (1 + df_resampled['bench_return'].fillna(0)) - 1
                else:
                    df_resampled['超额收益率'] = df_resampled['return']
                
                self.df_price = df_resampled.reset_index()

            # 计算净值
            self.df_price['超额净值'] = (1 + self.df_price['超额收益率']).cumprod()
            self.df_price['绝对净值'] = (1 + self.df_price['return'].fillna(0)).cumprod()
            
            # 计算持有期收益
            self.df_price[f'持有{self.holding_period}期相对收益'] = self.df_price['超额净值'].shift(-self.holding_period)/self.df_price['超额净值'] - 1

            # 5. 合并
            # 使用 merge_asof 进行模糊匹配，确保即使日期不完全对齐也能匹配到最近的因子值
            # 必须先排序
            self.df_price = self.df_price.sort_values('日期')
            self.df_features = self.df_features.sort_values('日期')
            
            # merge_asof 要求右侧表（因子表）的日期必须小于等于左侧表（价格表）的日期
            # direction='backward' 表示寻找最近的一个过去日期
            self.df = pd.merge_asof(self.df_price, self.df_features, on='日期', direction='backward')
            
            # 移除没有匹配到因子的行 (即价格日期早于最早的因子日期)
            self.df = self.df.dropna(subset=['feature1', 'feature2'])
            
            self.df = self.df.sort_values('日期').reset_index(drop=True)
            
            # 时间筛选
            self.df = self.df[(self.df['日期'] >= self.start_date) & (self.df['日期'] <= self.end_date)]
            
            if len(self.df) == 0:
                st.error("合并后数据为空！请检查：")
                st.write("1. 股价数据日期范围:", self.df_price['日期'].min(), "至", self.df_price['日期'].max())
                st.write("2. 因子数据日期范围:", self.df_features['日期'].min(), "至", self.df_features['日期'].max())
                st.write("3. 是否有重叠的时间段？")
                return False
            
            return True

    def calculate_labels(self):
        self.df['label_return'] = self.df[f'持有{self.holding_period}期相对收益']
        self.df['is_win'] = (self.df['label_return'] > self.win_threshold).astype(int)
        
        # 信号逻辑
        if self.feature1_mode == 'gt':
            cond1 = self.df['feature1'] > self.feature1_threshold
        else:
            cond1 = self.df['feature1'] < self.feature1_threshold
            
        if self.feature2_mode == 'gt':
            cond2 = self.df['feature2'] > self.feature2_threshold
        else:
            cond2 = self.df['feature2'] < self.feature2_threshold
            
        condition = cond1 & cond2
        self.df['is_signal'] = condition.astype(int)
        self.df.loc[self.df['feature1'].isna() | self.df['feature2'].isna(), 'is_signal'] = 0
        
        # 统计
        win_count = self.df['is_win'].sum()
        total_count = len(self.df)
        return win_count, total_count

    def calculate_ic(self):
        """计算因子IC (Information Coefficient)"""
        # IC = Corr(Factor_t, Return_t+1)
        # 在这里，label_return 已经是 t+holding 的收益，feature 是 t 时刻的因子值
        # 所以直接计算相关性即可
        
        ic_data = self.df.dropna(subset=['feature1', 'feature2', 'label_return'])
        
        if len(ic_data) < 2:
            return {}
            
        res = {}
        # Feature 1
        res['f1_pearson'] = ic_data['feature1'].corr(ic_data['label_return'], method='pearson')
        res['f1_spearman'] = ic_data['feature1'].corr(ic_data['label_return'], method='spearman')
        
        # Feature 2
        res['f2_pearson'] = ic_data['feature2'].corr(ic_data['label_return'], method='pearson')
        res['f2_spearman'] = ic_data['feature2'].corr(ic_data['label_return'], method='spearman')
        
        return res

    def run_bayesian_analysis(self):
        valid_mask = self.df['label_return'].notna()
        win_series = self.df['is_win'].where(valid_mask)
        signal_series = self.df['is_signal']
        
        win_and_signal = ((win_series == 1) & (signal_series == 1)).astype(float)
        win_and_signal[~valid_mask] = np.nan
        
        lose_and_signal = ((win_series == 0) & (signal_series == 1)).astype(float)
        lose_and_signal[~valid_mask] = np.nan
        
        roller = win_series.rolling(window=self.stats_period, min_periods=1)
        
        win_count = roller.sum().shift(self.holding_period)
        total_count = roller.count().shift(self.holding_period)
        prior_prob = win_count / total_count
        
        win_signal_count = win_and_signal.rolling(window=self.stats_period, min_periods=1).sum().shift(self.holding_period)
        likelihood_win = win_signal_count / win_count
        
        lose_count = total_count - win_count
        lose_signal_count = lose_and_signal.rolling(window=self.stats_period, min_periods=1).sum().shift(self.holding_period)
        likelihood_lose = lose_signal_count / lose_count
        
        numerator = likelihood_win * prior_prob
        denominator = numerator + likelihood_lose * (1 - prior_prob)
        posterior_prob = numerator / denominator
        posterior_prob = posterior_prob.fillna(0)
        
        # --- 新增：信号策略净值计算 ---
        # 将概率合并回 self.df 以便计算
        self.df['prior_prob'] = prior_prob
        self.df['posterior_prob'] = posterior_prob
        
        # 1. 计算买入信号
        cond_improve = self.df['posterior_prob'] > self.df['prior_prob']
        cond_trigger = self.df['is_signal'] == 1
        cond_robust = (self.df['posterior_prob'] > self.df['posterior_prob'].shift(1) * 0.9) | (self.df['posterior_prob'] > 0.5)

        self.df['buy_signal'] = (cond_improve & cond_trigger & cond_robust).astype(int)

        # 2. 计算仓位
        rolling_density = self.df['is_signal'].rolling(window=self.holding_period).sum() / self.holding_period
        self.df['position'] = np.where(self.df['buy_signal'] == 1, rolling_density, 0)

        # 3. 计算策略净值
        # 使用 'return' 列 (绝对收益)
        strategy_ret = self.df['position'].shift(1) * self.df['return']
        self.df['strategy_net_value'] = (1 + strategy_ret.fillna(0)).cumprod()

        # --- 计算统计指标 ---
        # 确定年化系数
        if self.data_freq == 'D':
            ann_factor = 252
        elif self.data_freq == 'W':
            ann_factor = 52
        elif self.data_freq == 'M':
            ann_factor = 12
        elif self.data_freq == 'Q':
            ann_factor = 4
        else:
            ann_factor = 252 # 默认

        # 1. 先验年化 (Buy & Hold 绝对收益)
        # 使用 'return' 列
        total_ret_prior = (1 + self.df['return'].fillna(0)).prod()
        days = len(self.df)
        if days > 0:
            self.ann_ret_prior = total_ret_prior ** (ann_factor / days) - 1
        else:
            self.ann_ret_prior = 0
            
        # 2. 后验年化 (策略收益)
        total_ret_posterior = self.df['strategy_net_value'].iloc[-1] if len(self.df) > 0 else 1
        if days > 0:
            self.ann_ret_posterior = total_ret_posterior ** (ann_factor / days) - 1
        else:
            self.ann_ret_posterior = 0
            
        # 3. 先验高于后验的比率 (Prior > Posterior)
        # 计算有多少比例的时间，先验概率 > 后验概率
        self.prob_diff_ratio = (self.df['prior_prob'] > self.df['posterior_prob']).mean()
        
        # 4. 先验夏普 (Buy & Hold)
        # 假设无风险利率为 0
        std_prior = self.df['return'].std()
        if std_prior != 0:
            self.sharpe_prior = (self.df['return'].mean() / std_prior) * np.sqrt(ann_factor)
        else:
            self.sharpe_prior = 0
            
        # 5. 后验夏普 (策略)
        std_posterior = strategy_ret.std()
        if std_posterior != 0:
            self.sharpe_posterior = (strategy_ret.mean() / std_posterior) * np.sqrt(ann_factor)
        else:
            self.sharpe_posterior = 0

        self.results_df = pd.DataFrame({
            '日期': self.df['日期'],
            'prior_prob': prior_prob,
            'likelihood_win': likelihood_win,
            'likelihood_lose': likelihood_lose,
            'posterior_prob': posterior_prob,
            'strategy_net_value': self.df['strategy_net_value']
        })

    def plot_factor_price(self):
        """绘制因子与价格走势图"""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 左轴: 价格 (绝对净值)
        ax1.set_xlabel('日期')
        ax1.set_ylabel('股价净值', color='black')
        ax1.plot(self.df['日期'], self.df['绝对净值'], color='black', label='股价净值', linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor='black')

        # 右轴: 因子值
        ax2 = ax1.twinx()
        ax2.set_ylabel('因子值', color='tab:blue')
        
        # 绘制因子1
        ax2.plot(self.df['日期'], self.df['feature1'], color='tab:blue', label=f'因子1: {self.feature1_name}', alpha=0.6, linewidth=1)
        
        # 如果有因子2，也绘制
        if self.feature2_name and 'feature2' in self.df.columns:
            ax2.plot(self.df['日期'], self.df['feature2'], color='tab:green', label=f'因子2: {self.feature2_name}', alpha=0.6, linewidth=1, linestyle='--')
            
        # 限制纵轴范围以排除极值
        all_factors = self.df['feature1'].dropna()
        if self.feature2_name and 'feature2' in self.df.columns:
            all_factors = pd.concat([all_factors, self.df['feature2'].dropna()])
        
        if not all_factors.empty:
            # 排除上下1%的极值来设定坐标轴范围
            lower = all_factors.quantile(0.01)
            upper = all_factors.quantile(0.99)
            # 稍微放宽一点
            margin = (upper - lower) * 0.1
            if upper > lower:
                ax2.set_ylim(lower - margin, upper + margin)

        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title(f'因子与价格走势 - {self.stock_code}')
        return fig

    def plot_results(self):
        # --- 图1: 贝叶斯概率 vs 累计超额 ---
        plot_df = pd.merge(self.results_df, self.df[['日期', '超额净值']], on='日期', how='inner')
        
        # 归一化
        if not plot_df.empty:
            plot_df['plot_value'] = plot_df['超额净值'] / plot_df['超额净值'].iloc[0] - 1
        else:
            plot_df['plot_value'] = 0

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.set_xlabel('日期')
        ax1.set_ylabel('累计超额收益', color='black')
        ax1.plot(plot_df['日期'], plot_df['plot_value'], color='darkblue', label='累计超额收益')
        ax1.tick_params(axis='y', labelcolor='black')
        
        import matplotlib.ticker as mtick
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        ax2 = ax1.twinx()
        color_prior = 'orange'
        color_posterior = 'gray'
        
        ax2.set_ylabel('胜率', color='black')
        ax2.plot(plot_df['日期'], plot_df['prior_prob'], color=color_prior, linestyle='-', label='先验胜率 P(W)')
        ax2.plot(plot_df['日期'], plot_df['posterior_prob'], color=color_posterior, linestyle='-', label='后验胜率 P(W|C)')
        
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(0, 1.1)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title(f'贝叶斯胜率测算结果 - {self.stock_code}')
        
        # --- 图2: 信号策略净值 ---
        plot_df2 = self.df.sort_values('日期').copy()
        
        # 归一化
        if not plot_df2.empty:
            base_excess = plot_df2['超额净值'].iloc[0]
            plot_df2['excess_nav_norm'] = plot_df2['超额净值'] / base_excess
            
            base_strategy = plot_df2['strategy_net_value'].iloc[0]
            plot_df2['strategy_nav_norm'] = plot_df2['strategy_net_value'] / base_strategy
        
        fig2, ax3 = plt.subplots(figsize=(12, 6))
        
        # 左轴: 净值
        ax3.set_xlabel('日期')
        ax3.set_ylabel('净值', color='black')
        ax3.plot(plot_df2['日期'], plot_df2['excess_nav_norm'], color='darkblue', label='累计超额净值', linewidth=2)
        ax3.plot(plot_df2['日期'], plot_df2['strategy_nav_norm'], color='grey', label='信号策略净值', linewidth=2)
        ax3.tick_params(axis='y', labelcolor='black')
        
        # 右轴: 仓位
        ax4 = ax3.twinx()
        ax4.set_ylabel('仓位', color='black')
        ax4.plot(plot_df2['日期'], plot_df2['position'], color='tab:orange', label='仓位', linewidth=1.5)
        ax4.set_ylim(0, 1.1)
        ax4.tick_params(axis='y', labelcolor='black')
        
        # 图例
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left')
        
        plt.title(f'信号策略净值 - {self.stock_code}')
        
        return fig1, fig2

# --- 路径配置 ---
# 定义默认的本地绝对路径
DEFAULT_PRICE_PATH = r"D:\Quant\data\all_stock_data_ts_20140102_20251231.csv"
DEFAULT_BENCHMARK_PATHS = {
    '000300.SH': r"D:\Quant\data\csi300_index_20140102_20251231.csv",
    '000905.SH': r"D:\Quant\data\csi500_index_20140102_20251231.csv",
    '000001.SH': r"D:\Quant\data\sse_composite_index_20140102_20251231.csv"
}

def get_data_path(default_path):
    """
    智能查找数据路径：
    1. 优先查找硬编码的绝对路径 (本机开发环境)
    2. 其次查找当前目录下的 data 文件夹 (便于打包/部署)
    3. 最后查找当前目录
    """
    # 1. 检查绝对路径
    if os.path.exists(default_path):
        return default_path
    
    filename = os.path.basename(default_path)
    
    # 2. 检查 ./data/filename
    data_subpath = os.path.join("data", filename)
    if os.path.exists(data_subpath):
        return data_subpath
        
    # 3. 检查 ./filename
    if os.path.exists(filename):
        return filename
        
    return None

# --- Streamlit UI ---

st.title("📊 贝叶斯胜率测算工具")
st.markdown("上传因子数据，动态调整参数进行回测分析。")

# 侧边栏：参数配置
with st.sidebar:
    st.header("1. 数据配置")
    
    # 自动检测股价文件
    real_price_path = get_data_path(DEFAULT_PRICE_PATH)
    if real_price_path:
        st.success(f"✅ 已加载股价数据")
    else:
        st.error(f"❌ 未找到股价数据")
        st.caption(f"请确保文件存在于以下位置之一:\n1. {DEFAULT_PRICE_PATH}\n2. ./data/{os.path.basename(DEFAULT_PRICE_PATH)}")
    
    factor_file = st.file_uploader("上传因子数据 (Excel)", type=['xlsx', 'xls'])
    
    st.header("2. 基础参数")
    stock_code = st.text_input("股票代码", value="601919.SH")
    benchmark_code = st.selectbox("基准指数", options=list(DEFAULT_BENCHMARK_PATHS.keys()), index=0)
    start_date = st.date_input("开始日期", value=pd.to_datetime("2018-01-01"))
    end_date = st.date_input("结束日期", value=pd.to_datetime("2025-11-30"))
    data_freq = st.selectbox("数据频率", options=['W', 'M', 'Q'], index=0, help="Q=季, M=月, W=周")
    
    st.header("3. 贝叶斯参数")
    holding_period = st.number_input("持有期 (期)", value=12, min_value=1)
    win_threshold = st.number_input("胜率阈值 (超额收益 > X)", value=0.00, step=0.01, format="%.2f")
    stats_period = st.number_input("统计窗口 (期)", value=100, min_value=5)
    
    st.header("4. 信号阈值与处理")
    
    # 动态获取列名
    feature_columns = []
    if factor_file:
        try:
            # 预读取 Excel 获取列名
            excel_file = pd.ExcelFile(factor_file)
            sheet_name = excel_file.sheet_names[0] # 默认读取第一个 sheet
            df_preview = pd.read_excel(factor_file, sheet_name=sheet_name, nrows=0)
            feature_columns = df_preview.columns.tolist()
            # 移除可能的日期列
            feature_columns = [c for c in feature_columns if 'date' not in str(c).lower() and '日期' not in str(c)]
        except Exception as e:
            st.error(f"读取Excel列名失败: {e}")

    col1, col2 = st.columns(2)
    with col1:
        if feature_columns:
            feature1_name = st.selectbox("特征1列名", options=feature_columns, index=0 if len(feature_columns)>0 else 0)
        else:
            feature1_name = st.text_input("特征1列名", value="")
            
        # 特征1变换
        f1_trans = st.selectbox("特征1处理", ['原始值', '同比', '环比', '滞后'], key='f1_trans')
        f1_lag = 1
        if f1_trans == '滞后':
            f1_lag = st.number_input("特征1滞后期数", value=1, min_value=1, key='f1_lag')
            
        feature1_mode = st.selectbox("特征1模式", options=['gt', 'lt'], index=0, format_func=lambda x: "大于" if x=='gt' else "小于")
        feature1_threshold = st.number_input("特征1阈值", value=0.00, step=0.01)
    with col2:
        if feature_columns:
            # 尝试默认选中第二列
            default_idx = 1 if len(feature_columns) > 1 else 0
            feature2_name = st.selectbox("特征2列名", options=feature_columns, index=default_idx)
        else:
            feature2_name = st.text_input("特征2列名", value="")
            
        # 特征2变换
        f2_trans = st.selectbox("特征2处理", ['原始值', '同比', '环比', '滞后'], key='f2_trans')
        f2_lag = 1
        if f2_trans == '滞后':
            f2_lag = st.number_input("特征2滞后期数", value=1, min_value=1, key='f2_lag')
            
        feature2_mode = st.selectbox("特征2模式", options=['gt', 'lt'], index=0, format_func=lambda x: "大于" if x=='gt' else "小于")
        feature2_threshold = st.number_input("特征2阈值", value=0.00, step=0.01)

# 主界面逻辑
if st.button("开始测算", type="primary"):
    if not factor_file:
        st.error("请先上传因子数据！")
    elif not real_price_path:
        st.error("未找到股价数据文件，无法进行测算。")
    else:
        # 读取数据
        try:
            # 读取股价数据
            try:
                df_price = pd.read_csv(real_price_path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df_price = pd.read_csv(real_price_path, encoding='gbk')
        except Exception as e:
            st.error(f"读取股价数据失败: {e}")
            st.stop()
            
        df_factors = pd.read_excel(factor_file)
        
        # 读取基准数据
        default_bench_path = DEFAULT_BENCHMARK_PATHS.get(benchmark_code)
        real_bench_path = get_data_path(default_bench_path) if default_bench_path else None
        
        df_bench = None
        if real_bench_path:
            try:
                try:
                    df_bench = pd.read_csv(real_bench_path, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    df_bench = pd.read_csv(real_bench_path, encoding='gbk')
            except Exception as e:
                st.warning(f"读取基准数据失败: {e}")
        else:
            if default_bench_path:
                st.warning(f"未找到基准文件: {os.path.basename(default_bench_path)}，将使用绝对收益。")


        # 配置字典
        config = {
            'STOCK_CODE': stock_code,
            'START_DATE': start_date,
            'END_DATE': end_date,
            'DATA_FREQ': data_freq,
            'HOLDING_PERIOD': holding_period,
            'WIN_THRESHOLD': win_threshold,
            'STATS_PERIOD': stats_period,
            'FEATURE1_THRESHOLD': feature1_threshold,
            'FEATURE1_MODE': feature1_mode,
            'FEATURE2_THRESHOLD': feature2_threshold,
            'FEATURE2_MODE': feature2_mode,
            'FEATURE1_NAME': feature1_name,
            'FEATURE2_NAME': feature2_name,
            'F1_TRANS': f1_trans,
            'F1_LAG': f1_lag,
            'F2_TRANS': f2_trans,
            'F2_LAG': f2_lag
        }

        # 初始化模型
        model = BayesianWinRateModel(df_price, df_factors, df_bench, config)
        
        # 运行
        if model.process_data():
            # --- 新增：优先展示因子与股价走势 ---
            st.subheader("因子与股价走势")
            fig_factor = model.plot_factor_price()
            st.pyplot(fig_factor)
            
            win_count, total_count = model.calculate_labels()
            
            # 计算IC
            ic_res = model.calculate_ic()
            
            model.run_bayesian_analysis()
            
            # 展示统计信息
            st.subheader("统计结果")
            col1, col2, col3 = st.columns(3)
            col1.metric("总样本数", total_count)
            col2.metric("满足胜率样本数", win_count)
            col3.metric("全局先验胜率", f"{win_count/total_count:.2%}" if total_count > 0 else "N/A")
            
            # 展示IC
            if ic_res:
                st.markdown("---")
                st.subheader("因子IC (Information Coefficient)")
                ic_col1, ic_col2 = st.columns(2)
                with ic_col1:
                    st.markdown(f"**{model.feature1_name}**")
                    st.write(f"Pearson IC: {ic_res.get('f1_pearson', 0):.4f}")
                    st.write(f"Spearman IC: {ic_res.get('f1_spearman', 0):.4f}")
                with ic_col2:
                    st.markdown(f"**{model.feature2_name}**")
                    st.write(f"Pearson IC: {ic_res.get('f2_pearson', 0):.4f}")
                    st.write(f"Spearman IC: {ic_res.get('f2_spearman', 0):.4f}")
            
            st.markdown("---")
            st.subheader("策略绩效指标")
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            m_col1.metric("先验年化收益", f"{model.ann_ret_prior:.2%}")
            m_col2.metric("后验年化收益", f"{model.ann_ret_posterior:.2%}")
            m_col3.metric("先验>后验占比", f"{model.prob_diff_ratio:.2%}")
            m_col4.metric("先验夏普", f"{model.sharpe_prior:.2f}")
            m_col5.metric("后验夏普", f"{model.sharpe_posterior:.2f}")
            
            # 绘图
            st.subheader("胜率结果图")
            fig1, fig2 = model.plot_results()
            st.pyplot(fig1)
            
            st.subheader("信号策略净值图")
            st.pyplot(fig2)
            
            # 展示数据详情
            with st.expander("查看详细数据"):
                st.dataframe(model.results_df)
