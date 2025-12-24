import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tushare as ts

# ==========================================
# 配置常量 - 放在文件开头
# ==========================================

# Tushare 初始化（从环境变量读取 token）
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '')
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
else:
    pro = None

# 基准指数代码映射
BENCHMARK_CODES = {
    "沪深300": "000300.SH",
    "中证500": "000905.SH", 
    "上证指数": "000001.SH"
}

# ==========================================
# 工具函数定义
# ==========================================

def fetch_stock_data(ts_code, start_date='20140101', end_date=None):
    """
    从 Tushare 获取股票日线数据
    参数：
        ts_code - 股票代码（如 601919.SH）
        start_date - 开始日期（默认 2014-01-01）
        end_date - 结束日期（默认为当前日期）
    """
    if pro is None:
        raise ValueError("Tushare 未初始化，请设置环境变量 TUSHARE_TOKEN")
    
    if end_date is None:
        end_date = pd.Timestamp.now().strftime('%Y%m%d')
    
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').set_index('trade_date')
    
    # 重命名列以匹配原有逻辑
    df = df.rename(columns={
        'close': '收盘',
        'open': '开盘',
        'high': '最高',
        'low': '最低',
        'vol': '成交量',
        'amount': '成交额'
    })
    return df

def fetch_index_data(ts_code, start_date='20140101', end_date=None):
    """
    从 Tushare 获取指数日线数据
    参数：
        ts_code - 指数代码（如 000300.SH）
        start_date - 开始日期（默认 2014-01-01）
        end_date - 结束日期（默认为当前日期）
    """
    if pro is None:
        raise ValueError("Tushare 未初始化，请设置环境变量 TUSHARE_TOKEN")
    
    if end_date is None:
        end_date = pd.Timestamp.now().strftime('%Y%m%d')
    
    df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').set_index('trade_date')
    return df

# ==========================================
# 核心算法逻辑
# ==========================================

def apply_filterpy_kalman(series, Q_val=0.01, R_val=0.1):
    """卡尔曼滤波"""
    from filterpy.kalman import KalmanFilter
    # 确保传入的是 numpy 数组且无空值
    vals = series.fillna(method='ffill').fillna(method='bfill').values
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[vals[0]]])
    kf.F = np.array([[1.]])
    kf.H = np.array([[1.]])
    kf.P *= 10.
    kf.R = R_val
    kf.Q = Q_val
    
    filtered_results = []
    for z in vals:
        kf.predict()
        kf.update(z)
        filtered_results.append(kf.x[0, 0])
    return filtered_results

def FE(original_feature, n_MA, n_D, Y_window, Q_window, feature_name, use_kalman):
    """
    特征工程：智能识别数值列，避开日期列导致的编码错误
    """
    # 1. 自动筛选数值列 (避开日期类型)
    numeric_df = original_feature.select_dtypes(include=[np.number])
    if numeric_df.empty:
        # 如果没有识别出数字列，尝试暴力转换
        numeric_df = original_feature.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    
    if numeric_df.empty:
        st.error("无法在所选表格中找到数值列，请检查数据格式。")
        return pd.DataFrame()

    target_col = numeric_df.columns[0]
    df = pd.DataFrame(index=original_feature.index)
    # 强制转换为 float64，防止 Timestamp 混入
    df['原始数据'] = numeric_df[target_col].astype(float).ffill().bfill()

    if use_kalman:
        df['卡尔曼滤波'] = apply_filterpy_kalman(df['原始数据'])
        data = df['卡尔曼滤波']
    else:
        data = df['原始数据']
        
    for op in feature_name:
        if op == "移动平均":
            for ma in n_MA:
                df[f'移动平均{ma}'] = data.rolling(window=ma).mean()
        if op == "差分":
            for d in n_D:
                df[f'差分{d}'] = data.pct_change(periods=d)
        if op == "一阶导数":
            df['一阶导数'] = data.diff(1)
        if op == "二阶导数":
            df['二阶导数'] = data.diff(1).diff(1)
    
    return df

def set_price_data(stock_data, baselinedata, feature_data, holding_period):
    """计算价格数据和超额收益"""
    # 确保索引对齐
    common_dates = stock_data.index.intersection(baselinedata.index).intersection(feature_data.index).sort_values()
    
    price_data = pd.DataFrame({
        '股价': stock_data.loc[common_dates, '收盘'],
        '基准': baselinedata.loc[common_dates, 'close'],
    }, index=common_dates)
    
    price_data['股价收益率'] = price_data['股价'].pct_change()
    price_data['基准收益率'] = price_data['基准'].pct_change()
    price_data['超额收益率'] = price_data['股价收益率'] - price_data['基准收益率']
    
    # 计算净值
    price_data['超额净值'] = (1 + price_data['超额收益率'].fillna(0)).cumprod()
    price_data['持有期超额收益率'] = price_data['超额净值'].shift(-holding_period) / price_data['超额净值'] - 1
    
    return price_data

def bayesian_analysis(price_data, feature_data, profit_setted, observation_periods, holding_period, f, s):
    """贝叶斯择时分析"""
    common_dates = price_data.index.intersection(feature_data.index).sort_values()
    df = price_data.loc[common_dates].copy()
    
    for col in f:
        df[col] = feature_data.loc[common_dates, col]
    
    df['胜率触发'] = (df['持有期超额收益率'] > profit_setted).astype(int)
    df['胜率不触发'] = 1 - df['胜率触发']
    
    # 贝叶斯核心计算
    pw_early = df['胜率触发'].rolling(window=observation_periods).mean().shift(holding_period + 1)
    pw_late = df['胜率触发'].rolling(window=observation_periods).mean().shift(holding_period + 1)
    cutoff = observation_periods + holding_period
    df['P(W)'] = pw_early
    if len(df) > cutoff:
        df.iloc[cutoff:, df.columns.get_loc('P(W)')] = pw_late.iloc[cutoff:]
    
    # 安全执行策略逻辑
    try:
        df['信号触发'] = eval(s).astype(int)
    except Exception as e:
        st.error(f"策略表达式错误: {e}")
        df['信号触发'] = 0

    # 条件概率 P(C|W) 和 P(C|not W)
    shift_n = holding_period + 1
    df['W_and_C'] = ((df['胜率触发'] == 1) & (df['信号触发'] == 1)).astype(int)
    df['notW_and_C'] = ((df['胜率触发'] == 0) & (df['信号触发'] == 1)).astype(int)
    
    p_c_w = (df['W_and_C'].rolling(observation_periods).sum().shift(shift_n) / 
             df['胜率触发'].rolling(observation_periods).sum().shift(shift_n))
    p_c_notw = (df['notW_and_C'].rolling(observation_periods).sum().shift(shift_n) / 
                df['胜率不触发'].rolling(observation_periods).sum().shift(shift_n))
    
    df['P(W|C)'] = (p_c_w * df['P(W)']) / (p_c_w * df['P(W)'] + p_c_notw * (1 - df['P(W)']))
    
    # 信号生成与仓位
    df['买入信号'] = np.where(
        (df['P(W|C)'] > df['P(W)']) & (df['信号触发'] == 1) & 
        ((df['P(W|C)'] > 0.5) | (df['P(W|C)'] > df['P(W|C)'].shift(1)*0.9)), 1, 0
    )
    df['仓位'] = np.where(df['买入信号'] == 1, 
                        df['信号触发'].rolling(holding_period).sum() / holding_period, 0)
    
    df['仓位净值'] = (1 + (df['仓位'].shift(1) * df['超额收益率'].fillna(0))).cumprod()
    df['先验仓位净值'] = (1 + (df['P(W)'].shift(1) * df['超额收益率'].fillna(0))).cumprod()
    
    return df

# ==========================================
# Streamlit 界面
# ==========================================

st.set_page_config(page_title="贝叶斯择时回测平台", layout="wide")
st.title("贝叶斯择时回测平台")

# 初始化会话状态
if 'feature_data_after' not in st.session_state:
    st.session_state['feature_data_after'] = None

# ==========================================
# 侧边栏：数据源配置
# ==========================================

st.sidebar.header("📁 数据配置")

# Tushare 状态检查
if pro is None:
    st.sidebar.error("⚠️ 未检测到 TUSHARE_TOKEN 环境变量")
    st.sidebar.info("请在系统中设置环境变量 TUSHARE_TOKEN")
else:
    st.sidebar.success("✅ Tushare 已连接")

# 1. 因子文件上传
factor_file = st.sidebar.file_uploader("上传因子数据", type=['xlsx', 'xls', 'csv'])
if factor_file is not None:
    try:
        # 根据文件类型选择读取方式
        if factor_file.name.endswith('.csv'):
            df_factor = pd.read_csv(factor_file)
        else:
            df_factor = pd.read_excel(factor_file)
        
        # 自动寻找日期列并设为索引
        for col in df_factor.columns:
            if '日期' in str(col) or 'Date' in str(col) or 'time' in str(col).lower():
                try:
                    df_factor[col] = pd.to_datetime(df_factor[col])
                    df_factor = df_factor.set_index(col)
                except Exception:
                    pass
                break
        st.session_state['raw_feature_df'] = df_factor
        st.sidebar.success("✅ 已上传因子文件")
        st.sidebar.caption(f"列数: {len(df_factor.columns)}")
    except Exception as e:
        st.sidebar.error(f"❌ 读取因子文件失败: {e}")

# 2. 输入标的股票代码
stock_selected = st.sidebar.text_input(
    "输入标的股票代码", 
    value="601919.SH",
    placeholder="例如: 601919.SH",
    help="请输入完整的股票代码，如 601919.SH (中国神华)"
)

# 3. 选择基准指数
baseline_selected = st.sidebar.selectbox(
    "选择基准指数", 
    list(BENCHMARK_CODES.keys()),
    index=0
)

# ==========================================
# 侧边栏：参数配置
# ==========================================

st.sidebar.divider()

# 数据处理参数
st.sidebar.subheader("数据处理参数")
use_kalman = st.sidebar.checkbox("启用卡尔曼滤波", value=True)

# 特征工程算子
features_op = st.sidebar.multiselect(
    "特征算子", 
    ["移动平均", "差分", "一阶导数", "二阶导数"], 
    default=["一阶导数"],
    help="选择用于生成特征的数学算子"
)

# 算子参数
if "移动平均" in features_op:
    n_MA = st.sidebar.slider("移动平均窗口", 1, 60, 5, help="计算移动平均的时间窗口大小")
else:
    n_MA = 5

if "差分" in features_op:
    n_D = st.sidebar.slider("差分期数", 1, 365, 1, help="计算差分的滞后期数")
else:
    n_D = 1

st.sidebar.divider()

# 贝叶斯参数
st.sidebar.subheader("贝叶斯参数")
op = st.sidebar.slider(
    "观察期（天数）", 
    1, 365, 60, 
    help="用于计算先验概率的历史观察窗口"
)

st.sidebar.divider()

# 信号生成参数
st.sidebar.subheader("信号生成参数")
hp = st.sidebar.slider(
    "持有期（天数）", 
    1, 365, 5, 
    help="持有仓位的时间周期"
)
profit_target = st.sidebar.number_input(
    "超额收益阈值", 
    value=0.0, 
    step=0.01,
    format="%.3f",
    help="定义盈利的超额收益率阈值"
)
s_input = st.sidebar.text_area(
    "策略逻辑 (Python格式)", 
    value="df['一阶导数'] < 0",
    help="使用 df['列名'] 引用特征，支持逻辑运算符"
)

st.sidebar.divider()

# 可视化配置
if st.session_state.get('feature_data_after') is not None:
    st.sidebar.subheader("可视化配置")
    available_factors = st.session_state['feature_data_after'].columns.tolist()
    default_factors = st.session_state.get('selected_plot_factors', available_factors)
    selected_factors = st.sidebar.multiselect(
        "选择绘制的因子", 
        available_factors, 
        default=default_factors
    )
    st.session_state['selected_plot_factors'] = selected_factors

# ==========================================
# 主界面：执行按钮
# ==========================================

# 一键执行：特征工程 + 回测分析
if st.button("🚀 执行回测分析", use_container_width=True):
    if 'raw_feature_df' not in st.session_state:
        st.error("❌ 请先在左侧上传因子数据！")
    elif pro is None:
        st.error("❌ Tushare 未初始化，请设置环境变量 TUSHARE_TOKEN")
    else:
        with st.spinner('执行回测分析中...'):
            # ========== 第一步：执行特征工程 ==========
            raw_f = st.session_state['raw_feature_df']
            processed_fe = FE(raw_f, [n_MA], [n_D], 12, 12, features_op, use_kalman)
            st.session_state['feature_data_after'] = processed_fe
            
            # ========== 第二步：贝叶斯回测 ==========
            fe_data = st.session_state['feature_data_after']
            
            try:
                # ========== 从 Tushare 获取标的股票数据 ==========
                stock_raw = fetch_stock_data(stock_selected)
                
                # ========== 从 Tushare 获取基准指数数据 ==========
                baseline_code = BENCHMARK_CODES[baseline_selected]
                baseline_raw = fetch_index_data(baseline_code)

            except Exception as e:
                st.error(f"❌ 数据获取失败: {e}")
                st.stop()

            # ========== 执行回测计算 ==========
            p_data = set_price_data(stock_raw, baseline_raw, fe_data, hp)
            df_res = bayesian_analysis(p_data, fe_data, profit_target, op, hp, fe_data.columns.tolist(), s_input)

            # ========== 结果展示 ==========
            final_nav = df_res['仓位净值'].iloc[-1]
            prior_nav = df_res['先验仓位净值'].iloc[-1]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("策略净值", f"{final_nav:.3f}", f"{(final_nav-1):.2%}")
            c2.metric("先验净值", f"{prior_nav:.3f}", f"{(prior_nav-1):.2%}", delta_color="off")
            c3.metric("超额增益", f"{(final_nav-prior_nav):.2%}")

            # ========== 因子与超额收益走势图 ==========
            st.subheader("📈 因子与超额收益走势")
            fig_factor = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 左轴：超额净值
            fig_factor.add_trace(
                go.Scatter(x=df_res.index, y=df_res['超额净值'], name='超额净值', line=dict(color='blue', width=2)),
                secondary_y=False
            )
            
            # 右轴：因子
            exclude_cols = ['股价', '基准', '股价收益率', '基准收益率', '超额收益率', '超额净值', '持有期超额收益率', 
                          '胜率触发', '胜率不触发', 'P(W)', '信号触发', 'W_and_C', 'notW_and_C', 'P(W|C)', 
                          '买入信号', '仓位', '仓位净值', '先验仓位净值']
            selected_factors = st.session_state.get('selected_plot_factors', [])
            if selected_factors:
                feature_cols = [c for c in selected_factors if c in df_res.columns and c not in exclude_cols]
            else:
                feature_cols = [c for c in df_res.columns if c not in exclude_cols]
            
            colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
            for i, col in enumerate(feature_cols):
                color = colors[i % len(colors)]
                fig_factor.add_trace(
                    go.Scatter(x=df_res.index, y=df_res[col], name=f'因子: {col}', 
                              line=dict(color=color, width=1, dash='dot')),
                    secondary_y=True
                )
                
            fig_factor.update_yaxes(title_text="超额净值", secondary_y=False)
            fig_factor.update_yaxes(title_text="因子值", secondary_y=True)
            fig_factor.update_layout(height=500, template="plotly_white", hovermode="x unified")
            
            st.plotly_chart(fig_factor, use_container_width=True)

            # ========== 贝叶斯分析结果图 ==========
            fig = make_subplots(
                rows=2, cols=2, 
                subplot_titles=("胜率修正", "净值表现", "信号触发", "实时仓位"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": True}]]
            )
            
            # 子图1: 胜率修正
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['P(W)'], name='先验', 
                                    line=dict(color='orange')), 1, 1)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['P(W|C)'], name='后验', 
                                    line=dict(color='grey', dash='dot')), 1, 1)
            
            # 子图2: 净值表现
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['仓位净值'], name='策略仓位净值', 
                                    line=dict(color='red')), 1, 2)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['先验仓位净值'], name='先验仓位净值', 
                                    line=dict(color='grey')), 1, 2)

            # 子图3: 信号触发
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['超额净值'], name='超额净值', 
                                    line=dict(color='blue', width=1.5)), 2, 1)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['信号触发'], name='触发脉冲', 
                                    fill='tozeroy', line=dict(width=0),
                                    fillcolor='rgba(255, 165, 0, 0.2)'), 2, 1)
            
            # 子图4: 实时仓位
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['超额净值'], name='超额净值', 
                                    line=dict(color='blue', width=2),
                                    hovertemplate='日期: %{x}<br>超额净值: %{y:.4f}<extra></extra>'), 
                         row=2, col=2, secondary_y=False)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['仓位'], name='策略仓位', 
                                    fill='tozeroy', line_shape='hv', 
                                    line=dict(color='rgba(255, 165, 0, 0.8)', width=1), 
                                    fillcolor='rgba(255, 165, 0, 0.2)', 
                                    hovertemplate='日期: %{x}<br>当前仓位: %{y:.2f}<extra></extra>'), 
                         row=2, col=2, secondary_y=True)
            
            fig.update_yaxes(title_text="净值水平", secondary_y=False, row=2, col=2)
            fig.update_yaxes(title_text="仓位权重", range=[0, 1.1], secondary_y=True, row=2, col=2)
            
            fig.update_layout(height=700, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
