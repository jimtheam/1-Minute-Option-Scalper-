import streamlit as st
import requests
import pandas as pd
# 移除 pandas_ta，改用原生计算
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import yfinance as yf
import os

# ================= 1. 核心配置 =================

# 从环境变量获取 Token，本地测试如果没有设置环境变量，可以用后面的默认值
TRADIER_ACCESS_TOKEN = os.environ.get("TRADIER_ACCESS_TOKEN")
BASE_URL = "https://sandbox.tradier.com/v1" 

HEADERS = {
    "Authorization": f"Bearer {TRADIER_ACCESS_TOKEN}",
    "Accept": "application/json"
}

# ================= 2. 数据获取 =================

def get_intraday_data_yf(symbol):
    """获取 1分钟 K线数据"""
    try:
        # 下载数据
        df = yf.download(symbol, period='1d', interval='1m', progress=False)
        
        if df is None or df.empty:
            return None
            
        # 修复 yfinance MultiIndex 列名问题
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 转为小写
        df.columns = [str(c).lower() for c in df.columns]
        
        # 确保有 volume
        if 'volume' not in df.columns:
            df['volume'] = 0
            
        # 处理时区
        if df.index.tz is not None:
            df.index = df.index.tz_convert('US/Eastern').tz_localize(None)
            
        return df
    except Exception as e:
        print(f"Data Error: {e}")
        return None

def get_0dte_option(symbol, sentiment):
    """从 Tradier 获取期权链"""
    if not TRADIER_ACCESS_TOKEN:
        return None, "No Token"

    try:
        # 1. 获取 Expirations
        exp_res = requests.get(f"{BASE_URL}/markets/options/expirations", headers=HEADERS, params={'symbol': symbol})
        if exp_res.status_code != 200: return None, None
        exps = exp_res.json().get('expirations', {}).get('date', [])
        
        if not exps: return None, None
        if isinstance(exps, str): exps = [exps]
        target_exp = exps[0] # 最近到期日
        
        # 2. 获取 Chain
        params = {'symbol': symbol, 'expiration': target_exp, 'greeks': 'true'}
        chain_res = requests.get(f"{BASE_URL}/markets/options/chains", headers=HEADERS, params=params)
        options = chain_res.json().get('options', {}).get('option', [])
        
        if not options: return None, None
        df = pd.DataFrame(options)
        
        # 筛选
        target_type = 'call' if sentiment == 'SCALP_LONG' else 'put'
        if 'option_type' in df.columns:
            df = df[df['option_type'] == target_type].copy()
        
        # 格式化
        cols = ['bid', 'ask', 'strike', 'volume']
        for c in cols:
            if c in df.columns: df[c] = df[c].fillna(0).astype(float)
            
        if 'greeks' in df.columns:
            df['delta'] = df['greeks'].apply(lambda x: x.get('delta', 0) if isinstance(x, dict) else 0)
        else:
            df['delta'] = 0
            
        # 计算价差
        if 'ask' in df.columns and 'bid' in df.columns:
             # 过滤非法数据
            df = df[df['ask'] > 0]
            df['spread_pct'] = ((df['ask'] - df['bid']) / df['ask']) * 100
        else:
            df['spread_pct'] = 0
            
        return df.sort_values('volume', ascending=False).head(5), target_exp
    except:
        return None, None

# ================= 3. 指标计算 (原生 Pandas 版) =================

def calculate_indicators(df):
    """
    完全不依赖 pandas_ta，手动计算 VWAP, EMA, MACD
    保证在 Render 上 100% 能运行
    """
    df = df.copy()
    
    # 1. 计算 EMA 9
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    
    # 2. 计算 MACD (12, 26, 9)
    # EMA 12
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    # EMA 26
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    # MACD Line
    df['MACD_LINE'] = ema12 - ema26
    # Signal Line
    df['MACD_SIGNAL'] = df['MACD_LINE'].ewm(span=9, adjust=False).mean()
    # Histogram
    df['MACD_HIST'] = df['MACD_LINE'] - df['MACD_SIGNAL']
    
    # 3. 计算 VWAP (日内)
    # VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
    # 这里的简单实现假设 df 是一天的数据
    v = df['volume'].values
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    
    return df

# ================= 4. 策略引擎 =================

def run_scalping_algo(df):
    # 计算指标
    df = calculate_indicators(df)
    
    # 获取最新数据
    curr = df.iloc[-1]
    
    price = curr['close']
    vwap = curr['VWAP']
    ema9 = curr['EMA_9']
    hist = curr['MACD_HIST']
    
    signal = "NEUTRAL"
    reasons = []
    
    # 策略逻辑
    # 多头: 价格在 VWAP 和 EMA9 之上，且 MACD 柱子 > 0
    if price > vwap and price > ema9:
        if hist > 0:
            signal = "SCALP_LONG"
            reasons = ["Price > VWAP", "Price > EMA9", "MACD Green"]
    
    # 空头: 价格在 VWAP 和 EMA9 之下，且 MACD 柱子 < 0
    elif price < vwap and price < ema9:
        if hist < 0:
            signal = "SCALP_SHORT"
            reasons = ["Price < VWAP", "Price < EMA9", "MACD Red"]
            
    return df, signal, reasons

# ================= 5. 页面显示 =================

st.set_page_config(page_title="1-Min Scalper", layout="wide")
st.title("⚡ 1-Minute Option Scalper (Cloud Ready)")

# 侧边栏
with st.sidebar:
    st.info("Status: Online")
    input_symbol = st.text_input("Symbol", "NVDA").upper()
    if not TRADIER_ACCESS_TOKEN:
        st.warning("⚠️ 未检测到 Tradier Token。期权链将不可用。请在 Render 后台 Environment Variables 添加 TRADIER_ACCESS_TOKEN。")

# 主界面
# 获取数据
with st.spinner(f"Fetching {input_symbol}..."):
    df_min = get_intraday_data_yf(input_symbol)

if df_min is not None and not df_min.empty:
    # 运行策略
    df_res, signal, reasons = run_scalping_algo(df_min)
    
    last_row = df_res.iloc[-1]
    last_price = last_row['close']
    last_time = df_res.index[-1].strftime('%H:%M')
    
    # 1. 顶部状态
    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"${last_price:.2f}", last_time)
    
    if signal == "SCALP_LONG":
        c2.error("🟢 CALL (LONG)")
    elif signal == "SCALP_SHORT":
        c2.error("🔴 PUT (SHORT)")
    else:
        c2.info("⚪ WAIT")
        
    c3.write(" | ".join(reasons) if reasons else "No Signal")
    
    # 2. 图表 (Plotly)
    df_plot = df_res.tail(60) # 只看最近60分钟
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    # K线
    fig.add_trace(go.Candlestick(x=df_plot.index,
                    open=df_plot['open'], high=df_plot['high'],
                    low=df_plot['low'], close=df_plot['close'], name='Price'), row=1, col=1)
    
    # VWAP & EMA
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['VWAP'], name='VWAP', line=dict(color='orange', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA_9'], name='EMA9', line=dict(color='cyan', width=1)), row=1, col=1)
    
    # 成交量
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['volume'], name='Vol', marker_color='gray'), row=2, col=1)
    
    fig.update_layout(height=550, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. 期权部分
    if signal != "NEUTRAL":
        st.subheader("Options Chain")
        recs, exp = get_0dte_option(input_symbol, signal)
        
        if recs is not None:
            st.caption(f"Expiration: {exp}")
            st.dataframe(recs[['description', 'strike', 'delta', 'volume', 'ask', 'spread_pct']])
        else:
            if not TRADIER_ACCESS_TOKEN:
                st.error("需要 API Key 才能查看期权数据。")
            else:
                st.warning("无可用期权数据。")

else:
    st.error("Data fetch failed. Market might be closed or symbol invalid.")