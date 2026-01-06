import streamlit as st
import requests
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import yfinance as yf  # 引入 yfinance

# ================= 1. 核心配置 =================

# 🔴 请确保填入 Token (如果没有实盘Token，期权部分会显示空白，但K线图会正常显示)
import os  # <--- 必须先导入这个库

# 括号里填的是 "变量名"，不是你的真实 Key
# 真实的 Key 要去 Render 网站的后台填
TRADIER_ACCESS_TOKEN = os.environ.get("TRADIER_ACCESS_TOKEN")

# 为了防止本地运行报错，可以加个默认值（可选）：
if not TRADIER_ACCESS_TOKEN:
    # ⚠️ 注意：这行只用于本地测试，推送到 GitHub 前建议删掉，防止泄露
    TRADIER_ACCESS_TOKEN = "vmuqxOFguYZWjqj34pujliCbBISI" 
BASE_URL = "https://sandbox.tradier.com/v1" 

HEADERS = {
    "Authorization": f"Bearer {TRADIER_ACCESS_TOKEN}",
    "Accept": "application/json"
}

# ================= 2. 数据获取 (修复版) =================

def get_intraday_data_yf(symbol):
    """
    使用 yfinance 获取 1分钟 K线数据 (修复 MultiIndex 报错问题)
    """
    try:
        # 获取最近 1 天的 1分钟数据
        df = yf.download(symbol, period='1d', interval='1m', progress=False)
        
        if df is None or df.empty:
            return None
            
        # [关键修复] 处理 yfinance 返回的 MultiIndex 列名问题
        # 如果列名是元组 (例如: ('Open', 'AMD'))，强制取第一层
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 统一转为小写字符串 (Open -> open)
        df.columns = [str(c).lower() for c in df.columns]
        
        # 确保包含 volume 列
        if 'volume' not in df.columns:
            df['volume'] = 0
            
        # 处理时区问题，转换为美东时间并移除时区信息以便绘图
        if df.index.tz is not None:
            df.index = df.index.tz_convert('US/Eastern').tz_localize(None)
            
        return df
    except Exception as e:
        st.error(f"yfinance 数据获取失败: {e}")
        return None

def get_0dte_option(symbol, sentiment):
    """从 Tradier 获取期权链"""
    # 1. 获取到期日
    try:
        exp_res = requests.get(f"{BASE_URL}/markets/options/expirations", headers=HEADERS, params={'symbol': symbol})
        if exp_res.status_code != 200: return None, None
        exps = exp_res.json().get('expirations', {}).get('date', [])
    except: return None, None

    if not exps: return None, None
    if isinstance(exps, str): exps = [exps]
    
    # 取最近的一个到期日
    target_exp = exps[0]
    
    # 2. 获取期权链
    params = {'symbol': symbol, 'expiration': target_exp, 'greeks': 'true'}
    try:
        chain_res = requests.get(f"{BASE_URL}/markets/options/chains", headers=HEADERS, params=params)
        if chain_res.status_code != 200: return None, None
        options = chain_res.json().get('options', {}).get('option', [])
    except: return None, None
    
    if not options: return None, None
    
    df = pd.DataFrame(options)
    
    # 容错处理
    if 'option_type' not in df.columns: return None, None
    
    target_type = 'call' if sentiment == 'SCALP_LONG' else 'put'
    df = df[df['option_type'] == target_type].copy()
    
    # 数据转换
    cols_to_float = ['bid', 'ask', 'strike']
    for c in cols_to_float:
        if c in df.columns: df[c] = df[c].fillna(0).astype(float)
        
    if 'volume' in df.columns:
        df['volume'] = df['volume'].fillna(0).astype(int)
    else:
        df['volume'] = 0
    
    # 解析 Delta
    if 'greeks' in df.columns:
        df['delta'] = df['greeks'].apply(lambda x: x.get('delta', 0) if isinstance(x, dict) else 0)
    else:
        df['delta'] = 0
        
    # 计算价差
    if 'ask' in df.columns and 'bid' in df.columns:
        # 避免分母为0
        df = df[df['ask'] > 0]
        df['spread_pct'] = ((df['ask'] - df['bid']) / df['ask']) * 100
    else:
        df['spread_pct'] = 0
        
    return df.sort_values('volume', ascending=False).head(5), target_exp

# ================= 3. 策略引擎 =================

def run_scalping_algo(df):
    # 复制一份，避免 SettingWithCopyWarning
    df = df.copy()
    
    # 1. VWAP (成交量加权均价)
    # 如果数据不足导致 VWAP 计算失败，使用均线兜底
    try:
        if len(df) > 0:
            df.ta.vwap(append=True)
            # pandas_ta 生成的 VWAP 列名通常是 VWAP_D
            vwap_col = 'VWAP_D'
        else:
            raise ValueError("Empty Data")
    except:
        df['VWAP_D'] = df['close'].rolling(20).mean()
        vwap_col = 'VWAP_D'

    # 2. EMA
    df.ta.ema(length=9, append=True)
    ema9_col = 'EMA_9'
    
    # 3. MACD
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # 动态寻找 MACD 列名
    cols = df.columns
    hist_col = None
    try:
        hist_cols = [c for c in cols if c.startswith('MACDh_')]
        if hist_cols:
            hist_col = hist_cols[0]
    except:
        pass

    # 获取最后一行
    curr = df.iloc[-1]
    
    price = curr['close']
    vwap_val = curr[vwap_col] if vwap_col in curr else price
    ema9_val = curr[ema9_col] if ema9_col in curr else price
    hist_val = curr[hist_col] if hist_col and hist_col in curr else 0
    
    signal = "NEUTRAL"
    reasons = []
    
    # 1分钟极速策略逻辑
    # 做多: 价格 > VWAP 且 价格 > EMA9 且 MACD 柱子为正
    if price > vwap_val and price > ema9_val:
        if hist_val > 0:
            signal = "SCALP_LONG"
            reasons = ["Price > VWAP (Trend Up)", "Price > EMA9 (Momentum)", "MACD Bullish"]
    
    # 做空: 价格 < VWAP 且 价格 < EMA9 且 MACD 柱子为负
    elif price < vwap_val and price < ema9_val:
        if hist_val < 0:
            signal = "SCALP_SHORT"
            reasons = ["Price < VWAP (Trend Down)", "Price < EMA9 (Momentum)", "MACD Bearish"]
            
    return df, signal, reasons, vwap_col, ema9_col

# ================= 4. 页面显示 =================

st.set_page_config(page_title="1-Min Scalper", layout="wide")
st.title("⚡ 1-Minute Option Scalper (Fixed)")

# 控制栏
col_input, col_refresh = st.columns([2, 1])
with col_input:
    symbol = st.text_input("Symbol", "AMD").upper() # 默认改成 AMD 方便你测试
with col_refresh:
    auto_refresh = st.checkbox("Auto-Refresh (30s)", value=False)

# 主体逻辑
placeholder = st.empty()

with placeholder.container():
    # 获取数据
    with st.spinner(f"Downloading {symbol} data..."):
        df_min = get_intraday_data_yf(symbol)
    
    if df_min is not None and not df_min.empty:
        # 运行策略
        df_res, signal, reasons, v_col, e_col = run_scalping_algo(df_min)
        
        last_price = df_res.iloc[-1]['close']
        last_time = df_res.index[-1].strftime('%H:%M')
        
        # 1. 顶部指标
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{symbol} Price", f"${last_price:.2f}", last_time)
        
        if signal == "SCALP_LONG":
            m2.error("🟢 CALL (LONG)") # 绿色
        elif signal == "SCALP_SHORT":
            m2.error("🔴 PUT (SHORT)") # 红色
        else:
            m2.info("⚪ WAIT")
            
        m3.write(" | ".join(reasons) if reasons else "Waiting for setup...")
        
        # 2. 绘图 (最后 60 分钟)
        df_plot = df_res.tail(60)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # K线
        fig.add_trace(go.Candlestick(x=df_plot.index,
                        open=df_plot['open'], high=df_plot['high'],
                        low=df_plot['low'], close=df_plot['close'], name='Price'), row=1, col=1)
        
        # VWAP & EMA
        if v_col in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[v_col], name='VWAP', line=dict(color='orange', width=2)), row=1, col=1)
        if e_col in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[e_col], name='EMA9', line=dict(color='cyan', width=1)), row=1, col=1)
            
        # 成交量
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['volume'], name='Vol', marker_color='gray'), row=2, col=1)
        
        # 黑色背景设置
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. 期权链 (容错保护)
        if signal != "NEUTRAL":
            st.markdown("### ⚡ Quick Options Chain")
            try:
                recs, exp = get_0dte_option(symbol, signal)
                if recs is not None:
                    st.caption(f"Expiration: {exp}")
                    st.dataframe(recs[['description', 'strike', 'delta', 'volume', 'ask', 'spread_pct']])
                else:
                    st.warning("无期权数据 (请检查 Tradier API Token 是否正确，或者该股票此时段无交易)")
            except Exception as e:
                st.error(f"期权模块错误: {e}")

    else:
        st.error(f"❌ 无法获取 {symbol} 数据。可能原因：\n1. 股票代码输入错误\n2. 网络连接问题\n3. 市场未开盘")

# 自动刷新
if auto_refresh:
    time.sleep(30)
    st.rerun()