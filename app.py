import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 Projeção de Peso Total (Ton) até Julho/2027")

# 1) Upload do Excel
uploaded_file = st.file_uploader("4600672730_Prog_Process_10.04.2025.xlsx", type=["xlsx"])
if uploaded_file is None:
    st.info("Aguardando o upload do arquivo...")
    st.stop()

# 2) Leitura e pré‑processamento
df = pd.read_excel(uploaded_file, engine="openpyxl")
df['Fim Real Caldeiraria'] = pd.to_datetime(df['Fim Real Caldeiraria'], errors='coerce')
df = df.dropna(subset=['Fim Real Caldeiraria', 'Peso Total (Ton)'])
df = df[(df['Fim Real Caldeiraria'] >= "2023-06-01") & (df['Fim Real Caldeiraria'] <= pd.to_datetime("today"))]

# 3) Séries mensais
media_mensal = df.groupby(pd.Grouper(key='Fim Real Caldeiraria', freq='M'))['Peso Total (Ton)'].mean()
soma_mensal  = df.groupby(pd.Grouper(key='Fim Real Caldeiraria', freq='M'))['Peso Total (Ton)'].sum()

# 4) Parâmetros de projeção
data_fim = pd.to_datetime("2027-07-31")
n_periodos = (data_fim.to_period("M") - media_mensal.index[-1].to_period("M")).n + 1
datas_futuras = pd.date_range(start=media_mensal.index[-1] + pd.offsets.MonthEnd(),
                              periods=n_periodos, freq='M')

# 5) ARIMA e forecasts para média
modelo_media = ARIMA(media_mensal, order=(1,0,3)).fit()
f_media = modelo_media.get_forecast(steps=n_periodos)
m_real = f_media.predicted_mean
ic_m   = f_media.conf_int(alpha=0.20)
m_otim = ic_m.iloc[:,1]
m_pess = ic_m.iloc[:,0]

# 6) ARIMA e forecasts para soma mensal
modelo_soma = ARIMA(soma_mensal, order=(1,0,3)).fit()
f_soma = modelo_soma.get_forecast(steps=n_periodos)
s_real = f_soma.predicted_mean
ic_s   = f_soma.conf_int(alpha=0.20)
s_otim = ic_s.iloc[:,1]
s_pess = ic_s.iloc[:,0]

# 7) Acumulado futuro
ultimo_acum = soma_mensal.cumsum().iloc[-1]
a_real = np.cumsum(s_real) + ultimo_acum
a_otim = np.cumsum(s_otim) + ultimo_acum
a_pess = np.cumsum(s_pess) + ultimo_acum

# 8) Plot Média Mensal
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=media_mensal.index, y=media_mensal,
                          mode='lines+markers', name='Histórico'))
fig1.add_trace(go.Scatter(x=datas_futuras, y=m_real, mode='lines', name='Realista'))
fig1.add_trace(go.Scatter(x=datas_futuras, y=m_otim, mode='lines', name='Otimista', line=dict(dash='dash')))
fig1.add_trace(go.Scatter(x=datas_futuras, y=m_pess, mode='lines', name='Pessimista', line=dict(dash='dash')))
fig1.update_layout(title="Média Mensal (Ton) até Jul/2027",
                   xaxis_title="Data", yaxis_title="Toneladas")

# 9) Plot Acumulado
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=soma_mensal.index, y=soma_mensal.cumsum(),
                          mode='lines+markers', name='Acumulado Real'))
fig2.add_trace(go.Scatter(x=datas_futuras, y=a_real, mode='lines', name='Realista'))
fig2.add_trace(go.Scatter(x=datas_futuras, y=a_otim, mode='lines', name='Otimista', line=dict(dash='dash')))
fig2.add_trace(go.Scatter(x=datas_futuras, y=a_pess, mode='lines', name='Pessimista', line=dict(dash='dash')))
fig2.update_layout(title="Acumulado (Ton) até Jul/2027",
                   xaxis_title="Data", yaxis_title="Toneladas Acumuladas")

# 10) Exibir no Streamlit
st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)
