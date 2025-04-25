import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 Projeção de Peso Total (Ton) até Julho/2027")

# 1) Leitura direta do GitHub
url = "https://raw.githubusercontent.com/Marlonb87/app.py/main/4600672730_Prog_Process_22.04.2025.xlsx"
df = pd.read_excel(url, engine="openpyxl")


# 2) Leitura e pré-processamento
df = pd.read_excel(url, engine="openpyxl")
df['Fim Real Caldeiraria'] = pd.to_datetime(df['Fim Real Caldeiraria'], errors='coerce')
df = df.dropna(subset=['Fim Real Caldeiraria', 'Peso Total (Ton)'])
df = df[df['Fim Real Caldeiraria'] <= pd.to_datetime("today")]

# 3) Série mensal (SOMA) – como no Jupyter
serie_mensal = df.groupby(pd.Grouper(key='Fim Real Caldeiraria', freq='M'))['Peso Total (Ton)'].sum()
serie_mensal = serie_mensal.last('24M')  # opcional: limitar aos últimos 23 meses

# 4) Parâmetros de projeção
data_fim = pd.to_datetime("2027-07-31")
n_periodos = (data_fim.to_period("M") - serie_mensal.index[-1].to_period("M")).n + 1
datas_futuras = pd.date_range(start=serie_mensal.index[-1] + pd.offsets.MonthEnd(), periods=n_periodos, freq='M')

# 5) Modelo ARIMA e previsão
modelo = ARIMA(serie_mensal, order=(1, 0, 3)).fit()
forecast = modelo.get_forecast(steps=n_periodos)
previsao_real = forecast.predicted_mean
ic = forecast.conf_int(alpha=0.20)
previsao_otim = ic.iloc[:, 1]
previsao_pess = ic.iloc[:, 0]

# 6) Acumulado futuro
ultimo_acum = serie_mensal.cumsum().iloc[-1]
acum_real = np.cumsum(previsao_real) + ultimo_acum
acum_otim = np.cumsum(previsao_otim) + ultimo_acum
acum_pess = np.cumsum(previsao_pess) + ultimo_acum

# 7) Plot – Soma mensal
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=serie_mensal.index, y=serie_mensal,
                          mode='lines+markers', name='Histórico'))
fig1.add_trace(go.Scatter(x=datas_futuras, y=previsao_real, mode='lines', name='Realista'))
fig1.add_trace(go.Scatter(x=datas_futuras, y=previsao_otim, mode='lines', name='Otimista', line=dict(dash='dash')))
fig1.add_trace(go.Scatter(x=datas_futuras, y=previsao_pess, mode='lines', name='Pessimista', line=dict(dash='dash')))
fig1.update_layout(title="📈 Soma Mensal (Ton) até Jul/2027",
                   xaxis_title="Data", yaxis_title="Toneladas")

# 8) Plot – Acumulado
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=serie_mensal.index, y=serie_mensal.cumsum(),
                          mode='lines+markers', name='Acumulado Real'))
fig2.add_trace(go.Scatter(x=datas_futuras, y=acum_real, mode='lines', name='Realista'))
fig2.add_trace(go.Scatter(x=datas_futuras, y=acum_otim, mode='lines', name='Otimista', line=dict(dash='dash')))
fig2.add_trace(go.Scatter(x=datas_futuras, y=acum_pess, mode='lines', name='Pessimista', line=dict(dash='dash')))
fig2.update_layout(title="📊 Acumulado (Ton) até Jul/2027",
                   xaxis_title="Data", yaxis_title="Toneladas Acumuladas")

# 9) Exibição
st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)
