import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import os

os.makedirs("C:/PowerBI/graficos", exist_ok=True)
st.set_page_config(layout="wide")
st.title("📊 Projeção de Peso Total (Ton) até Julho/2027")

# 1) Leitura direta do GitHub
url = "https://raw.githubusercontent.com/Marlonb87/app.py/main//4600672730_Prog_Process_08.05.2025.xlsx"
df = pd.read_excel(url, engine="openpyxl")

# 2) Pré-processamento
df['Fim Real Caldeiraria'] = pd.to_datetime(df['Fim Real Caldeiraria'], errors='coerce')
df = df.dropna(subset=['Fim Real Caldeiraria', 'Peso Total (Ton)'])
df = df[df['Fim Real Caldeiraria'] <= pd.to_datetime("today")]

# 3) Série mensal
serie_mensal = df.groupby(pd.Grouper(key='Fim Real Caldeiraria', freq='M'))['Peso Total (Ton)'].sum()
serie_mensal = serie_mensal.last('24M')

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

# 7) Gráfico: Soma mensal
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=serie_mensal.index, y=serie_mensal,
                          mode='lines+markers', name='Histórico'))
fig1.add_trace(go.Scatter(x=datas_futuras, y=previsao_real, mode='lines', name='Realista'))
fig1.add_trace(go.Scatter(x=datas_futuras, y=previsao_otim, mode='lines', name='Otimista', line=dict(dash='dash')))
fig1.add_trace(go.Scatter(x=datas_futuras, y=previsao_pess, mode='lines', name='Pessimista', line=dict(dash='dash')))

# Destaques máximo e mínimo
max_val = serie_mensal.max()
max_data = serie_mensal.idxmax()
min_val = serie_mensal.min()
min_data = serie_mensal.idxmin()
fig1.add_trace(go.Scatter(x=[max_data], y=[max_val], mode='markers+text',
                          marker=dict(color='green', size=10),
                          text=["⬆ Máximo"], textposition="top center"))
fig1.add_trace(go.Scatter(x=[min_data], y=[min_val], mode='markers+text',
                          marker=dict(color='red', size=10),
                          text=["⬇ Mínimo"], textposition="bottom center"))

fig1.update_layout(title="📈 Soma Mensal (Ton) até Jul/2027",
                   xaxis_title="Data", yaxis_title="Toneladas")

# 8) Gráfico: Acumulado
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=serie_mensal.index, y=serie_mensal.cumsum(),
                          mode='lines+markers', name='Acumulado Real'))
fig2.add_trace(go.Scatter(x=datas_futuras, y=acum_real, mode='lines', name='Realista'))
fig2.add_trace(go.Scatter(x=datas_futuras, y=acum_otim, mode='lines', name='Otimista', line=dict(dash='dash')))
fig2.add_trace(go.Scatter(x=datas_futuras, y=acum_pess, mode='lines', name='Pessimista', line=dict(dash='dash')))
fig2.update_layout(title="📊 Acumulado (Ton) até Jul/2027",
                   xaxis_title="Data", yaxis_title="Toneladas Acumuladas")

# 9) Gráfico: Variação percentual mês a mês
variacao_mensal = serie_mensal.pct_change() * 100
fig_var = go.Figure()
fig_var.add_trace(go.Scatter(x=variacao_mensal.index, y=variacao_mensal,
                             mode='lines+markers', name='% Variação Mensal'))
fig_var.update_layout(title="📉 Variação Percentual Mês a Mês",
                      xaxis_title="Data", yaxis_title="Variação (%)")

# 10) Gráfico: Barras por mês e ano
df['Ano'] = df['Fim Real Caldeiraria'].dt.year
df['Mês'] = df['Fim Real Caldeiraria'].dt.month_name().str[:3]
pivot = df.pivot_table(index='Mês', columns='Ano', values='Peso Total (Ton)', aggfunc='sum')
pivot = pivot.fillna(0).reindex(index=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                                       'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

fig_bar = go.Figure()
for ano in pivot.columns:
    fig_bar.add_trace(go.Bar(name=str(ano), x=pivot.index, y=pivot[ano]))
fig_bar.update_layout(barmode='stack',
                      title="📊 Produção por Mês (Empilhado por Ano)",
                      xaxis_title="Mês", yaxis_title="Toneladas")

# 11) Tabela de projeções
df_proj = pd.DataFrame({
    'Data': datas_futuras,
    'Realista': previsao_real,
    'Otimista': previsao_otim,
    'Pessimista': previsao_pess,
    'Acum Realista': acum_real,
    'Acum Otimista': acum_otim,
    'Acum Pessimista': acum_pess
})
df_proj.set_index('Data', inplace=True)

# 12) Interface do usuário
st.subheader("🔍 Selecione a Visualização")
opcao = st.radio("Escolha a série", ["Soma Mensal", "Acumulado", "Variação (%)", "Barras por Ano"])

if opcao == "Soma Mensal":
    st.plotly_chart(fig1, use_container_width=True)
elif opcao == "Acumulado":
    st.plotly_chart(fig2, use_container_width=True)
elif opcao == "Variação (%)":
    st.plotly_chart(fig_var, use_container_width=True)
elif opcao == "Barras por Ano":
    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("📅 Tabela de Projeções Mensais")
st.dataframe(df_proj.style.format("{:,.0f}"))

# 13) Salvar os gráficos como imagens para Power BI
fig1.write_image("C:/PowerBI/graficos/soma_mensal.png", width=1200, height=600, engine="kaleido")
fig2.write_image("C:/PowerBI/graficos/acumulado.png", width=1200, height=600, engine="kaleido")
fig_var.write_image("C:/PowerBI/graficos/variacao.png", width=1200, height=600, engine="kaleido")
fig_bar.write_image("C:/PowerBI/graficos/barras_por_ano.png", width=1200, height=600, engine="kaleido")
