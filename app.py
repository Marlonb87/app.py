import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import os

# Configuração geral
st.set_page_config(layout="wide")
CAMINHO_SAIDA = "C:/PowerBI/graficos"
os.makedirs(CAMINHO_SAIDA, exist_ok=True)

@st.cache_data
def carregar_dados():
    url = "https://raw.githubusercontent.com/Marlonb87/app.py/main/4600672730_Prog_Process_16.05.2025.xlsx"
    df = pd.read_excel(url, engine="openpyxl")
    df['Fim Real Caldeiraria'] = pd.to_datetime(df['Fim Real Caldeiraria'], errors='coerce')
    df = df.dropna(subset=['Fim Real Caldeiraria', 'Peso Total (Ton)', 'SS SAMC'])
    df = df[df['Fim Real Caldeiraria'] <= pd.to_datetime("today")]
    return df

def preparar_serie(df):
    agrupado = df.groupby(pd.Grouper(key='Fim Real Caldeiraria', freq='M')).agg(
        peso=('Peso Total (Ton)', 'sum'),
        ss=('SS SAMC', 'count')
    )
    agrupado['produtividade'] = agrupado['peso'] / agrupado['ss']
    return agrupado.last('24M')

def prever_serie(serie, fim_proj="2027-07-31"):
    serie = serie.copy()
    serie.index = pd.to_datetime(serie.index)
    serie = serie.astype(float)
    serie = serie.replace([np.inf, -np.inf], np.nan).dropna()
    modelo = ARIMA(serie['peso'], order=(1, 0, 3)).fit()
    n = (pd.to_datetime(fim_proj).to_period("M") - serie.index[-1].to_period("M")).n + 1
    datas_futuras = pd.date_range(start=serie.index[-1] + pd.offsets.MonthEnd(), periods=n, freq='M')
    forecast = modelo.get_forecast(steps=n)
    media, ic = forecast.predicted_mean, forecast.conf_int(alpha=0.20)
    pess, otim = ic.iloc[:, 0], ic.iloc[:, 1]
    return datas_futuras, media, otim, pess

def construir_acumulados(previsao, otim, pess, base):
    base_acum = base['peso'].cumsum().iloc[-1]
    return (np.cumsum(previsao) + base_acum,
            np.cumsum(otim) + base_acum,
            np.cumsum(pess) + base_acum)

def gerar_graficos(serie, datas_fut, real, otim, pess, acum_r, acum_o, acum_p, df):
    media_hist = serie['peso'].mean()

    fig1 = go.Figure([
        go.Scatter(x=serie.index, y=serie['peso'], mode='lines+markers', name='Histórico'),
        go.Scatter(x=datas_fut, y=real, mode='lines', name='Realista'),
        go.Scatter(x=datas_fut, y=otim, mode='lines', name='Otimista', line=dict(dash='dash')),
        go.Scatter(x=datas_fut, y=pess, mode='lines', name='Pessimista', line=dict(dash='dash')),
        go.Scatter(x=[serie.index.min(), datas_fut[-1]], y=[media_hist]*2,
                   mode='lines', name='Média Histórica', line=dict(color='black', dash='dot'))
    ])
    fig1.update_layout(title="📈 Soma Mensal com Média", xaxis_title="Data", yaxis_title="Ton")

    fig2 = go.Figure([
        go.Scatter(x=serie.index, y=serie['peso'].cumsum(), mode='lines+markers', name='Acumulado Real'),
        go.Scatter(x=datas_fut, y=acum_r, mode='lines', name='Realista'),
        go.Scatter(x=datas_fut, y=acum_o, mode='lines', name='Otimista', line=dict(dash='dash')),
        go.Scatter(x=datas_fut, y=acum_p, mode='lines', name='Pessimista', line=dict(dash='dash')),
    ])
    fig2.update_layout(title="📊 Acumulado", xaxis_title="Data", yaxis_title="Toneladas Acumuladas")

    variacao = serie['peso'].pct_change() * 100
    fig3 = go.Figure([go.Scatter(x=variacao.index, y=variacao, mode='lines+markers', name='Variação %')])
    fig3.update_layout(title="📉 Variação Percentual Mês a Mês", xaxis_title="Data", yaxis_title="%")

    df['Ano'] = df['Fim Real Caldeiraria'].dt.year
    df['Mês'] = df['Fim Real Caldeiraria'].dt.month_name().str[:3]
    pivot = df.pivot_table(index='Mês', columns='Ano', values='Peso Total (Ton)', aggfunc='sum')
    pivot = pivot.fillna(0).reindex(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    fig4 = go.Figure([go.Bar(name=str(ano), x=pivot.index, y=pivot[ano]) for ano in pivot.columns])
    fig4.update_layout(barmode='stack', title="📊 Barras por Ano", xaxis_title="Mês", yaxis_title="Ton")

    return fig1, fig2, fig3, fig4

def exportar_imagens(figs, nomes):
    for fig, nome in zip(figs, nomes):
        fig.write_image(f"{CAMINHO_SAIDA}/{nome}.png", width=1200, height=600, engine="kaleido")

def interface():
    st.title("📊 Projeção de Peso Total (Ton) até Julho/2027")
    df = carregar_dados()
    serie = preparar_serie(df)

    if serie.empty:
        st.warning("⚠️ A série de dados está vazia após o filtro.")
        return

    datas, real, otim, pess = prever_serie(serie)
    acum_r, acum_o, acum_p = construir_acumulados(real, otim, pess, serie)
    figs = gerar_graficos(serie, datas, real, otim, pess, acum_r, acum_o, acum_p, df)

    st.subheader("🔍 Selecione a Visualização")
    op = st.radio("Escolha a série", ["Soma Mensal", "Acumulado", "Variação (%)", "Barras por Ano"])
    st.plotly_chart({"Soma Mensal": figs[0], "Acumulado": figs[1], "Variação (%)": figs[2], "Barras por Ano": figs[3]}[op], use_container_width=True)

    st.subheader("📅 Tabela de Projeções Mensais")
    df_proj = pd.DataFrame({
        'Data': datas, 'Realista': real, 'Otimista': otim, 'Pessimista': pess,
        'Acum Realista': acum_r, 'Acum Otimista': acum_o, 'Acum Pessimista': acum_p
    }).set_index('Data')
    st.dataframe(df_proj.style.format("{:,.0f}"))

    exportar_imagens(figs, ["soma_mensal", "acumulado", "variacao", "barras_por_ano"])

# Executa a aplicação
if __name__ == "__main__":
    interface()