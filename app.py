import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import os

# Configuração
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

def preparar_series(df):
    agrupado = df.groupby(pd.Grouper(key='Fim Real Caldeiraria', freq='M'))
    serie_peso = agrupado['Peso Total (Ton)'].sum().last('24M')
    serie_qtd_ss = agrupado['SS SAMC'].count().last('24M')
    serie_produtividade = (serie_peso / serie_qtd_ss).replace([np.inf, -np.inf], np.nan).dropna()
    return serie_peso, serie_qtd_ss, serie_produtividade

def prever_serie(serie, fim_proj="2027-07-31"):
    serie = serie.copy()
    serie.index = pd.to_datetime(serie.index)
    serie = serie.astype(float).replace([np.inf, -np.inf], np.nan).dropna()

    modelo = ARIMA(serie, order=(1, 0, 3)).fit()
    n = (pd.to_datetime(fim_proj).to_period("M") - serie.index[-1].to_period("M")).n + 1
    datas_futuras = pd.date_range(start=serie.index[-1] + pd.offsets.MonthEnd(), periods=n, freq='M')

    forecast = modelo.get_forecast(steps=n)
    media, ic = forecast.predicted_mean, forecast.conf_int(alpha=0.20)
    pess, otim = ic.iloc[:, 0], ic.iloc[:, 1]
    return datas_futuras, media, otim, pess

def construir_acumulados(previsao, otim, pess, base):
    base_acum = base.cumsum().iloc[-1]
    return (np.cumsum(previsao) + base_acum,
            np.cumsum(otim) + base_acum,
            np.cumsum(pess) + base_acum)

def gerar_graficos(serie, datas_fut, real, otim, pess, acum_r, acum_o, acum_p,
                   serie_qtd_ss, serie_produtividade, df):
    
    fig1 = go.Figure([
        go.Scatter(x=serie.index, y=serie, mode='lines+markers', name='Histórico'),
        go.Scatter(x=datas_fut, y=real, mode='lines', name='Realista'),
        go.Scatter(x=datas_fut, y=otim, mode='lines', name='Otimista', line=dict(dash='dash')),
        go.Scatter(x=datas_fut, y=pess, mode='lines', name='Pessimista', line=dict(dash='dash')),
        go.Scatter(x=[serie.idxmax()], y=[serie.max()], mode='markers+text',
                   marker=dict(color='green', size=10), text=["⬆ Máximo"], textposition="top center"),
        go.Scatter(x=[serie.idxmin()], y=[serie.min()], mode='markers+text',
                   marker=dict(color='red', size=10), text=["⬇ Mínimo"], textposition="bottom center")
    ])
    fig1.update_layout(title="📈 Peso Total por Mês", xaxis_title="Data", yaxis_title="Toneladas")

    fig2 = go.Figure([
        go.Scatter(x=serie.index, y=serie.cumsum(), mode='lines+markers', name='Acumulado Real'),
        go.Scatter(x=datas_fut, y=acum_r, mode='lines', name='Realista'),
        go.Scatter(x=datas_fut, y=acum_o, mode='lines', name='Otimista', line=dict(dash='dash')),
        go.Scatter(x=datas_fut, y=acum_p, mode='lines', name='Pessimista', line=dict(dash='dash')),
    ])
    fig2.update_layout(title="📊 Peso Acumulado", xaxis_title="Data", yaxis_title="Toneladas")

    variacao = serie.pct_change() * 100
    fig3 = go.Figure([go.Scatter(x=variacao.index, y=variacao, mode='lines+markers', name='Variação %')])
    fig3.update_layout(title="📉 Variação % Mês a Mês", xaxis_title="Data", yaxis_title="%")

    df['Ano'] = df['Fim Real Caldeiraria'].dt.year
    df['Mês'] = df['Fim Real Caldeiraria'].dt.month_name().str[:3]
    pivot = df.pivot_table(index='Mês', columns='Ano', values='Peso Total (Ton)', aggfunc='sum')
    pivot = pivot.fillna(0).reindex(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    fig4 = go.Figure([go.Bar(name=str(ano), x=pivot.index, y=pivot[ano]) for ano in pivot.columns])
    fig4.update_layout(barmode='stack', title="📊 Barras por Ano", xaxis_title="Mês", yaxis_title="Ton")

    fig5 = go.Figure([
        go.Scatter(x=serie_qtd_ss.index, y=serie_qtd_ss, mode='lines+markers', name='Quantidade de SS')
    ])
    fig5.update_layout(title="📦 Quantidade de SS por Mês", xaxis_title="Data", yaxis_title="Qtde SS")

    fig6 = go.Figure([
        go.Scatter(x=serie_produtividade.index, y=serie_produtividade, mode='lines+markers', name='Ton/SS')
    ])
    fig6.update_layout(title="⚙️ Produtividade (Ton/SS)", xaxis_title="Data", yaxis_title="Toneladas por SS")

    return fig1, fig2, fig3, fig4, fig5, fig6

def exportar_imagens(figs, nomes):
    for fig, nome in zip(figs, nomes):
        fig.write_image(f"{CAMINHO_SAIDA}/{nome}.png", width=1200, height=600, engine="kaleido")

def interface():
    st.title("📊 Análise de Produtividade e Peso até Julho/2027")
    df = carregar_dados()
    serie_peso, serie_qtd_ss, serie_produtividade = preparar_series(df)

    if serie_peso.empty:
        st.warning("⚠️ A série de dados está vazia após o filtro.")
        return

    datas, real, otim, pess = prever_serie(serie_peso)
    acum_r, acum_o, acum_p = construir_acumulados(real, otim, pess, serie_peso)
    
    figs = gerar_graficos(serie_peso, datas, real, otim, pess,
                          acum_r, acum_o, acum_p, serie_qtd_ss, serie_produtividade, df)

    st.subheader("🔍 Selecione a Visualização")
    op = st.radio("Escolha a série", [
        "Peso por Mês", "Peso Acumulado", "Variação %", 
        "Barras por Ano", "Quantidade de SS", "Produtividade Ton/SS"
    ])
    fig_dict = {
        "Peso por Mês": figs[0],
        "Peso Acumulado": figs[1],
        "Variação %": figs[2],
        "Barras por Ano": figs[3],
        "Quantidade de SS": figs[4],
        "Produtividade Ton/SS": figs[5]
    }
    st.plotly_chart(fig_dict[op], use_container_width=True)

    st.subheader("📅 Tabela de Projeções de Peso")
    df_proj = pd.DataFrame({
        'Data': datas, 'Realista': real, 'Otimista': otim, 'Pessimista': pess,
        'Acum Realista': acum_r, 'Acum Otimista': acum_o, 'Acum Pessimista': acum_p
    }).set_index('Data')
    st.dataframe(df_proj.style.format("{:,.0f}"))

    exportar_imagens(figs, [
        "peso_mensal", "peso_acumulado", "variacao", "barras_ano", "quantidade_ss", "produtividade"
    ])
    
if __name__ == "__main__":
    interface()
