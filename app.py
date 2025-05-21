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
    df['Fim Real Caldeiraria'] = pd.to_datetime(df['Fim Real Caldeiraria'])
    grupo = df.groupby(pd.Grouper(key='Fim Real Caldeiraria', freq='M'))

    peso_mensal = grupo['Peso Total (Ton)'].sum().last('24M')
    ss_mensal = grupo['SS SAMC'].count().last('24M')

    produtividade = peso_mensal / ss_mensal
    produtividade = produtividade.replace([np.inf, -np.inf], np.nan)

    ss_por_ton = ss_mensal / peso_mensal
    ss_por_ton = ss_por_ton.replace([np.inf, -np.inf], np.nan)

    return peso_mensal, ss_mensal, produtividade, ss_por_ton

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

def gerar_graficos(peso, ss, prod, ss_ton, datas, real, otim, pess, acum_r, acum_o, acum_p, df):
    media_peso = peso.mean()
    media_acum = peso.cumsum().mean()
    media_var = (peso.pct_change() * 100).mean()
    media_prod = prod.mean()
    media_ss = ss.mean()
    media_sston = ss_ton.mean()

    fig1 = go.Figure([
        go.Scatter(x=peso.index, y=peso, mode='lines+markers', name='Peso (Ton)'),
        go.Scatter(x=datas, y=real, mode='lines', name='Realista'),
        go.Scatter(x=datas, y=otim, mode='lines', name='Otimista', line=dict(dash='dash')),
        go.Scatter(x=datas, y=pess, mode='lines', name='Pessimista', line=dict(dash='dash')),
        go.Scatter(x=peso.index, y=[media_peso]*len(peso), name='Média Histórica', line=dict(color='gray', dash='dot'))
    ])
    fig1.update_layout(title="📈 Soma Mensal de Peso", xaxis_title="Data", yaxis_title="Ton")

    fig2 = go.Figure([
        go.Scatter(x=peso.index, y=peso.cumsum(), mode='lines+markers', name='Acumulado Real'),
        go.Scatter(x=datas, y=acum_r, mode='lines', name='Acum Realista'),
        go.Scatter(x=datas, y=acum_o, mode='lines', name='Acum Otimista', line=dict(dash='dash')),
        go.Scatter(x=datas, y=acum_p, mode='lines', name='Acum Pessimista', line=dict(dash='dash')),
        go.Scatter(x=peso.index, y=[media_acum]*len(peso), name='Média Histórica', line=dict(color='gray', dash='dot'))
    ])
    fig2.update_layout(title="📊 Acumulado", xaxis_title="Data", yaxis_title="Ton Acumuladas")

    variacao = peso.pct_change() * 100
    fig3 = go.Figure([
        go.Scatter(x=variacao.index, y=variacao, mode='lines+markers', name='Variação %'),
        go.Scatter(x=variacao.index, y=[media_var]*len(variacao), name='Média Histórica', line=dict(color='gray', dash='dot'))
    ])
    fig3.update_layout(title="📉 Variação Percentual Mês a Mês", xaxis_title="Data", yaxis_title="%")

    df['Ano'] = df['Fim Real Caldeiraria'].dt.year
    df['Mês'] = df['Fim Real Caldeiraria'].dt.month_name().str[:3]
    pivot = df.pivot_table(index='Mês', columns='Ano', values='Peso Total (Ton)', aggfunc='sum')
    pivot = pivot.fillna(0).reindex(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    fig4 = go.Figure([go.Bar(name=str(ano), x=pivot.index, y=pivot[ano]) for ano in pivot.columns])
    fig4.update_layout(barmode='stack', title="📊 Barras por Ano", xaxis_title="Mês", yaxis_title="Ton")

    fig5 = go.Figure([
        go.Scatter(x=prod.index, y=prod, mode='lines+markers', name='Produtividade (Ton/SS)'),
        go.Scatter(x=prod.index, y=[media_prod]*len(prod), name='Média Histórica', line=dict(color='gray', dash='dot'))
    ])
    fig5.update_layout(title="⚙️ Produtividade (Ton/SS SAMC)", xaxis_title="Data", yaxis_title="Ton por SS")

    fig6 = go.Figure([
        go.Scatter(x=ss.index, y=ss, mode='lines+markers', name='Quantidade de SS SAMC'),
        go.Scatter(x=ss.index, y=[media_ss]*len(ss), name='Média Histórica', line=dict(color='gray', dash='dot'))
    ])
    fig6.update_layout(title="📋 Contagem de SS SAMC por mês", xaxis_title="Data", yaxis_title="SS SAMC")

    fig7 = go.Figure([
        go.Scatter(x=ss_ton.index, y=ss_ton, mode='lines+markers', name='SS/Ton'),
        go.Scatter(x=ss_ton.index, y=[media_sston]*len(ss_ton), name='Média Histórica', line=dict(color='gray', dash='dot'))
    ])
    fig7.update_layout(title="📐 Eficiência de Fabricação (SS/Ton)", xaxis_title="Data", yaxis_title="SS por Ton")

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7

def exportar_imagens(figs, nomes):
    for fig, nome in zip(figs, nomes):
        fig.write_image(f"{CAMINHO_SAIDA}/{nome}.png", width=1200, height=600, engine="kaleido")

def interface():
    st.title("📊 Projeção de Peso, SS SAMC, Produtividade e Eficiência (SS/Ton) até Julho/2027")
    df = carregar_dados()
    peso, ss, prod, ss_ton = preparar_series(df)

    if peso.empty:
        st.warning("⚠️ A série de dados está vazia após o filtro.")
        return

    datas, real, otim, pess = prever_serie(peso)
    acum_r, acum_o, acum_p = construir_acumulados(real, otim, pess, peso)

    figs = gerar_graficos(peso, ss, prod, ss_ton, datas, real, otim, pess, acum_r, acum_o, acum_p, df)

    st.subheader("🔍 Selecione a Visualização")
    op = st.radio("Escolha a série", [
        "Soma Mensal", "Acumulado", "Variação (%)", "Barras por Ano",
        "Produtividade (Ton/SS)", "Quantidade de SS", "Eficiência (SS/Ton)"
    ])
    st.plotly_chart({
        "Soma Mensal": figs[0],
        "Acumulado": figs[1],
        "Variação (%)": figs[2],
        "Barras por Ano": figs[3],
        "Produtividade (Ton/SS)": figs[4],
        "Quantidade de SS": figs[5],
        "Eficiência (SS/Ton)": figs[6]
    }[op], use_container_width=True)

    st.subheader("📅 Tabela de Projeções Mensais")
    df_proj = pd.DataFrame({
        'Data': datas, 'Realista': real, 'Otimista': otim, 'Pessimista': pess,
        'Acum Realista': acum_r, 'Acum Otimista': acum_o, 'Acum Pessimista': acum_p
    }).set_index('Data')
    st.dataframe(df_proj.style.format("{:,.0f}"))

    exportar_imagens(figs, [
        "soma_mensal", "acumulado", "variacao", "barras_por_ano",
        "produtividade", "ss_samc", "eficiencia_ss_ton"
    ])

if __name__ == "__main__":
    interface()
