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
    df = df.dropna(subset=['Fim Real Caldeiraria', 'Peso Total (Ton)', 'SS'])
    df = df[df['Fim Real Caldeiraria'] <= pd.to_datetime("today")]
    return df

def preparar_series(df):
    # Soma mensal para Peso Total (Ton)
    serie_peso = df.groupby(pd.Grouper(key='Fim Real Caldeiraria', freq='M'))['Peso Total (Ton)'].sum()
    # Soma mensal para SS
    serie_ss = df.groupby(pd.Grouper(key='Fim Real Caldeiraria', freq='M'))['SS'].sum()
    # Usar os últimos 24 meses
    return serie_peso.last('24M'), serie_ss.last('24M')

def prever_serie(serie, fim_proj="2027-07-31"):
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

def gerar_graficos(serie_peso, serie_ss, datas_fut, real_peso, otim_peso, pess_peso,
                   acum_r_peso, acum_o_peso, acum_p_peso,
                   real_ss, otim_ss, pess_ss,
                   acum_r_ss, acum_o_ss, acum_p_ss,
                   df):
    # Gráfico Peso Total (Ton) - Soma Mensal
    fig1 = go.Figure([
        go.Scatter(x=serie_peso.index, y=serie_peso, mode='lines+markers', name='Peso Histórico'),
        go.Scatter(x=datas_fut, y=real_peso, mode='lines', name='Peso Realista'),
        go.Scatter(x=datas_fut, y=otim_peso, mode='lines', name='Peso Otimista', line=dict(dash='dash')),
        go.Scatter(x=datas_fut, y=pess_peso, mode='lines', name='Peso Pessimista', line=dict(dash='dash')),
    ])
    fig1.update_layout(title="📈 Soma Mensal - Peso Total (Ton)", xaxis_title="Data", yaxis_title="Ton")

    # Gráfico SS - Soma Mensal
    fig2 = go.Figure([
        go.Scatter(x=serie_ss.index, y=serie_ss, mode='lines+markers', name='SS Histórico'),
        go.Scatter(x=datas_fut, y=real_ss, mode='lines', name='SS Realista'),
        go.Scatter(x=datas_fut, y=otim_ss, mode='lines', name='SS Otimista', line=dict(dash='dash')),
        go.Scatter(x=datas_fut, y=pess_ss, mode='lines', name='SS Pessimista', line=dict(dash='dash')),
    ])
    fig2.update_layout(title="📈 Soma Mensal - Quantidade de SS", xaxis_title="Data", yaxis_title="Quantidade de SS")

    # Gráfico acumulado Peso Total (Ton)
    fig3 = go.Figure([
        go.Scatter(x=serie_peso.index, y=serie_peso.cumsum(), mode='lines+markers', name='Peso Acumulado Real'),
        go.Scatter(x=datas_fut, y=acum_r_peso, mode='lines', name='Peso Acumulado Realista'),
        go.Scatter(x=datas_fut, y=acum_o_peso, mode='lines', name='Peso Acumulado Otimista', line=dict(dash='dash')),
        go.Scatter(x=datas_fut, y=acum_p_peso, mode='lines', name='Peso Acumulado Pessimista', line=dict(dash='dash')),
    ])
    fig3.update_layout(title="📊 Acumulado - Peso Total (Ton)", xaxis_title="Data", yaxis_title="Toneladas Acumuladas")

    # Gráfico acumulado SS
    fig4 = go.Figure([
        go.Scatter(x=serie_ss.index, y=serie_ss.cumsum(), mode='lines+markers', name='SS Acumulado Real'),
        go.Scatter(x=datas_fut, y=acum_r_ss, mode='lines', name='SS Acumulado Realista'),
        go.Scatter(x=datas_fut, y=acum_o_ss, mode='lines', name='SS Acumulado Otimista', line=dict(dash='dash')),
        go.Scatter(x=datas_fut, y=acum_p_ss, mode='lines', name='SS Acumulado Pessimista', line=dict(dash='dash')),
    ])
    fig4.update_layout(title="📊 Acumulado - Quantidade de SS", xaxis_title="Data", yaxis_title="SS Acumulados")

    # Gráfico barras por ano - Peso e SS
    df['Ano'] = df['Fim Real Caldeiraria'].dt.year
    df['Mês'] = df['Fim Real Caldeiraria'].dt.month_name().str[:3]

    pivot_peso = df.pivot_table(index='Mês', columns='Ano', values='Peso Total (Ton)', aggfunc='sum').fillna(0)
    pivot_peso = pivot_peso.reindex(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

    pivot_ss = df.pivot_table(index='Mês', columns='Ano', values='SS', aggfunc='sum').fillna(0)
    pivot_ss = pivot_ss.reindex(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

    fig5 = go.Figure()
    for ano in pivot_peso.columns:
        fig5.add_trace(go.Bar(name=f'Peso {ano}', x=pivot_peso.index, y=pivot_peso[ano], marker_color='blue'))
        fig5.add_trace(go.Bar(name=f'SS {ano}', x=pivot_ss.index, y=pivot_ss[ano], marker_color='orange'))

    fig5.update_layout(barmode='stack', title="📊 Barras por Ano - Peso Total (Azul) e SS (Laranja)", xaxis_title="Mês", yaxis_title="Total")

    return fig1, fig2, fig3, fig4, fig5

def exportar_imagens(figs, nomes):
    for fig, nome in zip(figs, nomes):
        fig.write_image(f"{CAMINHO_SAIDA}/{nome}.png", width=1200, height=600, engine="kaleido")

def interface():
    st.title("📊 Projeção de Peso Total (Ton) e SS até Julho/2027")
    df = carregar_dados()
    serie_peso, serie_ss = preparar_series(df)

    if serie_peso.empty or serie_ss.empty:
        st.warning("⚠️ A série de dados está vazia após o filtro.")
        return

    # Previsões para Peso
    datas, real_peso, otim_peso, pess_peso = prever_serie(serie_peso)
    acum_r_peso, acum_o_peso, acum_p_peso = construir_acumulados(real_peso, otim_peso, pess_peso, serie_peso)

    # Previsões para SS
    _, real_ss, otim_ss, pess_ss = prever_serie(serie_ss)
    acum_r_ss, acum_o_ss, acum_p_ss = construir_acumulados(real_ss, otim_ss, pess_ss, serie_ss)

    figs = gerar_graficos(serie_peso, serie_ss, datas,
                         real_peso, otim_peso, pess_peso,
                         acum_r_peso, acum_o_peso, acum_p_peso,
                         real_ss, otim_ss, pess_ss,
                         acum_r_ss, acum_o_ss, acum_p_ss,
                         df)

    st.subheader("🔍 Selecione a Visualização")
    op = st.radio("Escolha a série", [
        "Peso - Soma Mensal",
        "SS - Soma Mensal",
        "Peso - Acumulado",
        "SS - Acumulado",
        "Barras por Ano - Peso e SS"
    ])

    mapa_figs = {
        "Peso - Soma Mensal": figs[0],
        "SS - Soma Mensal": figs[1],
        "Peso - Acumulado": figs[2],
        "SS - Acumulado": figs[3],
        "Barras por Ano - Peso e SS": figs[4]
    }

    st.plotly_chart(mapa_figs[op], use_container_width=True)

    # Tabela com projeções para Peso e SS
    df_proj = pd.DataFrame({
        'Data': datas,
        'Peso Realista': real_peso,
        'Peso Otimista': otim_peso,
        'Peso Pessimista': pess_peso,
        'Peso Acum Realista': acum_r_peso,
        'Peso Acum Otimista': acum_o_peso,
        'Peso Acum Pessimista': acum_p_peso,
        'SS Realista': real_ss,
        'SS Otimista': otim_ss,
        'SS Pessimista': pess_ss,
        'SS Acum Realista': acum_r_ss,
        'SS Acum Otimista': acum
    })