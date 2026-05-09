import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="COR - Hidrologia Urbana", page_icon="🌐", layout="wide")

@st.cache_resource
def carregar_e_treinar_modelo():
    df = pd.read_csv("dataset_treinamento_predicao.csv", low_memory=False)
    features = ['05_min', '10_min', '15_min', '01_h', '04_h', '24_h', '96_h', 'Altitude_m']
    
    for col in features:
        if col not in df.columns:
            df[col] = 0.0 
            
    X = df[features]
    y = df['Ocorreu_Desastre'] 
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)
    
    return modelo, features

with st.spinner("Inicializando Motor de Inteligência Artificial e carregando tensores..."):
    modelo, features = carregar_e_treinar_modelo()

st.sidebar.title("🎛️ Módulo de Simulação")
st.sidebar.write("Defina as métricas pluviométricas e de infraestrutura para simular cenários de impacto operacional.")

chuva_1h = st.sidebar.slider("🌧️ Precipitação Atual (Última 1h) [mm]:", 
                             min_value=0.0, max_value=150.0, value=0.0, step=5.0)

saturacao_drenagem = st.sidebar.slider("🕳️ Saturação da Rede de Drenagem [%]:", 
                                   min_value=0, max_value=100, value=20, step=10)

st.sidebar.markdown("---")

st.title("🚨 Painel de Monitoramento: Risco de Alagamentos")
st.markdown("Identificação em tempo real de vulnerabilidades na malha viária devido ao estresse hidrológico.")

estacoes_completas = {
    'Alto Da Boa Vista': [-22.96583, -43.27833, 355.0],
    'Tijuca': [-22.93194, -43.22167, 340.0],
    'Santa Teresa': [-22.93167, -43.19639, 170.0],
    'Rocinha': [-22.98583, -43.24500, 160.0],
    'Grota Funda': [-23.01444, -43.52139, 116.0],
    'Penha': [-22.84444, -43.27528, 111.0],
    'Grajau Jacarepagua': [-22.92556, -43.31583, 105.0],
    'Copacabana': [-22.98639, -43.18944, 90.0],
    'Urca': [-22.95583, -43.16667, 90.0],
    'Vidigal': [-22.99250, -43.23306, 85.0],
    'Grajau': [-22.92222, -43.26750, 80.0],
    'Tanque': [-22.91250, -43.36472, 73.0],
    'Sepetiba': [-22.96889, -43.71167, 62.0],
    'Laranjeiras': [-22.94056, -43.18750, 60.0],
    'Anchieta': [-22.82694, -43.40333, 50.0],
    'Piedade': [-22.89182, -43.31005, 50.0],
    'Madureira': [-22.87333, -43.33889, 45.0],
    'Tijuca Muda': [-22.93278, -43.24333, 31.0],
    'Campo Grande': [-22.90361, -43.56194, 30.0],
    'Av Brasil Mendanha': [-22.85694, -43.54111, 30.0],
    'Grande Meier': [-22.89056, -43.27806, 25.0],
    'Sao Cristovao': [-22.89667, -43.22167, 25.0],
    'Iraja': [-22.82694, -43.33694, 20.0],
    'Bangu': [-22.88028, -43.46583, 15.0],
    'Santa Cruz': [-22.90944, -43.68444, 15.0],
    'Cidade De Deus': [-22.94556, -43.36278, 15.0],
    'Saude': [-22.89606, -43.18786, 15.0],
    'Recreio': [-23.01000, -43.44056, 10.0],
    'Barrinha': [-23.00849, -43.29965, 7.0],
    'Jardim Botanico': [-22.97278, -43.22389, 0.0],
    'Guaratiba': [-23.05028, -43.59472, 0.0],
    'Riocentro': [-22.97721, -43.39155, 0.0],
    'Ilha Do Governador': [-22.81806, -43.21028, 0.0]
}

mapa_rj = folium.Map(location=[-22.91, -43.30], zoom_start=11, tiles='CartoDB positron')

for nome_estacao, dados_geo in estacoes_completas.items():
    lat, lon, altitude = dados_geo
    
    historico_96h = (saturacao_drenagem / 100.0) * 200.0
    historico_24h = (saturacao_drenagem / 100.0) * 80.0
    historico_4h = (saturacao_drenagem / 100.0) * 25.0
    
    chuva_5min = chuva_1h * 0.15
    chuva_10min = chuva_1h * 0.30
    chuva_15min = chuva_1h * 0.50
    
    chuva_4h_calc = chuva_1h + historico_4h
    chuva_24h_calc = chuva_1h + historico_24h
    chuva_total_96h = chuva_1h + historico_96h 
    
    dados_para_ia = [chuva_5min, chuva_10min, chuva_15min, chuva_1h, chuva_4h_calc, chuva_24h_calc, chuva_total_96h, altitude]
    df_agora = pd.DataFrame([dados_para_ia], columns=features)
    
    if len(modelo.classes_) == 1:
        prob_ml = 0.0 
    else:
        prob_ml = modelo.predict_proba(df_agora)[0][1]
    
    risco_fisico = (chuva_1h / 120.0) * 0.70 + (saturacao_drenagem / 100.0) * 0.30
    
    prob_alagamento = max(prob_ml, risco_fisico)
    
    if altitude >= 80.0:
        prob_alagamento *= 0.30
    elif altitude <= 25.0:
        prob_alagamento *= 1.40
            
    prob_alagamento = min(max(prob_alagamento, 0.0), 0.99)
    
    if prob_alagamento >= 0.50:
        cor, status, icone = 'red', 'ALERTA CRÍTICO', 'cloud-showers-heavy'
    elif prob_alagamento >= 0.25:
        cor, status, icone = 'orange', 'BOLSÃO D\'ÁGUA', 'droplet'
    else:
        cor, status, icone = 'green', 'FLUXO NORMAL', 'check'
        
    html_popup = f"""
    <div style="width: 220px; font-family: Arial, sans-serif;">
        <h4 style="margin-bottom: 5px; color: #333;"><b>📍 {nome_estacao.upper()}</b></h4>
        <p style="margin-top: 0px; font-size: 12px; color: gray;">Cota Altimétrica: {altitude}m</p>
        <hr style="margin: 5px 0;">
        <b>Status:</b> <span style="color: {cor}; font-size: 14px;"><b>{status}</b></span><br>
        <b>Probabilidade Final:</b> {prob_alagamento:.1%}<br>
    </div>
    """
    
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(html_popup, max_width=300),
        icon=folium.Icon(color=cor, icon=icone, prefix='fa')
    ).add_to(mapa_rj)

st_folium(mapa_rj, width=1000, height=500)
