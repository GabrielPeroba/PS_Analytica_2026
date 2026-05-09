import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import folium
import requests
import time

# Omissão de alertas não-críticos para limpeza do console de execução
warnings.filterwarnings('ignore') 

# ========================================================================
#                    FASE 1: TREINAMENTO DO MODELO PREDITIVO
# ========================================================================

print("1. Ingestão de Dados: Carregamento do dataset de treinamento...")
try:
    df = pd.read_csv("dataset_treinamento_predicao.csv", low_memory=False)
except FileNotFoundError:
    print("ERRO: Arquivo 'dataset_treinamento_predicao.csv' não encontrado. Execute o processamento de dados primeiro.")
    exit()

print("1.1. Enriquecimento de Dados: Mapeando Altitudes das Estações...")
dicionario_altitudes = {
    'Alto Da Boa Vista': 355.0, 'Tijuca': 340.0, 'Santa Teresa': 170.0,
    'Rocinha': 160.0, 'Grota Funda': 116.0, 'Penha': 111.0,
    'Grajau Jacarepagua': 105.0, 'Copacabana': 90.0, 'Urca': 90.0,
    'Vidigal': 85.0, 'Grajau': 80.0, 'Tanque': 73.0, 'Sepetiba': 62.0,
    'Laranjeiras': 60.0, 'Anchieta': 50.0, 'Piedade': 50.0,
    'Madureira': 45.0, 'Tijuca Muda': 31.0, 'Campo Grande': 30.0,
    'Av Brasil Mendanha': 30.0, 'Grande Meier': 25.0, 'Sao Cristovao': 25.0,
    'Iraja': 20.0, 'Bangu': 15.0, 'Santa Cruz': 15.0, 'Cidade De Deus': 15.0,
    'Saude': 15.0, 'Recreio': 10.0, 'Barrinha': 7.0, 'Jardim Botanico': 0.0,
    'Guaratiba': 0.0, 'Riocentro': 0.0, 'Ilha Do Governador': 0.0
}

# Padroniza os nomes das estações no dataset para bater perfeitamente com o dicionário
df['Estacao'] = df['Estacao'].astype(str).str.title()
# Cria a coluna Altitude_m cruzando o dicionário com o nome da Estação
df['Altitude_m'] = df['Estacao'].map(dicionario_altitudes).fillna(0.0)

features = ['05_min', '10_min', '15_min', '01_h', '04_h', '24_h', '96_h', 'Altitude_m']
X = df[features]
y = df['Ocorreu_Desastre'] 

print("2. Particionamento do Dataset (Holdout: 80% Treino / 20% Teste)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Instanciação e Treinamento do Modelo (Random Forest Classifier)...")
# class_weight='balanced' ajusta o peso para classes minoritárias (eventos críticos)
modelo = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)

print("4. Validação de Desempenho e Métricas do Modelo...")
y_pred = modelo.predict(X_test)

print("\n--- RELATÓRIO DE AVALIAÇÃO DO MODELO DE HIDROLOGIA URBANA ---")
print(f"Acurácia Global: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))

print("5. Extração de Importância de Atributos (Feature Importance)...")
importancia = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importancia.values, y=importancia.index, palette='mako')
plt.title('Matriz de Impacto: Fatores Preditivos de Alagamento Urbano', fontsize=14, fontweight='bold')
plt.xlabel('Relevância Estatística no Modelo (Peso)', fontsize=12)
plt.ylabel('Atributos (Pluviometria e Topografia)', fontsize=12)
plt.tight_layout()
plt.savefig("importancia_fatores_preditivos.png", dpi=300)
plt.close('all') 
print(" -> Gráfico salvo como 'importancia_fatores_preditivos.png'")

# ========================================================================
#              FASE 2: MÓDULO DE SIMULAÇÃO DE CENÁRIOS OPERACIONAIS
# ========================================================================

print("\n" + "="*60)
print("     MÓDULO DE ALERTA: CENTRO DE OPERAÇÕES (SIMULAÇÃO)     ")
print("="*60)

try:
    df_bairros = pd.read_csv("dicionario_bairros_rj.csv")
    tradutor_bairros = dict(zip(df_bairros['id_bairro'], df_bairros['nome']))
except:
    tradutor_bairros = {}

# Mapeamento de Zonas de Risco para Acionamento de Equipes
mapa_operacional = {
    'Copacabana': [24.0, 17.0, 18.0],
    'Bangu': [141.0, 135.0, 136.0],   
    'Tijuca': [33.0, 32.0, 34.0]      
}

def disparar_alerta(nome_estacao):
    ids_afetados = mapa_operacional.get(nome_estacao, [])
    nomes_afetados = [tradutor_bairros.get(b, f"ID {b}") for b in ids_afetados]
    print("   -> 🚨 PROTOCOLO DE CONTINGÊNCIA: Solicitar desvio de tráfego para:")
    for bairro in nomes_afetados:
        print(f"      📍 {bairro}")

def avaliar_risco_hidrologico(nome_estacao, dados_sensores):
    print(f"\n📡 Análise de Telemetria: {nome_estacao.upper()} (Cota Altimétrica: {dados_sensores[-1]}m)")
    
    df_agora = pd.DataFrame([dados_sensores], columns=features)
    prob_alagamento = modelo.predict_proba(df_agora)[0][1]
    
    # Validação Física: Mitigação de falsos positivos na ausência de precipitação imediata
    chuva_atual = dados_sensores[3]
    if chuva_atual == 0.0:
        prob_alagamento = prob_alagamento * 0.15
    
    print(f"📊 Probabilidade de Evento Crítico: {prob_alagamento:.1%}")
    
    if prob_alagamento >= 0.50:
        print("🔴 ALERTA VERMELHO: Risco de alagamento intrafegável e bloqueio de vias.")
        disparar_alerta(nome_estacao)
    elif prob_alagamento >= 0.25:
        print("🟡 ALERTA AMARELO: Formação de bolsões d'água. Impacto moderado na mobilidade.")
        disparar_alerta(nome_estacao)
    else:
        print("🟢 NORMALIDADE: Capacidade de escoamento da via dentro das margens de segurança.")

# Simulações de Estresse do Modelo
cenario_copacabana = [0.0, 0.0, 0.2, 1.5, 3.0, 10.0, 25.0, 90.0] 
avaliar_risco_hidrologico('Copacabana', cenario_copacabana)

cenario_bangu = [15.0, 30.0, 50.0, 90.0, 100.0, 110.0, 120.0, 15.0]
avaliar_risco_hidrologico('Bangu', cenario_bangu)

cenario_tijuca = [2.0, 5.0, 10.0, 15.0, 25.0, 80.0, 200.0, 34.0]
avaliar_risco_hidrologico('Tijuca', cenario_tijuca)

print("\n--- Fim da Simulação de Cenários ---")

# ========================================================================
#           FASE 3: MÓDULO DE TELEMETRIA E MAPA EM TEMPO REAL
# ========================================================================

print("\n" + "="*60)
print("  CONEXÃO API (OPEN-METEO): MONITORAMENTO EM TEMPO REAL  ")
print("="*60)

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

# Inicialização da interface cartográfica
mapa_rj = folium.Map(location=[-22.91, -43.30], zoom_start=11, tiles='CartoDB positron')

def requisitar_dados_meteorologicos(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=precipitation"
    try:
        resposta = requests.get(url, timeout=5)
        dados = resposta.json()
        return dados['current']['precipitation'] 
    except:
        return 0.0 

print("Iniciando varredura geoespacial das 33 estações pluviométricas...\n")

for nome_estacao, dados_geo in estacoes_completas.items():
    lat, lon, altitude = dados_geo
    
    # Requisição HTTP para ingestão de dados em tempo real
    chuva_1h_real = requisitar_dados_meteorologicos(lat, lon)
    
    # Aproximação estatística da distribuição pluviométrica para preenchimento de tensores
    chuva_5min = chuva_1h_real * 0.1
    chuva_10min = chuva_1h_real * 0.2
    chuva_15min = chuva_1h_real * 0.3
    chuva_4h = chuva_1h_real * 2.0
    chuva_24h = chuva_1h_real * 5.0
    chuva_96h = chuva_1h_real * 10.0
    
    dados_para_ia = [chuva_5min, chuva_10min, chuva_15min, chuva_1h_real, chuva_4h, chuva_24h, chuva_96h, altitude]
    
    # Inferência do Modelo ML com payload da API
    df_agora = pd.DataFrame([dados_para_ia], columns=features)
    prob_alagamento = modelo.predict_proba(df_agora)[0][1]
    
    # Ajuste fino de limiar preditivo
    if chuva_1h_real == 0.0:
        prob_alagamento = prob_alagamento * 0.15
    
    # Classificação visual para interface do usuário (UI)
    if prob_alagamento >= 0.50:
        cor = 'blue'
        status = 'ALAGAMENTO CRÍTICO'
        icone = 'tint'
    elif prob_alagamento >= 0.25:
        cor = 'cadetblue'
        status = "BOLSÃO D'ÁGUA"
        icone = 'info-sign'
    else:
        cor = 'green'
        status = 'FLUXO NORMAL'
        icone = 'ok'
        
    print(f"📡 {nome_estacao.ljust(20)} | Precipitação Atual: {chuva_1h_real:04.1f}mm | Confiança IA: {prob_alagamento:.1%} -> {status}")
        
    # Renderização de pop-up HTML integrado ao Folium
    html_popup = f"""
    <div style="width: 220px; font-family: Arial, sans-serif;">
        <h4 style="margin-bottom: 5px; color: #333;"><b>📍 {nome_estacao.upper()}</b></h4>
        <p style="margin-top: 0px; font-size: 12px; color: gray;">Cota Altimétrica: {altitude}m</p>
        <hr style="margin: 5px 0;">
        <b>Status Operacional:</b> <span style="color: {cor}; font-size: 14px;"><b>{status}</b></span><br>
        <b>Risco de Alagamento:</b> {prob_alagamento:.1%}<br>
        <hr style="margin: 5px 0;">
        <i style="color: #4da6ff;">🌦️ Telemetria API Open-Meteo<br>Precipitação: {chuva_1h_real}mm</i>
    </div>
    """
    
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(html_popup, max_width=300),
        icon=folium.Icon(color=cor, icon=icone)
    ).add_to(mapa_rj)
    
    time.sleep(0.1) # Controle de requisições (Rate Limiting)

arquivo_mapa = "dashboard_hidrologico_tempo_real.html"
mapa_rj.save(arquivo_mapa)

print("\n" + "="*60)
print(f"✅ RENDERIZAÇÃO CONCLUÍDA: Interface cartográfica salva em '{arquivo_mapa}'")
print("============================================================")
