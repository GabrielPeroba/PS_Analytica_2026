import pandas as pd

ARQUIVO_CHUVA = "historico_pluviometrico_2021_2025.csv"
ARQUIVO_OCORRENCIAS = "base_ocorrencias_1746.csv"
ARQUIVO_SAIDA = "dataset_treinamento_predicao.csv"

print("1. Carregando bases de dados históricas...")
df_chuva = pd.read_csv(ARQUIVO_CHUVA, low_memory=False)
df_1746 = pd.read_csv(ARQUIVO_OCORRENCIAS)

print("2. Normalização dos dados pluviométricos...")
col_chuva = ['05_min', '10_min', '15_min', '01_h', '04_h', '24_h', '96_h']
for col in col_chuva:
    df_chuva[col] = pd.to_numeric(df_chuva[col], errors='coerce')

df_chuva.fillna(0, inplace=True)

df_chuva = df_chuva[df_chuva['Hora'].str.endswith('00:00')].copy()
df_chuva['Data_Hora'] = pd.to_datetime(df_chuva['Dia'] + ' ' + df_chuva['Hora'], format='%d/%m/%Y %H:%M:%S')

print("3. Processamento da base de ocorrências (Foco restrito em Hidrologia Urbana)...")
df_1746['texto_busca'] = df_1746['tipo'].astype(str) + " " + df_1746['subtipo'].astype(str)

palavras_chave = 'alagamento|enchente|bolsão|bueiro|inundação|inundacao|bolsao'

df_1746 = df_1746[df_1746['texto_busca'].str.contains(palavras_chave, case=False, na=False)]

df_1746['Data_Hora'] = pd.to_datetime(df_1746['data_inicio']).dt.floor('h')
df_1746['Ocorreu_Desastre'] = 1 

df_1746_limpo = df_1746[['Data_Hora', 'id_bairro', 'Ocorreu_Desastre']].drop_duplicates()

print("4. Mapeamento geográfico: Estações Pluviométricas vs. Bairros...")
mapeamento_estacoes = {
    'Alto Da Boa Vista': [34.0], 'Anchieta': [107.0], 'Av Brasil Mendanha': [144.0],
    'Bangu': [141.0], 'Barrinha': [126.0], 'Campo Grande': [144.0],
    'Cidade De Deus': [118.0], 'Copacabana': [24.0], 'Grajau': [38.0],
    'Grajau Jacarepagua': [115.0], 'Grande Meier': [63.0], 'Grota Funda': [132.0],
    'Guaratiba': [151.0], 'Ilha Do Governador': [100.0], 'Iraja': [76.0],
    'Jardim Botanico': [28.0], 'Laranjeiras': [17.0], 'Madureira': [83.0],
    'Penha': [43.0], 'Piedade': [69.0], 'Recreio': [132.0], 'Riocentro': [5.0],
    'Rocinha': [154.0], 'Santa Cruz': [149.0], 'Santa Teresa': [14.0],
    'Sao Cristovao': [10.0], 'Saude': [1.0], 'Sepetiba': [150.0],
    'Tanque': [123.0], 'Tijuca': [33.0], 'Tijuca Muda': [33.0],
    'Urca': [22.0], 'Vidigal': [30.0]
}

print("5. Executando interseção de dados (Merge)...")
lista_final = []

for estacao, bairros in mapeamento_estacoes.items():
    df_c = df_chuva[df_chuva['Estacao'].str.lower() == estacao.lower()].copy()
    
    if not df_c.empty:
        df_d = df_1746_limpo[df_1746_limpo['id_bairro'].isin(bairros)].copy()
        df_d = df_d[['Data_Hora', 'Ocorreu_Desastre']].drop_duplicates()
        
        df_m = pd.merge(df_c, df_d, on='Data_Hora', how='left')
        df_m['Ocorreu_Desastre'] = df_m['Ocorreu_Desastre'].fillna(0)
        
        lista_final.append(df_m)
        print(f"  [CONCLUÍDO] Estação: {estacao}")
    else:
        print(f"  [ALERTA] Estação {estacao} sem registros correspondentes na base de chuva.")

df_final = pd.concat(lista_final, ignore_index=True)

df_final.to_csv(ARQUIVO_SAIDA, index=False)

print("\n--- RELATÓRIO DE PROCESSAMENTO FINAL ---")
print(f"Total de registros gerados: {len(df_final)}")
print(f"Eventos positivos para treinamento (100% Água): {int(df_final['Ocorreu_Desastre'].sum())}")
print(f"Dataset exportado com sucesso: {ARQUIVO_SAIDA}")    
