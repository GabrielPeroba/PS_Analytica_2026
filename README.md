# PS_Analytica_2026 - Gabriel Peroba
Repositório de base de dados e scripts python para o Hackathon do grupo Analytica - UFRJ.

# Previsão de Alagamentos via IA e Topografia

## O Problema
No Rio de Janeiro, a topografia é o fator determinante para o caos urbano. Enquanto áreas altas escoam, as baixadas sofrem com bolsões d'água e alagamentos críticos que param a mobilidade da cidade.

## Nossa Solução
Desenvolvemos um modelo de aprendizado de máquinas (**Random Forest Classifier**) que não apenas lê a chuva, mas entende a **física da gravidade**. Cruzamos dados pluviométricos históricos com a **cota altimétrica** das 33 estações do Alerta Rio para antecipar onde a água irá acumular.

### Diferenciais:
- **Engenharia de Features:** Janelas móveis de tempo (5min a 96h) para medir a saturação da drenagem e a memória hídrica do solo.
- **Consciência Geográfica:** O modelo aprendeu que baixas altitudes + chuva forte = Alerta Crítico.
- **Simulador Interativo:** Interface em Streamlit para simulação tática da Defesa Civil e Centro de Operações (COR).

## Tecnologias Utilizadas
- **Linguagem:** Python
- **IA e Machine Learning:** Scikit-Learn (Random Forest Classifier)
- **Interface e Dashboard:** Streamlit
- **Mapas Georreferenciados:** Folium (com integração Font Awesome)
- **Dados e Telemetria:** API Open-Meteo & Dados abertos da Prefeitura do Rio (Data.Rio / 1746).

---

##  Bases de Dados 
Devido ao grande volume de registros históricos (que ultrapassam o limite de 100MB do GitHub), os datasets originais e a matriz de treinamento final foram hospedados na nuvem. 

Faça o download dos arquivos `.csv` através dos links públicos abaixo e coloque-os na mesma pasta dos scripts antes de rodar o projeto:

- 📊 **[Base Histórica Pluviométrica 2021-2025 (Geo-Rio/Alerta Rio)](https://drive.google.com/file/d/1fpxHzwTXxCTDOxl7cCj-J5GPvlBx9lJ4/view?usp=drive_link)**
- 🚨 **[Base de Ocorrências Operacionais 1746 (Foco Hidrologia)](https://drive.google.com/file/d/1_yzwXqVP93jpvpDYdZZghrJFz4uaur_N/view?usp=drive_link)**
- 🤖 **[Dataset de Treinamento e Predição (Merge final para a IA)](https://drive.google.com/file/d/19riYhZ_4GhgmEH5BmsLCsTU-2ZKpPx_m/view?usp=drive_link)**

---

## Como Rodar o Projeto

Siga o pipeline de execução abaixo na sua máquina local:

**1. Instale as dependências:**

pip install -r requirements.txt

**2. Baixe as bases de dados e arquivos e deixe tudo na mesma pasta:**


**3. Gere a base de dados purificada (ETL):**

python processamento_dados.py


**4. Treine a o modelo de aprendizado e gere o mapa ao vivo:**

python treinar_ia.py

**5. Inicie o Simulador Interativo do COR via CMD ou PowerShell:**

streamlit run app_simulador.py
