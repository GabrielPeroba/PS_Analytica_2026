# PS_Analytica_2026 - Gabriel Peroba
Repositório de base de dados e scripts python para o Hackathon do grupo Analytica - UFRJ.

# 🌊 Previsão de Alagamentos via IA e Topografia

## 📌 O Problema
No Rio de Janeiro, a topografia é o fator determinante para o caos urbano. Enquanto áreas altas escoam, as baixadas sofrem com bolsões d'água e alagamentos críticos que param a mobilidade da cidade.

## 🚀 Nossa Solução
Desenvolvemos uma Inteligência Artificial (**Random Forest Classifier**) que não apenas lê a chuva, mas entende a **física da gravidade**. Cruzamos dados pluviométricos históricos com a **cota altimétrica** das 33 estações do Alerta Rio para antecipar onde a água irá acumular.

### Diferenciais:
- **Engenharia de Features:** Janelas móveis de tempo (5min a 96h) para medir a saturação da drenagem.
- **Consciência Geográfica:** O modelo aprendeu que baixas altitudes + chuva forte = Alerta Crítico.
- **Simulador Interativo:** Interface em Streamlit para simulação tática da Defesa Civil e COR.

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python
- **IA:** Scikit-Learn (Random Forest)
- **Interface:** Streamlit
- **Mapas:** Folium (com integração Font Awesome)
- **Dados:** API Open-Meteo & Dados abertos da Prefeitura do Rio (1746).

## 📂 Como Rodar o Projeto
1. Instale as dependências: `pip install -r requirements.txt`
2. Gere a base turbinada: `python asas.py`
3. Inicie o simulador: `streamlit run app_simulador.py`
