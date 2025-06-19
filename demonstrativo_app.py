import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------#
# Configuração da Página
# --------------------------------#
st.set_page_config(
    page_title="Análise Preditiva de Consumo de Aço",
    page_icon="🏗️",
    layout="wide"
)

# --------------------------------#
# Carregamento e Cache de Dados e Modelo (para performance)
# --------------------------------#

@st.cache_data
def carregar_dados():
    """Carrega o dataframe original a partir do CSV."""
    df = pd.read_csv('Dataset_Aco 1.csv')
    return df

@st.cache_resource
def processar_e_treinar_modelo(df):
    """
    Realiza o pré-processamento, treina o modelo XGBoost,
    e retorna o modelo treinado, colunas, métricas de performance e resultados para validação.
    """
    df_processed = df.drop('ID_Obra', axis=1)
    
    # Engenharia de Atributos
    df_processed['Area_Por_Pavimento'] = df_processed['Area_Construida_m2'] / (df_processed['N_Pavimentos'] + 1e-6)
    df_processed['Interacao_Area_Pav'] = df_processed['Area_Construida_m2'] * df_processed['N_Pavimentos']
    df_processed['N_Pavimentos_Sq'] = df_processed['N_Pavimentos']**2
    
    df_processed = pd.get_dummies(df_processed, columns=['Cidade', 'Tipologia', 'Metodo_Construtivo'], drop_first=True)

    X = df_processed.drop('Consumo_Aco_kg', axis=1)
    y = df_processed['Consumo_Aco_kg']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = XGBRegressor(random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    df_resultados = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})
    
    return modelo, X.columns, mae, r2, df_resultados

# Execução das funções de carregamento e treinamento
df_original = carregar_dados()
modelo, colunas_treino, mae, r2, df_resultados = processar_e_treinar_modelo(df_original.copy())

# --------------------------------#
# Início do Layout do Dashboard
# --------------------------------#

st.title("🏗️ Análise Preditiva de Consumo de Aço")
st.markdown("Um demonstrativo sobre a aplicação de Machine Learning para otimizar orçamentos na Construção Civil.")
st.markdown("---")

# --- Seção 1: Análise Exploratória ---
st.header("1. Análise Exploratória dos Dados")
st.markdown("O primeiro passo foi entender os dados históricos de 1.000 obras para descobrir insights e variáveis de maior impacto.")

with st.expander("Clique para visualizar a base de dados utilizada"):
    st.dataframe(df_original, height=300)

col1, col2 = st.columns(2)
with col1:
    fig1 = px.scatter(df_original, x='Area_Construida_m2', y='Consumo_Aco_kg', 
                      title="<b>Área Construída vs. Consumo de Aço</b>", trendline="ols",
                      labels={'Area_Construida_m2': 'Área Construída (m²)'},
                      color_discrete_sequence=['#004B8D'])
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = px.box(df_original, x='Tipologia', y='Consumo_Aco_kg',
                  title="<b>Consumo de Aço por Tipologia de Obra</b>",
                  color_discrete_sequence=['#004B8D'])
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# --- Seção 2: Modelagem e Resultados ---
st.header("2. Construção e Performance do Modelo Preditivo")
st.markdown("Após testes com diferentes algoritmos, o modelo **XGBoost** foi selecionado por sua alta precisão. Seus resultados no conjunto de dados de teste foram:")

res1, res2 = st.columns(2)
with res1:
    st.metric(label="**Erro Médio Absoluto (MAE)**", value=f"{mae:,.0f} kg",
              help="Em média, as previsões do modelo têm um desvio de 1,353 kg em relação ao valor real.")
with res2:
    st.metric(label="**Coeficiente de Determinação (R²)**", value=f"{r2:.2%}",
              help="O modelo explica 99.86% da variação no consumo de aço.")

st.markdown("---")

col3, col4 = st.columns([0.6, 0.4])
with col3:
    fig_pred = px.scatter(df_resultados, x='Real', y='Previsto', 
                           title="<b>Validação do Modelo: Valor Real vs. Previsão</b>",
                           labels={'Real': 'Consumo Real (kg)', 'Previsto': 'Consumo Previsto (kg)'},
                           trendline="ols", trendline_color_override="red")
    st.plotly_chart(fig_pred, use_container_width=True)
with col4:
    importances = modelo.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': colunas_treino, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    fig_imp = px.bar(feature_importance_df.head(7), x='Importance', y='Feature', orientation='h',
                     title='<b>Principais Variáveis de Impacto</b>', color_discrete_sequence=['#004B8D'])
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")

# --- Seção 3: Simulador Interativo ---
st.header("3. Simulador Interativo de Consumo de Aço")
st.markdown("Ajuste os parâmetros abaixo para obter uma estimativa para um novo projeto.")

# Controles do Simulador no corpo principal da página
with st.container(border=True):
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        st.markdown("##### Parâmetros da Obra")
        area_sim = st.number_input("Área Construída (m²)", 500, 25000, 15000, 100)
        pavimentos_sim = st.number_input("Número de Pavimentos", 1, 50, 20, 1)

    with sim_col2:
        st.markdown("##### Características")
        tipologia_sim = st.selectbox("Tipologia da Obra", sorted(df_original['Tipologia'].unique()), key='sim_tipo')
        metodo_sim = st.selectbox("Método Construtivo", sorted(df_original['Metodo_Construtivo'].unique()), key='sim_metodo')

    with sim_col3:
        st.markdown("##### Localização e Custo")
        cidade_sim = st.selectbox("Cidade", sorted(df_original['Cidade'].unique()), key='sim_cidade')
        preco_kg_aco_sim = st.number_input("Custo do Aço (R$/kg)", 3.0, 15.0, 7.5, 0.1)

# Lógica da previsão é executada a cada interação com os widgets acima
dados_simulacao = {
    'area': area_sim, 'pavimentos': pavimentos_sim, 'tipologia': tipologia_sim, 
    'metodo': metodo_sim, 'cidade': cidade_sim
}

# Prepara o dataframe para a previsão, garantindo a mesma estrutura do treino
df_para_prever = pd.DataFrame(columns=colunas_treino, index=[0])
df_para_prever.fillna(0, inplace=True)
df_para_prever['Area_Construida_m2'] = dados_simulacao['area']
df_para_prever['N_Pavimentos'] = dados_simulacao['pavimentos']
df_para_prever['Area_Por_Pavimento'] = dados_simulacao['area'] / (dados_simulacao['pavimentos'] + 1e-6)
df_para_prever['Interacao_Area_Pav'] = dados_simulacao['area'] * dados_simulacao['pavimentos']
df_para_prever['N_Pavimentos_Sq'] = dados_simulacao['pavimentos']**2
if f"Cidade_{dados_simulacao['cidade']}" in colunas_treino:
    df_para_prever[f"Cidade_{dados_simulacao['cidade']}"] = 1
if f"Tipologia_{dados_simulacao['tipologia']}" in colunas_treino:
    df_para_prever[f"Tipologia_{dados_simulacao['tipologia']}"] = 1
if f"Metodo_Construtivo_{dados_simulacao['metodo']}" in colunas_treino:
    df_para_prever[f"Metodo_Construtivo_{dados_simulacao['metodo']}"] = 1

previsao_sim = modelo.predict(df_para_prever[colunas_treino])[0]
custo_estimado_sim = previsao_sim * preco_kg_aco_sim

st.markdown("---")
st.markdown("#### **Estimativa para o Projeto Simulado:**")

out_col1, out_col2 = st.columns(2)
with out_col1:
    st.metric(label="**Consumo de Aço Previsto**", value=f"{previsao_sim:,.0f} kg")
    st.metric(label="**Custo Total Estimado do Aço**", value=f"R$ {custo_estimado_sim:,.2f}")

with out_col2:
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = previsao_sim,
        title = {'text': "Consumo Previsto (kg)"},
        gauge = {
            'axis': {'range': [0, df_original['Consumo_Aco_kg'].max() * 1.2]},
            'bar': {'color': "#004B8D"},
            'steps' : [
                {'range': [0, 80000], 'color': "#D0DDE7"},
                {'range': [80000, 160000], 'color': "#A2BBD0"}
            ],
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)