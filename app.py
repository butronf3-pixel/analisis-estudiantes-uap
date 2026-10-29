import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

# Configuración de la página
st.set_page_config(page_title="Análisis de Datos Estudiantes UAP", layout="wide")

st.title("📊 Análisis Estadístico - Estudiantes UAP")
st.write("Estadística Descriptiva y Prueba de Hipótesis")

# ---------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Cambia 'datos_estudiantes.csv' por el nombre de tu archivo CSV
    return pd.read_csv('datos_estudiantes.csv')

try:
    df = load_data()
    st.sidebar.success("✅ Datos cargados correctamente")
except Exception as e:
    st.error("Por favor sube tu archivo 'datos_estudiantes.csv' al mismo directorio.")
    st.stop()

# ---------------------------------------------------------
# PESTAÑAS PRINCIPALES
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📈 Estadística Descriptiva", "🧪 Pruebas de Hipótesis"])

# =========================================================
# PASO 2: ESTADÍSTICA DESCRIPTIVA
# =========================================================
with tab1:
    st.header("1. Resumen Estadístico Descriptivo")
    
    # Vista previa de datos
    if st.checkbox("Mostrar tabla de datos"):
        st.dataframe(df)

    # Selección de variable para analizar
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Variables Cuantitativas")
        if num_cols:
            var_num = st.selectbox("Selecciona una variable numérica:", num_cols)
            
            # Métricas
            mean_val = df[var_num].mean()
            median_val = df[var_num].median()
            std_val = df[var_num].std()
            q1 = df[var_num].quantile(0.25)
            q3 = df[var_num].quantile(0.75)
            iqr = q3 - q1

            m1, m2, m3 = st.columns(3)
            m1.metric("Media", f"{mean_val:.2f}")
            m2.metric("Mediana", f"{median_val:.2f}")
            m3.metric("Desv. Estándar", f"{std_val:.2f}")

            m4, m5 = st.columns(2)
            m4.metric("Rango (Min - Max)", f"{df[var_num].min():.2f} - {df[var_num].max():.2f}")
            m5.metric("IQR (Q3 - Q1)", f"{iqr:.2f}")

            # Gráficos
            fig_hist = px.histogram(df, x=var_num, marginal="box", title=f"Distribución de {var_num}")
            st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Variables Cualitativas")
        if cat_cols:
            var_cat = st.selectbox("Selecciona una variable categórica:", cat_cols)
            
            # Tabla de frecuencias
            freq_df = df[var_cat].value_counts().reset_index()
            freq_df.columns = [var_cat, 'Frecuencia']
            st.dataframe(freq_df)

            # Gráfico de barras
            fig_bar = px.bar(freq_df, x=var_cat, y='Frecuencia', title=f"Frecuencia por {var_cat}")
            st.plotly_chart(fig_bar, use_container_width=True)

# =========================================================
# PASO 3: PRUEBAS DE HIPÓTESIS
# =========================================================
with tab2:
    st.header("2. Pruebas de Hipótesis (Paramétricas y No Paramétricas)")

    tipo_prueba = st.radio(
        "Selecciona el tipo de análisis:",
        ["Prueba de Normalidad (Shapiro-Wilk)", 
         "Comparación de 2 Grupos (t-Student / Mann-Whitney)", 
         "Independencia Categórica (Chi-Cuadrada)"]
    )

    alpha = st.slider("Nivel de significancia (α):", 0.01, 0.10, 0.05)

    # A) NORMALIDAD
    if tipo_prueba == "Prueba de Normalidad (Shapiro-Wilk)":
        st.subheader("Prueba de Shapiro-Wilk")
        var = st.selectbox("Variable continua a evaluar:", num_cols, key="shapiro")
        
        stat, p_val = stats.shapiro(df[var].dropna())
        
        st.write(f"**Estadístico W:** {stat:.4f}")
        st.write(f"**p-valor:** {p_val:.4f}")
        
        if p_val > alpha:
            st.success(f"Con α = {alpha}, **no se rechaza H0**. Los datos siguen una distribución normal.")
        else:
            st.warning(f"Con α = {alpha}, **se rechaza H0**. Los datos NO siguen una distribución normal (usar pruebas no paramétricas).")

    # B) COMPARACIÓN DE 2 GRUPOS
    elif tipo_prueba == "Comparación de 2 Grupos (t-Student / Mann-Whitney)":
        st.subheader("Comparación entre dos grupos")
        
        var_continua = st.selectbox("Variable numérica a comparar:", num_cols, key="comp_num")
        var_grupo = st.selectbox("Variable de agrupación (debe tener 2 categorías):", cat_cols, key="comp_cat")
        
        grupos = df[var_grupo].dropna().unique()
        
        if len(grupos) == 2:
            g1 = df[df[var_grupo] == grupos[0]][var_continua].dropna()
            g2 = df[df[var_grupo] == grupos[1]][var_continua].dropna()

            es_parametrica = st.checkbox("Usar prueba Paramétrica (t-Student)", value=True)

            if es_parametrica:
                stat, p_val = stats.ttest_ind(g1, g2)
                nombre_prueba = "t-Student para muestras independientes"
            else:
                stat, p_val = stats.mannwhitneyu(g1, g2)
                nombre_prueba = "U de Mann-Whitney (No paramétrica)"

            st.write(f"**Prueba utilizada:** {nombre_prueba}")
            st.write(f"**Estadístico:** {stat:.4f}")
            st.write(f"**p-valor:** {p_val:.4f}")

            if p_val < alpha:
                st.success(f"**Resultado:** Diferencia estadísticamente significativa entre {grupos[0]} y {grupos[1]} (p < {alpha}).")
            else:
                st.info(f"**Resultado:** No hay diferencia estadísticamente significativa entre los grupos (p ≥ {alpha}).")

            # Visualización del contraste
            fig_box = px.box(df, x=var_grupo, y=var_continua, color=var_grupo, title=f"{var_continua} por {var_grupo}")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.error("La variable categórica seleccionada debe tener exactamente 2 categorías.")

    # C) CHI-CUADRADA
    elif tipo_prueba == "Independencia Categórica (Chi-Cuadrada)":
        st.subheader("Prueba Chi-Cuadrada de Independencia (No Paramétrica)")
        
        c1 = st.selectbox("Variable 1:", cat_cols, key="chi1")
        c2 = st.selectbox("Variable 2:", cat_cols, key="chi2")

        if c1 != c2:
            tabla_contingencia = pd.crosstab(df[c1], df[c2])
            st.write("Tabla de contingencia:")
            st.dataframe(tabla_contingencia)

            chi2, p_val, dof, ex = stats.chi2_contingency(tabla_contingencia)

            st.write(f"**Estadístico Chi²:** {chi2:.4f}")
            st.write(f"**Grados de libertad:** {dof}")
            st.write(f"**p-valor:** {p_val:.4f}")

            if p_val < alpha:
                st.success(f"**Resultado:** Existe asociación significativa entre {c1} y {c2} (p < {alpha}).")
            else:
                st.info(f"**Resultado:** Las variables son independientes (p ≥ {alpha}).")
        else:
            st.error("Selecciona dos variables categóricas distintas.")