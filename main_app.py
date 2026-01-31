import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import requests
from io import StringIO

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Datos - ETL y EDA",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("📊 Aplicación de Análisis de Datos")
st.markdown("### Ingesta, Procesamiento y Visualización Dinámica")

# ============================================================================
# MÓDULO 1: INGESTA Y PROCESAMIENTO (ETL)
# ============================================================================

st.header("🔧 Módulo 1: Ingesta y Procesamiento (ETL)")

# 1.1 Carga dinámica de datos
st.subheader("📂 Carga de Datos")

# Selector de fuente de datos
data_source = st.radio(
    "Selecciona la fuente de datos:",
    ["Archivo CSV", "Archivo JSON", "URL"]
)

df = None

if data_source == "Archivo CSV":
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Archivo cargado: {df.shape[0]} filas x {df.shape[1]} columnas")

elif data_source == "Archivo JSON":
    uploaded_file = st.file_uploader("Sube tu archivo JSON", type=['json'])
    if uploaded_file is not None:
        df = pd.read_json(uploaded_file)
        st.success(f"✅ Archivo cargado: {df.shape[0]} filas x {df.shape[1]} columnas")

elif data_source == "URL":
    url = st.text_input("Ingresa la URL del archivo CSV:")
    if url:
        try:
            if url.endswith('.csv'):
                df = pd.read_csv(url)
            elif url.endswith('.json'):
                df = pd.read_json(url)
            else:
                response = requests.get(url)
                df = pd.read_csv(StringIO(response.text))
            st.success(f"✅ Datos cargados desde URL: {df.shape[0]} filas x {df.shape[1]} columnas")
        except Exception as e:
            st.error(f"❌ Error al cargar datos desde URL: {str(e)}")

# Si hay datos cargados, proceder con limpieza y procesamiento
if df is not None:
    # Guardar copia original
    df_original = df.copy()

    st.divider()

    # 1.2 Limpieza Interactiva
    st.subheader("🧹 Limpieza Interactiva")

    col1, col2 = st.columns(2)

    with col1:
        # Checkbox para eliminar duplicados
        eliminar_duplicados = st.checkbox("Eliminar registros duplicados")
        if eliminar_duplicados:
            duplicados_antes = df.duplicated().sum()
            df = df.drop_duplicates()
            st.info(f"🗑️ Se eliminaron {duplicados_antes} registros duplicados")

    with col2:
        # Mostrar información de valores nulos
        nulos_totales = df.isnull().sum().sum()
        st.metric("Valores nulos en el dataset", nulos_totales)

    # Imputación de valores nulos en variables numéricas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(columnas_numericas) > 0 and df[columnas_numericas].isnull().sum().sum() > 0:
        st.markdown("#### Imputación de Valores Nulos (Variables Numéricas)")

        metodo_imputacion = st.selectbox(
            "Selecciona el método de imputación:",
            ["Ninguno", "Media", "Mediana", "Cero"]
        )

        if metodo_imputacion != "Ninguno":
            for col in columnas_numericas:
                if df[col].isnull().sum() > 0:
                    if metodo_imputacion == "Media":
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif metodo_imputacion == "Mediana":
                        df[col].fillna(df[col].median(), inplace=True)
                    elif metodo_imputacion == "Cero":
                        df[col].fillna(0, inplace=True)
            st.success(f"✅ Imputación completada usando: {metodo_imputacion}")

    # 1.3 Detección y tratamiento de valores atípicos (Outliers)
    st.markdown("#### Detección de Valores Atípicos (Outliers)")

    if len(columnas_numericas) > 0:
        tratar_outliers = st.checkbox("Detectar y tratar outliers usando método IQR")

        if tratar_outliers:
            col_outlier = st.selectbox(
                "Selecciona la variable para tratar outliers:",
                columnas_numericas
            )

            metodo_outlier = st.radio(
                "Método de tratamiento:",
                ["Eliminar", "Reemplazar con límites (IQR)", "Mantener (solo visualizar)"]
            )

            # Calcular IQR
            Q1 = df[col_outlier].quantile(0.25)
            Q3 = df[col_outlier].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR

            outliers_mask = (df[col_outlier] < limite_inferior) | (df[col_outlier] > limite_superior)
            num_outliers = outliers_mask.sum()

            st.info(f"📊 Detectados {num_outliers} outliers en '{col_outlier}'")
            st.write(f"Rango normal: [{limite_inferior:.2f}, {limite_superior:.2f}]")

            if metodo_outlier == "Eliminar" and num_outliers > 0:
                df = df[~outliers_mask]
                st.success(f"✅ Se eliminaron {num_outliers} outliers")
            elif metodo_outlier == "Reemplazar con límites (IQR)" and num_outliers > 0:
                df.loc[df[col_outlier] < limite_inferior, col_outlier] = limite_inferior
                df.loc[df[col_outlier] > limite_superior, col_outlier] = limite_superior
                st.success(f"✅ Se reemplazaron {num_outliers} outliers con los límites IQR")

    st.divider()

    # 1.4 Feature Engineering
    st.subheader("⚙️ Feature Engineering")
    st.markdown("Crea nuevas columnas calculadas a partir de las existentes")

    crear_feature = st.checkbox("Crear nueva columna calculada")

    if crear_feature:
        col1, col2, col3 = st.columns(3)

        with col1:
            nombre_nueva_col = st.text_input("Nombre de la nueva columna:", "Nueva_Columna")

        with col2:
            col_operando1 = st.selectbox("Primera columna:", columnas_numericas, key="op1")

        with col3:
            operacion = st.selectbox("Operación:", ["+", "-", "*", "/"])

        col_operando2 = st.selectbox("Segunda columna:", columnas_numericas, key="op2")

        if st.button("Crear columna"):
            try:
                if operacion == "+":
                    df[nombre_nueva_col] = df[col_operando1] + df[col_operando2]
                elif operacion == "-":
                    df[nombre_nueva_col] = df[col_operando1] - df[col_operando2]
                elif operacion == "*":
                    df[nombre_nueva_col] = df[col_operando1] * df[col_operando2]
                elif operacion == "/":
                    df[nombre_nueva_col] = df[col_operando1] / df[col_operando2].replace(0, np.nan)

                st.success(f"✅ Columna '{nombre_nueva_col}' creada exitosamente")
                columnas_numericas.append(nombre_nueva_col)
            except Exception as e:
                st.error(f"❌ Error al crear columna: {str(e)}")

    # Mostrar vista previa de los datos procesados
    st.divider()
    st.subheader("👀 Vista Previa de Datos Procesados")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Filas", df.shape[0])
    with col2:
        st.metric("Total de Columnas", df.shape[1])
    with col3:
        st.metric("Valores Nulos", df.isnull().sum().sum())

    # ============================================================================
    # MÓDULO 2: VISUALIZACIÓN DINÁMICA (EDA)
    # ============================================================================

    st.header("📈 Módulo 2: Visualización Dinámica (EDA)")

    # 2.1 Filtros Globales
    st.subheader("🔍 Filtros Globales")

    df_filtrado = df.copy()

    col1, col2, col3 = st.columns(3)

    # Filtro de fechas (detectar columnas de fecha)
    with col1:
        columnas_fecha = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].dropna().head(100), errors='raise')
                    columnas_fecha.append(col)
                except:
                    pass

        if columnas_fecha:
            col_fecha = st.selectbox("Columna de fecha:", ["Ninguna"] + columnas_fecha)
            if col_fecha != "Ninguna":
                df_filtrado[col_fecha] = pd.to_datetime(df_filtrado[col_fecha])
                min_fecha = df_filtrado[col_fecha].min().date()
                max_fecha = df_filtrado[col_fecha].max().date()

                rango_fechas = st.date_input(
                    "Rango de fechas:",
                    value=(min_fecha, max_fecha),
                    min_value=min_fecha,
                    max_value=max_fecha
                )

                if len(rango_fechas) == 2:
                    mask = (df_filtrado[col_fecha].dt.date >= rango_fechas[0]) & \
                           (df_filtrado[col_fecha].dt.date <= rango_fechas[1])
                    df_filtrado = df_filtrado[mask]

    # Filtro de categorías
    with col2:
        columnas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
        if columnas_categoricas:
            col_categoria = st.selectbox("Filtrar por categoría:", ["Ninguna"] + columnas_categoricas)
            if col_categoria != "Ninguna":
                valores_unicos = df_filtrado[col_categoria].dropna().unique().tolist()
                valores_seleccionados = st.multiselect(
                    f"Selecciona valores de '{col_categoria}':",
                    valores_unicos,
                    default=valores_unicos[:min(5, len(valores_unicos))]
                )
                if valores_seleccionados:
                    df_filtrado = df_filtrado[df_filtrado[col_categoria].isin(valores_seleccionados)]

    # Filtro de valores numéricos
    with col3:
        if columnas_numericas:
            col_numerica = st.selectbox("Filtrar por valor numérico:", ["Ninguna"] + columnas_numericas)
            if col_numerica != "Ninguna":
                min_val = float(df_filtrado[col_numerica].min())
                max_val = float(df_filtrado[col_numerica].max())

                rango_valores = st.slider(
                    f"Rango de '{col_numerica}':",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )

                df_filtrado = df_filtrado[
                    (df_filtrado[col_numerica] >= rango_valores[0]) &
                    (df_filtrado[col_numerica] <= rango_valores[1])
                ]

    st.info(f"📊 Datos filtrados: {df_filtrado.shape[0]} filas de {df.shape[0]} totales")

    st.divider()

    # 2.2 Pestañas para organizar análisis
    tab1, tab2, tab3 = st.tabs(["📊 Análisis Univariado", "🔗 Análisis Bivariado", "📋 Reporte"])

    # ========================================================================
    # TAB 1: ANÁLISIS UNIVARIADO
    # ========================================================================
    with tab1:
        st.subheader("Análisis Univariado")

        col1, col2 = st.columns(2)

        # Histogramas
        with col1:
            st.markdown("#### Distribución (Histograma)")
            if columnas_numericas:
                col_hist = st.selectbox("Selecciona variable:", columnas_numericas, key="hist")
                bins = st.slider("Número de bins:", 10, 100, 30)

                fig_hist = px.histogram(
                    df_filtrado,
                    x=col_hist,
                    nbins=bins,
                    title=f"Distribución de {col_hist}",
                    marginal="box"
                )
                fig_hist.update_layout(showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)

        # Boxplots
        with col2:
            st.markdown("#### Distribución (Boxplot)")
            if columnas_numericas:
                col_box = st.selectbox("Selecciona variable:", columnas_numericas, key="box")

                fig_box = px.box(
                    df_filtrado,
                    y=col_box,
                    title=f"Boxplot de {col_box}",
                    points="outliers"
                )
                st.plotly_chart(fig_box, use_container_width=True)

        # Estadísticas descriptivas
        st.markdown("#### Estadísticas Descriptivas")
        if columnas_numericas:
            st.dataframe(df_filtrado[columnas_numericas].describe(), use_container_width=True)

    # ========================================================================
    # TAB 2: ANÁLISIS BIVARIADO
    # ========================================================================
    with tab2:
        st.subheader("Análisis Bivariado")

        # Heatmap de correlaciones
        st.markdown("#### Matriz de Correlación (Heatmap)")
        if len(columnas_numericas) >= 2:
            # Seleccionar variables para correlación
            vars_corr = st.multiselect(
                "Selecciona variables para correlación:",
                columnas_numericas,
                default=columnas_numericas[:min(8, len(columnas_numericas))]
            )

            if len(vars_corr) >= 2:
                corr_matrix = df_filtrado[vars_corr].corr()

                fig_heatmap = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Matriz de Correlación"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

        st.divider()

        # Scatter plot
        if len(columnas_numericas) >= 2:
            st.markdown("#### Gráfico de Dispersión")
            col1, col2, col3 = st.columns(3)

            with col1:
                x_var = st.selectbox("Eje X:", columnas_numericas, key="scatter_x")
            with col2:
                y_var = st.selectbox("Eje Y:", columnas_numericas, index=min(1, len(columnas_numericas)-1), key="scatter_y")
            with col3:
                color_var = st.selectbox("Color (opcional):", ["Ninguno"] + columnas_categoricas, key="scatter_color")

            if color_var == "Ninguno":
                fig_scatter = px.scatter(
                    df_filtrado,
                    x=x_var,
                    y=y_var,
                    title=f"{y_var} vs {x_var}",
                    trendline="ols"
                )
            else:
                fig_scatter = px.scatter(
                    df_filtrado,
                    x=x_var,
                    y=y_var,
                    color=color_var,
                    title=f"{y_var} vs {x_var}"
                )

            st.plotly_chart(fig_scatter, use_container_width=True)

        st.divider()

        # Evolución temporal
        if columnas_fecha:
            st.markdown("#### Evolución Temporal")

            col_fecha_temp = st.selectbox("Columna de fecha:", columnas_fecha, key="temporal")
            col_valor_temp = st.selectbox("Variable a visualizar:", columnas_numericas, key="temporal_val")
            tipo_grafico = st.radio("Tipo de gráfico:", ["Línea", "Área"], horizontal=True)

            df_temp = df_filtrado.copy()
            df_temp[col_fecha_temp] = pd.to_datetime(df_temp[col_fecha_temp])
            df_temp = df_temp.sort_values(col_fecha_temp)

            if tipo_grafico == "Línea":
                fig_temp = px.line(
                    df_temp,
                    x=col_fecha_temp,
                    y=col_valor_temp,
                    title=f"Evolución de {col_valor_temp} en el tiempo"
                )
            else:
                fig_temp = px.area(
                    df_temp,
                    x=col_fecha_temp,
                    y=col_valor_temp,
                    title=f"Evolución de {col_valor_temp} en el tiempo"
                )

            st.plotly_chart(fig_temp, use_container_width=True)

    # ========================================================================
    # TAB 3: REPORTE
    # ========================================================================
    with tab3:
        st.subheader("📋 Reporte Completo del Análisis")

        st.markdown("#### Resumen del Dataset")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Filas", df_filtrado.shape[0])
        with col2:
            st.metric("Total Columnas", df_filtrado.shape[1])
        with col3:
            st.metric("Variables Numéricas", len(columnas_numericas))
        with col4:
            st.metric("Variables Categóricas", len(columnas_categoricas))

        st.markdown("#### Información del Dataset")

        # Crear tabla de información
        info_data = []
        for col in df_filtrado.columns:
            info_data.append({
                'Columna': col,
                'Tipo': str(df_filtrado[col].dtype),
                'Valores Únicos': df_filtrado[col].nunique(),
                'Nulos': df_filtrado[col].isnull().sum(),
                '% Nulos': f"{(df_filtrado[col].isnull().sum() / len(df_filtrado) * 100):.2f}%"
            })

        df_info = pd.DataFrame(info_data)
        st.dataframe(df_info, use_container_width=True)

        st.markdown("#### Datos Procesados (Primeras 100 filas)")
        st.dataframe(df_filtrado.head(100), use_container_width=True)

        # Botón para descargar datos procesados
        csv = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar datos procesados (CSV)",
            data=csv,
            file_name='datos_procesados.csv',
            mime='text/csv',
        )

else:
    st.info("👆 Por favor, carga un archivo de datos para comenzar el análisis")

    # Información de ayuda
    with st.expander("ℹ️ Información de uso"):
        st.markdown("""
        ### Cómo usar esta aplicación:

        **Módulo 1: Ingesta y Procesamiento (ETL)**
        1. Selecciona la fuente de datos (CSV, JSON o URL)
        2. Carga tu archivo o ingresa la URL
        3. Usa las opciones de limpieza:
           - Elimina duplicados
           - Imputa valores nulos
           - Trata outliers
        4. Crea nuevas columnas calculadas (Feature Engineering)

        **Módulo 2: Visualización Dinámica (EDA)**
        1. Aplica filtros globales (fechas, categorías, valores numéricos)
        2. Explora los diferentes análisis en las pestañas:
           - **Univariado**: Histogramas, boxplots y estadísticas
           - **Bivariado**: Correlaciones, dispersión y evolución temporal
           - **Reporte**: Resumen completo y descarga de datos
        """)
