# 📊 Aplicación de Análisis de Datos - ETL y EDA

Aplicación web interactiva desarrollada con Streamlit para el análisis exploratorio de datos (EDA) y procesos de extracción, transformación y carga (ETL).

## 🚀 Características

### Módulo 1: Ingesta y Procesamiento (ETL)
- **Carga dinámica de datos**: Soporte para CSV, JSON y URLs
- **Limpieza interactiva**:
  - Eliminación de duplicados
  - Imputación de valores nulos (Media, Mediana, Cero)
  - Detección y tratamiento de outliers usando método IQR
- **Feature Engineering**: Creación de nuevas columnas calculadas

### Módulo 2: Visualización Dinámica (EDA)
- **Filtros globales**: Por fechas, categorías y valores numéricos
- **Análisis Univariado**: Histogramas, boxplots y estadísticas descriptivas
- **Análisis Bivariado**:
  - Matriz de correlación (Heatmap)
  - Gráficos de dispersión con líneas de tendencia
  - Evolución temporal (Line/Area Charts)
- **Reporte completo**: Resumen del dataset y descarga de datos procesados

## 📋 Requisitos Previos

- Python 3.11 o superior
- pip (gestor de paquetes de Python)
- Git (opcional, para clonar el repositorio)

## 🛠️ Instalación

### 1. Clonar el repositorio (opcional)

```bash
git clone <url-del-repositorio>
cd trabajo_final_ciencia_datos
```

O simplemente descarga los archivos del proyecto.



### 2. Instalar las dependencias

```bash
pip install -r requirements.txt
```

Esto instalará todas las librerías necesarias:
- streamlit (≥1.31.0)
- pandas (≥2.0.0)
- numpy (≥1.24.0)
- matplotlib (≥3.7.0)
- seaborn (≥0.12.0)
- plotly (≥5.18.0)

## 🎯 Uso

### Ejecutar la aplicación localmente

```bash
streamlit run main_app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Si el navegador no se abre automáticamente

Abre manualmente tu navegador y visita: `http://localhost:8501`

## 📖 Guía de Uso

### Paso 1: Cargar Datos
1. Selecciona la fuente de datos (CSV, JSON o URL)
2. Sube tu archivo o ingresa la URL
3. El sistema mostrará información sobre el tamaño y estructura del dataset

### Paso 2: Limpieza de Datos
1. **Eliminar duplicados**: Activa el checkbox si deseas remover registros duplicados
2. **Imputar valores nulos**: Selecciona el método de imputación (Media, Mediana o Cero)
3. **Tratar outliers**: Detecta y trata valores atípicos usando el método IQR

### Paso 3: Feature Engineering
1. Crea nuevas columnas calculadas
2. Selecciona dos columnas numéricas y una operación (+, -, *, /)
3. Asigna un nombre a la nueva columna

### Paso 4: Análisis Exploratorio
1. **Aplica filtros globales** (opcional):
   - Rango de fechas
   - Categorías específicas
   - Valores numéricos

2. **Explora las pestañas**:
   - **Análisis Univariado**: Distribuciones y estadísticas
   - **Análisis Bivariado**: Correlaciones y relaciones entre variables
   - **Reporte**: Vista completa y descarga de datos procesados

## 📁 Estructura del Proyecto

```
trabajo_final_ciencia_datos/
│
├── main_app.py                          # Aplicación principal de Streamlit
├── requirements.txt                     # Dependencias del proyecto
├── runtime.txt                          # Versión de Python para deployment
├── README.md                            # Este archivo
│
├── .streamlit/
│   └── config.toml                      # Configuración de Streamlit
│
└── verificar_dataset.ipynb              # Notebook de verificación del dataset
```


```
