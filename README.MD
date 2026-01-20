# Inti-Gap: Solar Material Discovery with AI

> **Acelerando el Descubrimiento Fotovoltaico mediante Inteligencia Artificial**

[![Estado](https://img.shields.io/badge/Estado-Prototipo%20Funcional%20v1.0-success)]()
[![Python](https://img.shields.io/badge/Python-3.8+-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

---

## 📋 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [Características Principales](#-características-principales)
- [Rendimiento del Modelo](#-rendimiento-del-modelo)
- [Instalación](#️-instalación)
- [Guía de Uso](#-guía-de-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Información Técnica](#-información-técnica)
- [Autor](#-autor)

---

## 🔬 Descripción General

**Inti-Gap** es un pipeline de Materials Informatics diseñado para predecir la brecha energética (Band Gap) de compuestos inorgánicos mediante técnicas de Machine Learning. El proyecto busca acelerar el proceso de descubrimiento de nuevos materiales para celdas solares, reduciendo significativamente la necesidad de costosas pruebas experimentales de laboratorio.

### Motivación

El desarrollo de nuevos materiales fotovoltaicos es un proceso tradicionalmente lento y costoso. Inti-Gap permite realizar un filtrado rápido de candidatos viables, optimizando recursos y tiempo en la investigación de materiales para energía solar.

---

## ✨ Características Principales

Inti-Gap implementa un flujo de trabajo completo que incluye:

1. **Ingesta de Datos**: Procesamiento de fórmulas químicas en formato texto (ej: `GaAs`, `SnO2`, `CdTe`)

2. **Extracción Automática de Características**: Generación de descriptores físico-químicos utilizando la biblioteca `matminer`:
   - Masa atómica promedio
   - Electronegatividad de Pauling
   - Estados de valencia
   - Propiedades de elementos individuales (Preset: Magpie)

3. **Predicción de Band Gap**: Clasificación automática del material como:
   - **Metal** (Band Gap ≈ 0 eV)
   - **Semiconductor** (Band Gap: 0.5 - 4.0 eV)
   - **Aislante** (Band Gap > 4.0 eV)

---

## 📊 Rendimiento del Modelo

### Especificaciones Técnicas

| Parámetro | Valor |
|-----------|-------|
| **Algoritmo** | Random Forest Regressor |
| **Dataset de Entrenamiento** | Matbench Expt Gap (4,600 materiales) |
| **Error (RMSE)** | ~0.72 eV |
| **Características** | 132 descriptores químicos |

### Validación con Materiales Conocidos

| Material | Band Gap Real | Predicción Inti-Gap | Clasificación |
|----------|---------------|---------------------|---------------|
| **GaAs** | ~1.42 eV | ~1.01 eV | ✅ Semiconductor |
| **SnO2** | ~3.60 eV | ~3.14 eV | ✅ Óxido Transparente |
| **CdTe** | ~1.50 eV | ~1.48 eV* | ✅ Semiconductor |

*Valores ilustrativos basados en el rendimiento típico del modelo.

---

## 🛠️ Instalación

### Requisitos del Sistema

- Python 3.8 o superior
- Google Colab (recomendado) o entorno Jupyter local
- 2 GB de RAM disponibles

### Instalación de Dependencias

Ejecute el siguiente comando en su entorno de Colab o terminal:

```bash
!pip install pymatgen matminer scikit-learn joblib pandas numpy
```

### Verificación de Instalación

```python
import pymatgen
import matminer
import sklearn
print("✅ Todas las dependencias instaladas correctamente")
```

---

## 💻 Guía de Uso

### 1. Cargar el Modelo Pre-entrenado

```python
import joblib
import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

# Cargar el modelo entrenado
model = joblib.load('inti_gap_model.pkl')
print("✅ Modelo Inti-Gap cargado exitosamente")
```

### 2. Función de Predicción

```python
def predecir_band_gap(formula_quimica):
    """
    Predice el Band Gap de un material a partir de su fórmula química.
    
    Args:
        formula_quimica (str): Fórmula química del compuesto (ej: 'GaAs', 'SnO2')
    
    Returns:
        float: Band Gap predicho en electronvoltios (eV)
    """
    # Crear DataFrame temporal
    df_temp = pd.DataFrame({"formula": [formula_quimica]})
    
    # Feature Engineering: Convertir texto a composición química
    df_temp = StrToComposition().featurize_dataframe(df_temp, "formula")
    
    # Extraer propiedades elementales (132 características)
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df_temp = ep_feat.featurize_dataframe(df_temp, col_id="composition")
    
    # Preparar entrada para el modelo
    X_input = df_temp.drop(columns=['formula', 'composition'])
    
    # Realizar predicción
    prediccion = model.predict(X_input)[0]
    return prediccion
```

### 3. Ejemplo de Uso

```python
# Predecir Band Gap para Telururo de Cadmio (CdTe)
material = "CdTe"
gap_predicho = predecir_band_gap(material)

print("=" * 50)
print(f"🧪 Material Analizado: {material}")
print(f"⚡ Band Gap Predicho: {gap_predicho:.4f} eV")
print("=" * 50)

# Clasificación automática
if gap_predicho < 0.5:
    print("📊 Clasificación: Metal/Conductor")
elif gap_predicho < 4.0:
    print("📊 Clasificación: Semiconductor")
else:
    print("📊 Clasificación: Aislante")
```

### 4. Predicción por Lotes

```python
# Analizar múltiples materiales
materiales = ["Si", "GaAs", "CdTe", "Cu2O", "ZnO", "TiO2"]

resultados = []
for mat in materiales:
    gap = predecir_band_gap(mat)
    resultados.append({"Material": mat, "Band Gap (eV)": round(gap, 2)})

# Mostrar resultados en formato tabla
import pandas as pd
df_resultados = pd.DataFrame(resultados)
print(df_resultados.to_string(index=False))
```

---

## 📂 Estructura del Proyecto

```
Inti-Gap/
│
├── 📓 Inti_Gap_Training.ipynb    # Notebook principal de entrenamiento
├── 🤖 inti_gap_model.pkl         # Modelo pre-entrenado (Random Forest)
├── 📄 README.md                   # Documentación del proyecto
├── 📊 data/                       # Datasets de entrenamiento (opcional)
│   └── matbench_expt_gap.csv
└── 🔬 notebooks/                  # Notebooks de exploración (opcional)
    └── exploratory_analysis.ipynb
```

---

## 🔍 Información Técnica

### Pipeline de Procesamiento

```
Fórmula Química (Texto)
        ↓
    Parsing con PyMatGen
        ↓
  Extracción de 132 Features
  (Masa, Electronegatividad, Radio, etc.)
        ↓
    Random Forest Regressor
        ↓
   Predicción de Band Gap (eV)
```

### Limitaciones Conocidas

- **Dominio de aplicación**: Optimizado para compuestos inorgánicos binarios y ternarios
- **Precisión variable**: Mayor error en materiales con propiedades exóticas (superconductores, topológicos)
- **Dependencia de datos**: Rendimiento limitado por la calidad del dataset Matbench
- **Sin extrapolación cuántica**: No sustituye cálculos DFT para diseño de precisión

### Próximas Mejoras (Roadmap)

- [ ] Integración con bases de datos experimentales (Materials Project API)
- [ ] Implementación de modelos de Deep Learning (Graph Neural Networks)
- [ ] Predicción de múltiples propiedades (movilidad de portadores, estabilidad térmica)
- [ ] Interfaz web para uso sin código

---

## 👤 Autor

**Julián Gómez Brizuela**

- 📍 Ubicación: Santiago, Chile 🇨🇱
- 🔬 Área: Materials Informatics & Machine Learning
- 📧 Contacto: [Disponible bajo solicitud]

---

## 📜 Licencia

Este proyecto se distribuye bajo la Licencia MIT. Consulte el archivo `LICENSE` para más detalles.

---

## 🙏 Agradecimientos

- **Matminer Team**: Por la biblioteca de featurización química
- **Materials Project**: Por el dataset Matbench
- **PyMatGen Community**: Por las herramientas de ciencia computacional de materiales

---

<div align="center">

**¿Tienes preguntas o sugerencias?**  
¡Abre un Issue o contacta al autor!

Hecho con ☀️ para acelerar la transición energética

</div>
