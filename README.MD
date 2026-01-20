Markdown
# 🌞 Inti-Gap: Solar Material Discovery with AI

> **"Accelerating Photovoltaic Discovery"**

Este proyecto implementa un pipeline de **Materials Informatics** diseñado para predecir la brecha energética (*Band Gap*) de compuestos inorgánicos utilizando Machine Learning. Su objetivo es filtrar rápidamente candidatos viables para nuevas celdas solares, reduciendo la necesidad de pruebas de laboratorio costosas.

---

## 🚀 Descripción del Proyecto

**Inti-Gap** utiliza algoritmos de Random Forest y técnicas de ingeniería de características químicas para:
1.  **Ingestar** fórmulas químicas brutas (ej: `GaAs`, `SnO2`).
2.  **Extraer** propiedades físico-químicas automáticas (masa atómica, electronegatividad, valencia) usando `matminer`.
3.  **Predecir** el Band Gap (eV) para determinar si un material es metal, semiconductor o aislante.

### 🎯 Rendimiento del Modelo
El modelo actual (`v1.0`) ha sido entrenado con el dataset `matbench_expt_gap` (4,600 materiales reales).
- **Algoritmo:** Random Forest Regressor.
- **Error (RMSE):** ~0.72 eV.
- **Validación:**
  - *GaAs (Real ~1.42 eV)* -> Predicción Inti-Gap: **~1.01 eV** (Correctamente identificado como semiconductor).
  - *SnO2 (Real ~3.60 eV)* -> Predicción Inti-Gap: **~3.14 eV** (Correctamente identificado como óxido transparente).

---

## 🛠️ Instalación (Google Colab)

Para ejecutar este proyecto o usar el modelo guardado, es necesario instalar las dependencias de ciencia de materiales:

```bash
!pip install pymatgen matminer scikit-learn joblib
💾 Persistencia del Modelo (Cómo usarlo)
El modelo entrenado se guarda automáticamente como inti_gap_model.pkl. No es necesario re-entrenar cada vez.

Cargar y Predecir con el modelo guardado:
Python
import joblib
import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

# 1. Cargar el "cerebro" (Asegúrate de tener el archivo .pkl en tu entorno)
model = joblib.load('inti_gap_model.pkl')
print("✅ Modelo Inti-Gap cargado exitosamente.")

# 2. Función auxiliar para procesar la fórmula química
def predecir_material(formula_quimica):
    # Crear dataframe temporal
    df_temp = pd.DataFrame({"formula": [formula_quimica]})
    
    # Feature Engineering (Transformar texto a números)
    # Nota: Usamos las mismas transformaciones que en el entrenamiento
    df_temp = StrToComposition().featurize_dataframe(df_temp, "formula")
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df_temp = ep_feat.featurize_dataframe(df_temp, col_id="composition")
    
    # Limpiar columnas extra
    X_input = df_temp.drop(columns=['formula', 'composition'])
    
    # Predecir
    prediccion = model.predict(X_input)[0]
    return prediccion

# 3. ¡Prueba tu propio material!
material = "CdTe"  # Telururo de Cadmio (usado en paneles solares thin-film)
gap = predecir_material(material)

print(f"---------------------------------------------")
print(f"🧪 Material: {material}")
print(f"⚡ Band Gap Predicho: {gap:.4f} eV")
print(f"---------------------------------------------")
📂 Estructura de Archivos
Inti_Gap_Training.ipynb: Cuaderno principal de entrenamiento y visualización.

inti_gap_model.pkl: Archivo binario con el modelo entrenado (listo para producción).

README.md: Documentación del proyecto.

Autor: Julián Gómez Brizuela
Estado: Prototipo Funcional v1.0 Ubicación: Santiago, Chile 🇨🇱


***
.
