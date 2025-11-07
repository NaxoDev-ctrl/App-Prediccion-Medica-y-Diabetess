Integrantes:
Tamara Larenas,
Ivan Hernandez,
Ignacio Sanhueza

link: app-prediccion-medica-y-diabetess-production.up.railway.app

# Aplicación de Predicciones Médicas
Una aplicación web interactiva construida con Streamlit que predice costos de seguros médicos y riesgo de diabetes usando modelos de machine learning.

## Características

### Predicción de Seguros Médicos
Calcula el costo estimado de un seguro médico basado en:
  - Edad
  - Índice de masa corporal (BMI)
  - Número de hijos
  - Hábito de fumar
  - Sexo
  - Región geográfica

### Diagnóstico de Diabetes
Predice la probabilidad de diabetes considerando:
  - Número de embarazos
  - Nivel de glucosa
  - Índice de masa corporal
  - Edad

## Instalación

1. **Clona o descarga el proyecto**
2. **Crea un entorno virtual:**
   ```bash
   python -m venv venv
3. **Instala requirements**
4. **Inicia la app**
   ```bash
   streamlit run app.py
Preguntas:
1) ¿Cuál es el umbral ideal para el modelo de predicción de diabetes?
2) ¿Cuales son los factores que más influyen en el precio de los costos asociados al seguro médico?
3) Hacer un análisis comparativo de cada características de ambos modelos utilizando RandomForest.
4) ¿Qué técnica de optimización mejora el rendimiento de ambos modelos?
5) Explicar contexto de los datos.
6) Analizar el sesgo que presentan los modelos y explicar porqué.

Respuestas:
1.	Umbral ideal para el modelo de predicción de diabetes
   
    •	Si los costos de FN y FP son similares: 0.50.

    •	Para uso de cribado (priorizar sensibilidad/recall): 0.40 es un umbral práctico inicial; afínalo maximizando F1 o Youden (TPR−FPR) en validación, con objetivo de sensibilidad ≥0.85.


3.	Factores que más influyen en el costo del seguro médico
   
    •	Influencia Principal: fumador (smoker) domina el costo.

    •	Influencia media: edad y BMI (obesidad).

    •	Influencia menor: número de hijos (children), región; sexo suele ser marginal.


5.	Análisis comparativo de características con RandomForest
   
    •	Seguro (charges): importancia típica → smoker > age > BMI > children ≈ region ≈ sex.

    •	Diabetes: importancia típica → glucose > BMI > age > pregnancies/insulin/blood_pressure (dependiendo de calidad de mediciones y nulos).

    •	Conclusión comparativa: en seguro domina un factor conductual (tabaco); en diabetes dominan marcadores metabólicos (glucosa, adiposidad) y edad.


7.	Técnica de optimización que mejora ambos modelos
   
    •	Validación cruzada estratificada + búsqueda bayesiana o aleatoria de hiperparámetros con regularización:

       o	Clasificación (diabetes): calibrar y ajustar umbral tras entrenar (Platt/Isotónica), ponderar clases (class_weight) si hay desbalance.
  
       o	Regresión (seguro): usar regularización (Ridge/Lasso/ElasticNet) y transformar objetivo (log-charges) para estabilidad.


9.	Contexto de los datos
    
    •	insurance.csv: dataset clásico (Kaggle) con age, sex, bmi, children, smoker, region y charges (costo).

    •	diabetes.csv: versión del Pima Indians Diabetes con variables clínicas (pregnancies, glucose, bloodPressure, skinThickness, insulin, BMI, diabetesPedigreeFunction, age) y outcome (0/1).


11.	Sesgo de los modelos y por qué
    
    •	Seguro: fuerte dependencia en “smoker” puede inducir decisiones muy agresivas contra fumadores; posibles sesgos por región/sexo si la muestra no es representativa.

    •	Diabetes: sesgo poblacional (mujeres Pima) limita generalización a otras etnias/edades/sexos; medidas como insulin/skinThickness pueden tener muchos faltantes, sesgando importancias y calibración.

    •	Ambos: riesgo de sesgo de muestreo y de medición; mitigación con recolección diversa, calibración, y evaluación por subgrupos.
  
