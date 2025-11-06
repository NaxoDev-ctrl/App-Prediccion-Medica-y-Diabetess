import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import joblib

# Cargar datos
df = pd.read_csv('data/diabetes.csv')

# Verificar las columnas disponibles
print("Columnas en el dataset:", df.columns.tolist())

# Usar el nombre correcto de la columna (cambiar si es diferente)
X = df.drop('outcome', axis=1)
y = df['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Calcular umbral ideal
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
optimal_threshold = thresholds[(tpr - fpr).argmax()]

print(f"Umbral ideal: {optimal_threshold:.2f}")

# Guardar modelo
joblib.dump(model, 'modulos/diabetes.pkl')
print("Modelo guardado exitosamente!")