import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Cargar datos
df = pd.read_csv('data/insurance.csv')

# Convertir variables categóricas
df = pd.get_dummies(df, drop_first=True)

# Dividir datos
X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Guardar modelo
joblib.dump(model, 'modulos/seguro.pkl')

print("Modelo de costos médicos entrenado y guardado con éxito.")
