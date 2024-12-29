# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib
from scipy.stats import pearsonr
import seaborn as sns
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from datetime import datetime
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Funktion zum Hinzufügen der letzten Null bei Postleitzahlen
def add_leading_zero(df, column):
    df[column] = df[column].astype(str)  # In String umwandeln
    df[column] = df[column].apply(lambda x: x + '0' if len(x) == 4 else x)
    return df

# Benutzerdefinierte RMSE-Funktion
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Einlesen der Dateien
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#Hintere Null für cityCode hinzufügen
train_data = add_leading_zero(train_data, 'cityCode')
test_data = add_leading_zero(test_data, 'cityCode')

# Für cityCode nur erste beiden Ziffern verwenden
train_data['cityCode'] = train_data['cityCode'].str[:2]

# Heutiges Jahr ermitteln
current_year = datetime.now().year

# Alter des Hauses berechnen
train_data['age'] = current_year - train_data['made']

# Spearman-Korrelationsmatrix
spearman_corr = train_data.corr(method='spearman')
print("Spearman-Korrelationsmatrix:")

# Visualisierung  Korrelationsmatrix mit einer Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Korrelationsmatrix nach Spearman")
plt.show()

# --- Korrelationsplots mit Zielvariable ---
# Ziel- und Feature-Variablen
target = "price"
features = ["hasYard","hasPool", "cityPartRange", "numberOfRooms", "floors", "numPrevOwners",
            "isNewBuilt","hasStormProtector", "hasStorageRoom","hasGuestRoom","age", "basement", "attic", "garage"]


# Korrelationsplots der x-Variablen in Abhängigkeit der y-Variable
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=train_data[feature], y=train_data[target])
    plt.title(f"Korrelation zwischen {feature} und {target}")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Spalte 'made' entfernen
train_data = train_data.drop(columns=['made'])

# One-Hot-Encoding für cityCode
train_data = pd.get_dummies(train_data, columns=['cityCode'])

# Numerische Variablen im DataFrame ermitteln
numerical_features = ["squareMeters", "numberOfRooms", "floors", "cityPartRange", "numPrevOwners", "age", "basement", "attic", "garage", "hasGuestRoom"]

# Standardisierung der numerischen Variablen (Z-Score-Normalisierung)
scaler_standard = StandardScaler()
train_data[numerical_features] = scaler_standard.fit_transform(train_data[numerical_features])

# --- Hypothesentest zur Untersuchung des Zusammenhang "sqaureMeters" und "price" ---

# Berechne den Pearson-Korrelationskoeffizienten und p-Wert
corr, p_value = pearsonr(train_data['squareMeters'], train_data['price'])

# Ausgabe der Ergebnisse
print(f"Pearson-Korrelationskoeffizient: {corr}")
print(f"P-Wert: {p_value}")

# Hypothesen testen mit Signifikanzniveau (5%)
alpha = 0.05

if p_value < alpha:
    print("Die Nullhypothese wird abgelehnt: Es gibt einen signifikanten linearen Zusammenhang.")
    print("\n")
else:
    print("Die Nullhypothese wird nicht abgelehnt: Es gibt keinen signifikanten linearen Zusammenhang.")
    print("\n")


# Daten für die Modelle aufteilen
X_all = train_data.drop(columns=['price'])
y = train_data['price']

# Trainings- und Testdaten splitten
X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(X_all, y, test_size=0.2, random_state=42)

# Zielvariable skalieren (da y große Werte hat)
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).flatten()

# Berechnung der RMSE
rmse_scorer = make_scorer(rmse)

# --- Lineare Regression ---
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train_scaled)

# Vorhersagen
y_train_pred_linear = linear_model.predict(X_train_scaled)
y_val_pred_linear = linear_model.predict(X_val_scaled)

# RMSE berechnen
train_rmse_linear = rmse(y_train_scaled, y_train_pred_linear)
val_rmse_linear = rmse(y_val_scaled, y_val_pred_linear)

# --- Rücktransformation der Vorhersagen ---
y_train_pred_linear_original = y_scaler.inverse_transform(y_train_pred_linear.reshape(-1, 1)).flatten()
y_val_pred_linear_original = y_scaler.inverse_transform(y_val_pred_linear.reshape(-1, 1)).flatten()

# RMSE auf Originalskala berechnen
train_rmse_linear_original = rmse(y_train, y_train_pred_linear_original)
val_rmse_linear_original = rmse(y_val, y_val_pred_linear_original)

print(f"Linear Regression - Train RMSE (original): {train_rmse_linear_original:.4f}")
print(f"Linear Regression - Validation RMSE (original): {val_rmse_linear_original:.4f}\n")


# --- XGBoost ---
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_scaled, y_train_scaled)

# Vorhersagen
y_train_pred_xgb = xgb_model.predict(X_train_scaled)
y_val_pred_xgb = xgb_model.predict(X_val_scaled)

# RMSE berechnen
train_rmse_xgb = np.sqrt(mean_squared_error(y_train_scaled, y_train_pred_xgb))
val_rmse_xgb = np.sqrt(mean_squared_error(y_val_scaled, y_val_pred_xgb))

# Rücktransformation der Vorhersagen
y_train_pred_xgb_original = y_scaler.inverse_transform(y_train_pred_xgb.reshape(-1, 1)).flatten()
y_val_pred_xgb_original = y_scaler.inverse_transform(y_val_pred_xgb.reshape(-1, 1)).flatten()

train_rmse_xgb_original = rmse(y_train, y_train_pred_xgb_original)
val_rmse_xgb_original = rmse(y_val, y_val_pred_xgb_original)

print(f"XGBoost - Train RMSE: {train_rmse_xgb_original:.4f}")
print(f"XGBoost - Validation RMSE: {val_rmse_xgb_original:.4f}\n")


# --- LightGBM ---
lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgbm_model.fit(X_train_scaled, y_train_scaled)

# Vorhersagen
y_train_pred_lgbm = lgbm_model.predict(X_train_scaled)
y_val_pred_lgbm = lgbm_model.predict(X_val_scaled)

# RMSE berechnen
train_rmse_lgbm = np.sqrt(mean_squared_error(y_train_scaled, y_train_pred_lgbm))
val_rmse_lgbm = np.sqrt(mean_squared_error(y_val_scaled, y_val_pred_lgbm))

# Rücktransformation der Vorhersagen
y_train_pred_lgbm_original = y_scaler.inverse_transform(y_train_pred_lgbm.reshape(-1, 1)).flatten()
y_val_pred_lgbm_original = y_scaler.inverse_transform(y_val_pred_lgbm.reshape(-1, 1)).flatten()

train_rmse_lgbm_original = rmse(y_train, y_train_pred_lgbm_original)
val_rmse_lgbm_original = rmse(y_val, y_val_pred_lgbm_original)

print(f"LightGBM - Train RMSE: {train_rmse_lgbm_original:.4f}")
print(f"LightGBM - Validation RMSE: {val_rmse_lgbm_original:.4f}\n")

# --- CatBoost ---
cat_model = CatBoostRegressor(iterations=100, learning_rate=0.1, random_state=42, verbose=0)
cat_model.fit(X_train_scaled, y_train_scaled)

# Vorhersagen
y_train_pred_cat = cat_model.predict(X_train_scaled)
y_val_pred_cat = cat_model.predict(X_val_scaled)

# RMSE berechnen
train_rmse_cat = np.sqrt(mean_squared_error(y_train_scaled, y_train_pred_cat))
val_rmse_cat = np.sqrt(mean_squared_error(y_val_scaled, y_val_pred_cat))

# Rücktransformation der Vorhersagen
y_train_pred_cat_original = y_scaler.inverse_transform(y_train_pred_cat.reshape(-1, 1)).flatten()
y_val_pred_cat_original = y_scaler.inverse_transform(y_val_pred_cat.reshape(-1, 1)).flatten()

train_rmse_cat_original = rmse(y_train, y_train_pred_cat_original)
val_rmse_cat_original = rmse(y_val, y_val_pred_cat_original)

print(f"CatBoost - Train RMSE: {train_rmse_cat_original:.4f}")
print(f"CatBoost - Validation RMSE: {val_rmse_cat_original:.4f}\n")

# --- Neuronales Netz ---
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dense(64, activation='relu'),
    Dense(1)
])

# Modell kompilieren
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# --- Modell trainieren ---
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=100,
    batch_size=32,
    verbose=1
)

# --- Vorhersagen ---
y_train_pred_nn = y_scaler.inverse_transform(model.predict(X_train_scaled).flatten().reshape(-1, 1)).flatten()
y_val_pred_nn = y_scaler.inverse_transform(model.predict(X_val_scaled).flatten().reshape(-1, 1)).flatten()

# --- RMSE berechnen ---
train_rmse_nn = rmse(y_train, y_train_pred_nn)
val_rmse_nn = rmse(y_val, y_val_pred_nn)

print(f"Neural Network - Train RMSE: {train_rmse_nn:.4f}")
print(f"Neural Network - Validation RMSE: {val_rmse_nn:.4f}")

# --- Visualisierung der Trainings- und Validierungsverluste ---
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

