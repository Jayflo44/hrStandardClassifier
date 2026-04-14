import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path


print("1. Cargando dataset...")

_ROOT = Path(__file__).resolve().parent
CSV_PATH = _ROOT / "data" / "processed" / "train_clean_validated.csv"
df = pd.read_csv(CSV_PATH)


# Crear etiqueta binaria (OK vs FLAGGED)
toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df['true_label'] = df[toxicity_cols].max(axis=1)

# Para probar rápido tu código la primera vez, tomaremos una muestra. 
# Cuando estés listo para el reporte final, borra o comenta esta línea:
# df = df.sample(20000, random_state=42) 

print("2. Separando datos de entrenamiento y prueba (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    df['comment_text'].astype(str), 
    df['true_label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['true_label'] # Mantiene la proporción de tóxicos en train y test
)

print("3. Construyendo el Extractor de Características TF-IDF...")
# Extrae palabras y bigramas de palabras
word_vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    max_features=10000, # Limita a las 10,000 palabras/pares más frecuentes
    stop_words='english'
)

# Extrae n-gramas de caracteres (3 a 5 letras)
char_vectorizer = TfidfVectorizer(
    analyzer='char_wb', # 'char_wb' crea n-gramas solo dentro de las palabras (ignora espacios)
    ngram_range=(3, 5),
    max_features=10000  # Limita a los 10,000 fragmentos de caracteres más frecuentes
)

# FeatureUnion combina las columnas de palabras y las de caracteres
combined_features = FeatureUnion([
    ("words", word_vectorizer),
    ("chars", char_vectorizer)
])

# 4. Configurar la Regresión Logística
# class_weight='balanced' es VITAL porque solo el ~10% de tu dataset es tóxico. 
# Esto le dice al modelo que preste más atención a la clase minoritaria.
log_reg = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)

# 5. Crear el Pipeline (Conecta la extracción con el entrenamiento)
pipeline = Pipeline([
    ("features", combined_features),
    ("classifier", log_reg)
])

print("4. Entrenando el modelo (Esto puede tomar unos minutos)...")
pipeline.fit(X_train, y_train)

print("5. Evaluando el modelo...")
y_pred = pipeline.predict(X_test)

print("\n==== MATRIZ DE CONFUSIÓN ====")
cm = confusion_matrix(y_test, y_pred)
print("             Predicción OK | Predicción FLAGGED")
print(f"Real OK      |    {cm[0][0]:,}   |      {cm[0][1]:,}")
print(f"Real FLAGGED |    {cm[1][0]:,}     |      {cm[1][1]:,}")

print("\n==== REPORTE DE CLASIFICACIÓN (F1-SCORE) ====")
print(classification_report(y_test, y_pred, target_names=['OK (0)', 'FLAGGED (1)']))