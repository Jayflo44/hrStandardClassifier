import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, hstack
from pathlib import Path

print("1. Cargando Tokenizador y Modelo en modo 'Feature Extraction'...")
# Usamos AutoModel (modelo base sin la capa de clasificación final)
tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
model = AutoModel.from_pretrained("unitary/toxic-bert")

# Mover a GPU si está disponible en la nube
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Modo evaluación (apaga el dropout para resultados consistentes)

_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = _ROOT / "hrStandardClassifier" / "data" / "processed" / "train_clean_validated.csv"

df = pd.read_csv(CSV_PATH).sample(1000, random_state=42) # Muestra para prueba

toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df['true_label'] = df[toxicity_cols].max(axis=1)

print("\n2. Extrayendo Embeddings Densos (768 dimensiones) por lotes...")
batch_size = 32
all_embeddings = []

for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df['comment_text'].iloc[i:i+batch_size].astype(str).tolist()
    
    # Tokenizamos el lote
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    
    with torch.no_grad(): # No calculamos gradientes (ahorra muchísima RAM)
        outputs = model(**inputs)
        
    # outputs.last_hidden_state tiene forma [batch_size, sequence_length, 768]
    # Extraemos solo el token [CLS] (el primer token de la secuencia, índice 0)
    # que actúa como el resumen semántico de toda la oración.
    cls_embeddings = outputs.last_hidden_state[:, 0, :] 
    
    # Lo pasamos de vuelta a la CPU y lo guardamos
    all_embeddings.append(cls_embeddings.cpu().numpy())

# Unimos todos los lotes en una sola matriz de NumPy (1000, 768)
X_dense_transformer = np.vstack(all_embeddings)
print(f"Forma de los Embeddings: {X_dense_transformer.shape}")

# Opcional (Recomendado): Unir con tu TF-IDF viejo
# X_tfidf = ... (tu código tfidf viejo que genera 10,000 columnas)
# X_final = hstack([X_tfidf, csr_matrix(X_dense_transformer)])

print("\n3. Entrenando Regresión Logística sobre los Embeddings...")
# Si usas SOLO embeddings, ya no es sparse, puedes usar solvers normales como 'lbfgs'
X_train, X_test, y_train, y_test = train_test_split(
    X_dense_transformer,
    df['true_label'],
    test_size=0.2,
    random_state=42,
    stratify=df['true_label']
)
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000)
log_reg.fit(X_train, y_train)

print("¡Modelo entrenado exitosamente con Embeddings!")