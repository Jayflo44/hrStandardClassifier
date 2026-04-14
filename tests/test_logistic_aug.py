"""Pipeline de Logistic Regression Aumentado (Stacking Ensemble) - CLOUD OPTIMIZED.

Cambios para la nube:
- Procesamiento del dataset completo (159k+ filas).
- Inferencia de Transformer optimizada con GPU (device=0).
- Procesamiento por lotes (batch_size=128) para máxima velocidad.
- Truncamiento explícito para evitar desbordamientos de memoria en textos anómalos.
"""

import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm
from transformers import pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = _ROOT / "hrStandardClassifier"/ "data" / "processed" / "train_clean_validated.csv"
REPORT_PATH = _ROOT / "artifacts" / "reports" / "logreg_augmented_cloud_report.json"

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
def main():
    t0 = time.perf_counter()
    print("=== LOGISTIC REGRESSION AUMENTADO (CLOUD ENSEMBLE) ===")
    
    # 1. CARGA DE DATOS
    print(f"\n[1] Cargando dataset completo: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['true_label'] = df[toxicity_cols].max(axis=1)
    
    df_sample = df.copy()
    print(f"  → Dataset listo: {len(df_sample):,} filas.")

    # 2. EXTRACCIÓN DE CARACTERÍSTICAS TF-IDF
    print("\n[2] Construyendo matriz TF-IDF (10,000 features)...")
    t_tfidf = time.perf_counter()
    
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=5000, stop_words='english')
    char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=5000)

    combined_features = FeatureUnion([
        ("words", word_vectorizer),
        ("chars", char_vectorizer)
    ])

    X_tfidf = combined_features.fit_transform(df_sample['comment_text'].astype(str))
    print(f"  → Forma TF-IDF: {X_tfidf.shape} (completado en {time.perf_counter() - t_tfidf:.2f}s)")

    # 3. EXTRACCIÓN SEMÁNTICA ACELERADA POR GPU (TRANSFORMER)
    print("\n[3] Extrayendo probabilidades semánticas en la Nube (GPU + Batching)...")
    t_trans = time.perf_counter()
    
    # device=0 usa la primera GPU disponible. batch_size acelera el proceso masivamente.
    # Si la nube no tiene GPU habilitada, cambia device=0 por device=-1 (solo CPU).
    semantic_classifier = pipeline(
        "text-classification", 
        model="unitary/toxic-bert", 
        top_k=None, 
        device=0, 
        batch_size=128 
    )

    text_list = df_sample['comment_text'].astype(str).tolist()
    transformer_probs = []
    
    # Al pasar la lista completa, el pipeline maneja los lotes automáticamente
    for raw_results in tqdm(semantic_classifier(text_list, truncation=True, max_length=512), total=len(text_list)):
        results = raw_results[0] if isinstance(raw_results[0], list) else raw_results
        
        prob = 0.0
        for item in results:
            if item['label'].lower() == 'toxic':
                prob = item['score']
                break
        transformer_probs.append(prob)
        
    X_transformer = csr_matrix(np.array(transformer_probs).reshape(-1, 1))
    print(f"  → Extracción completada en {time.perf_counter() - t_trans:.2f}s")

    # 4. STACKING (FEATURE AUGMENTATION)
    print("\n[4] Ensamblando matrices (Stacking)...")
    X_augmented = hstack([X_tfidf, X_transformer])
    print(f"  → Forma de la Matriz Final: {X_augmented.shape}")

    # 5. SPLIT Y ENTRENAMIENTO (META-MODELO)
    print("\n[5] Entrenando Regresión Logística...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_augmented, df_sample['true_label'], test_size=0.2, random_state=42, stratify=df_sample['true_label']
    )

    t_train = time.perf_counter()
    log_reg = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
    log_reg.fit(X_train, y_train)
    print(f"  → Modelo entrenado en {time.perf_counter() - t_train:.2f}s")

    # 6. EVALUACIÓN DE RENDIMIENTO
    y_pred = log_reg.predict(X_test)

    print("\n\n" + "=" * 60)
    print("REPORTE DE RESULTADOS — CLOUD ENSEMBLE (159k registros)")
    print("=" * 60)
    
    print("\n[Matriz de Confusión (Datos de Prueba - 20%)]")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"                Pred OK    Pred FLAGGED")
    print(f" Real OK         {tn:>8,}      {fp:>8,}")
    print(f" Real FLAGGED    {fn:>8,}      {tp:>8,}")
    
    report_text = classification_report(
        y_test, y_pred, 
        target_names=['OK (0)', 'FLAGGED (1)']
    )
    print(f"\n[Reporte de Clasificación]")
    print(report_text)

    elapsed = time.perf_counter() - t0

    # --- REPORTE JSON ---
    report_data = {
        "meta": {
            "script": "test_logreg_augmented_cloud.py",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "methodology": "stacking_ensemble_full_dataset",
            "model": "LogisticRegression"
        },
        "sample": {
            "total_extracted": len(df_sample),
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0]
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "metrics": {
            "precision": round(tp / max(tp + fp, 1), 6),
            "recall": round(tp / max(tp + fn, 1), 6),
            "f1_score": round(2 * tp / max(2 * tp + fp + fn, 1), 6)
        }
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\nReporte guardado en: {REPORT_PATH}")
    print(f"Tiempo total: {elapsed:.2f}s")

if __name__ == "__main__":
    main()