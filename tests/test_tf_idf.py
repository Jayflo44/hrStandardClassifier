"""Pipeline de Gemini — adaptado para probar contra tu proyecto.

Cambios vs original:
- Usa tu CSV preprocesado (train_clean_validated.csv) en vez de train.csv raw.
- Ruta corregida para tu estructura de proyecto.
- Genera reporte JSON para comparar con tus resultados.
- Mantiene la lógica original intacta (AMBIGUOUS = FLAGGED, muestra 1000).
"""

import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import re
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

_ROOT = Path(__file__).resolve().parent
CSV_PATH = _ROOT / "data" / "processed" / "train_clean_validated.csv"
REPORT_PATH = _ROOT / "data" / "processed" / "gemini_pipeline_report.json"

# ==========================================
# MÓDULO 1: TIER 1 - EL MOTOR DE REGLAS
# ==========================================
TIER_1_CONFIG = {
    'rules': [
        {
            'category': "Hate Speech and Slurs",
            'type': "lexicon",
            'boundary': True,
            'words': ["nigger", "nigga", "faggot", "fag", "spic", "chink", "kike", "tranny", "dyke", "wetback", "raghead", "retard", "halfbreed"]
        },
        {
            'category': "Severe Profanity (NSFW)",
            'type': "lexicon",
            'boundary': True,
            'words': ["fuck", "motherfucker", "cunt", "cocksucker", "dickhead", "whore", "slut", "bitch", "pussy", "bullshit", "cum"]
        },
        {
            'category': "Obfuscated Profanity",
            'type': "regex",
            'pattern': r"f[u\*\@\!]+ck|sh[i\*\@\!]+t|b[i\*\@\!]+tch"
        }
    ]
}

class Tier1Bouncer:
    def __init__(self, config):
        self.compiled_rules = []
        for rule in config['rules']:
            if rule['type'] == 'lexicon':
                joined_words = '|'.join(map(re.escape, rule['words']))
                pattern = f"\\b({joined_words})\\b" if rule.get('boundary', True) else f"({joined_words})"
                self.compiled_rules.append((rule['category'], re.compile(pattern, re.IGNORECASE)))
            elif rule['type'] == 'regex':
                self.compiled_rules.append((rule['category'], re.compile(rule['pattern'], re.IGNORECASE)))

    def inspect(self, text):
        text = str(text).lower()
        for category, regex in self.compiled_rules:
            if regex.search(text):
                return {"status": "FLAGGED", "reason": category}
        return {"status": "PASS"}

# ==========================================
# MÓDULO 2: TIER 2 - EL MOTOR SEMÁNTICO
# ==========================================
class Tier2SemanticEngine:
    def __init__(self):
        print("Cargando el Transformer Tier 2 (unitary/toxic-bert)...")
        self.classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
        self.THRESHOLD_SAFE = 0.20
        self.THRESHOLD_TOXIC = 0.80

    def analyze(self, text):
        raw_results = self.classifier(text[:512])
        results = raw_results[0] if isinstance(raw_results[0], list) else raw_results
            
        toxic_prob = 0.0
        for item in results:
            if item['label'].lower() == 'toxic':
                toxic_prob = item['score']
                break
        
        if toxic_prob > self.THRESHOLD_TOXIC:
            return {"status": "FLAGGED", "confidence": toxic_prob}
        elif toxic_prob < self.THRESHOLD_SAFE:
            return {"status": "OK", "confidence": toxic_prob}
        else:
            return {"status": "AMBIGUOUS", "confidence": toxic_prob}

# ==========================================
# MÓDULO 3: EJECUCIÓN
# ==========================================
def main():
    t0 = time.perf_counter()
    print("=== GEMINI PIPELINE — TEST COMPARATIVO ===")
    
    tier1 = Tier1Bouncer(TIER_1_CONFIG)
    tier2 = Tier2SemanticEngine()
    
    print(f"\nCargando dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  → {len(df):,} filas")

    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['true_label'] = df[toxicity_cols].max(axis=1)
    
    # Muestra balanceada de 1,000 (como en el original de Gemini)
    n_sample = 500
    df_toxic = df[df['true_label'] == 1].sample(n_sample, random_state=42)
    df_clean = df[df['true_label'] == 0].sample(n_sample, random_state=42)
    df_sample = pd.concat([df_toxic, df_clean]).sample(frac=1, random_state=42)
    
    print(f"\nMuestra balanceada: {len(df_sample):,} filas ({n_sample} tóxicas + {n_sample} limpias)")
    
    predicciones_finales = []
    detalles_captura = []

    print("\nEjecutando pipeline cascada (Tier 1 → Tier 2)...")
    t_pred = time.perf_counter()
    
    for index, row in tqdm(df_sample.iterrows(), total=df_sample.shape[0]):
        texto = str(row['comment_text'])
        
        t1_result = tier1.inspect(texto)
        
        if t1_result['status'] == 'FLAGGED':
            predicciones_finales.append(1)
            detalles_captura.append('Tier 1 (Reglas)')
            continue
            
        t2_result = tier2.analyze(texto)
        
        # AMBIGUOUS = FLAGGED (decisión de Gemini)
        if t2_result['status'] in ['FLAGGED', 'AMBIGUOUS']:
            predicciones_finales.append(1)
            detalles_captura.append('Tier 2 (Semántica)')
        else:
            predicciones_finales.append(0)
            detalles_captura.append('OK por Tier 2')

    pred_time = time.perf_counter() - t_pred

    df_sample['prediccion'] = predicciones_finales
    df_sample['atrapado_por'] = detalles_captura

    # --- Resultados ---
    print("\n\n" + "=" * 60)
    print("REPORTE DE RESULTADOS — GEMINI PIPELINE")
    print("=" * 60)
    
    print("\n[Distribución del Triaje]")
    triage_dist = df_sample['atrapado_por'].value_counts()
    print(triage_dist)
    
    print("\n[Matriz de Confusión]")
    cm = confusion_matrix(df_sample['true_label'], df_sample['prediccion'])
    tn, fp, fn, tp = cm.ravel()
    print(f"                 Pred OK    Pred FLAGGED")
    print(f"  Real OK       {tn:>8,}      {fp:>8,}")
    print(f"  Real FLAGGED  {fn:>8,}      {tp:>8,}")
    
    report_text = classification_report(
        df_sample['true_label'], df_sample['prediccion'],
        target_names=['OK (0)', 'FLAGGED (1)'],
    )
    print(f"\n[Reporte de Clasificación]")
    print(report_text)

    elapsed = time.perf_counter() - t0

    # --- Reporte JSON para comparar ---
    report_data = {
        "meta": {
            "script": "test_gemini_pipeline.py",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "prediction_seconds": round(pred_time, 3),
            "methodology": "balanced_sample_1000",
            "ambiguous_treated_as": "FLAGGED",
        },
        "sample": {
            "total": len(df_sample),
            "toxic": n_sample,
            "clean": n_sample,
            "balance": "50/50 (artificial)",
        },
        "triage_distribution": triage_dist.to_dict(),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "metrics": {
            "precision": round(tp / max(tp + fp, 1), 6),
            "recall": round(tp / max(tp + fn, 1), 6),
            "f1_score": round(2 * tp / max(2 * tp + fp + fn, 1), 6),
            "accuracy": round((tp + tn) / len(df_sample), 6),
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\nReporte JSON en: {REPORT_PATH}")
    print(f"Tiempo total: {elapsed:.2f}s")


if __name__ == "__main__":
    main()