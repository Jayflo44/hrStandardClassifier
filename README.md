# HR Standards Classifier

An automated, multi-tier text classification and moderation system designed for Human Resources (HR) environments. Its goal is to analyze written communications to detect toxicity, harassment, offensive language, and sensitive data leaks.

## System Architecture

The project utilizes a "funnel" architecture optimized to save computational resources:

* **Tier 1 (Fast Filter - `TierOneBouncer`):** A high-speed, deterministic engine based on rules and dictionaries (lexicons). It evaluates text using optimized regular expressions to detect hate speech, severe profanity, obfuscated profanity, and sensitive data leaks (PII such as US Social Security Numbers or Credit Cards).
* **Tier 2 (Semantic Engine - `TierTwoSemanticEngine`):** If a message passes Tier 1, it undergoes deep contextual analysis using Deep Learning (the `unitary/toxic-bert` model via Hugging Face). It classifies toxicity based on configurable thresholds (OK, FLAGGED, or AMBIGUOUS).

## Requirements

The project requires Python 3.7+ and the following main dependencies:

* `transformers` (Hugging Face)
* `torch` (PyTorch)
* `PyYAML`

## Usage

The system can be used either as a module within another Python script or directly from the Command Line Interface (CLI) using the main orchestrator `pipeline.py`.

### Command Line Interface (CLI) Usage

You can classify a single live message:
```bash
python hrClassifierSrc/pipeline.py --message "Great job on the project, everyone contributed well."
