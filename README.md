# Customer Churn Prediction — Assessment Bundle

This project follows the supplied Data Scientist Technical Assessment and preserves the core logic of the provided `app.py`.

## Contents
- `app.py` — Streamlit application: churn dashboard, instant prediction, model comparison, simple TF-IDF retrieval Q&A, and test-plan page.
- `customer_churn_analysis.ipynb` — reproducible notebook for data inspection, preprocessing, model training, evaluation, and saving outputs.
- `architecture_diagram.png` — system architecture diagram.
- `technical_report.pdf` — concise technical report (design, data, preprocessing, models, evaluation plan, limitations).
- `test_case_report.csv` — ten required retrieval/test scenarios with expected behavior and current status.
- `dataset_metadata.md` — public source URL, collection notes, license-check reminder, and data dictionary summary.
- `run_evaluation.py` — standalone script to generate model metrics, confusion-matrix images, and a comparison CSV.
- `declaration_template.md` — fill in truthfully before submission.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install packages:
   `pip install -r requirements.txt`
5. Start the app:
   `streamlit run app.py`

The app reads the CSV from the public URL in `app.py`, so internet access is required at first load. If that URL is unavailable, download the dataset from the source and update `DATA_URL` to a local CSV path.

## Generate evaluation artifacts
Run:
`python run_evaluation.py`
This downloads/reads the dataset using the same source URL, cleans `TotalCharges`, removes incomplete/duplicate rows, trains three models with a fixed 80/20 stratified split (random_state=42), and writes `model_comparison.csv`, `confusion_matrix_*.png`, and `classification_report_*.json`.

## Important accuracy and assessment notes
- No model scores are hard-coded in this bundle. Run the app or evaluation script to calculate them in your environment.
- The Streamlit Q&A page is a small TF-IDF/cosine-similarity retrieval demo over three built-in notes. It is **not** a generative RAG/LLM system and does not yet pass all ten edge cases.
- `test_case_report.csv` records expected behavior and marks tests as “Not executed” until you run and document actual outputs.
- Review the dataset license/usage terms yourself and record the date you collect it.
- Read the code and report and be ready to explain them in your own words. Complete the declaration only if it truthfully reflects your work and the assessment rules.
