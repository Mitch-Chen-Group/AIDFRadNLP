# AIDFRadNLP
End-to-end pipeline for batch processing radiology reports using OpenAI API

Tested on Windows 11

Mitch Chen v1.0 

Updated 2025.11.08
License: CC-A-NC-SA 4.0

## Usage instructions

## Method 1: Step-wise pipeline 

### 1. Git Clone repository
git clone https://github.com/Mitch-Chen-Group/AIDFRadNLP.git

### 2. Set up environment

```bash
conda create -n AIDFRadNLP python=3.10
conda activate AIDFRadNLP
pip install -r requirements.txt
```
### 3. Save reports as reports.csv (1 line per report, template provided)
### 4. Update schema.json based on keywords and criteria
### 5. Set API key in rad_nlp_pipeline.py (Found on [https://platform.openai.com/account/api-keys])
### 6. Set model to use
### 7. Run pipeline 

```bash
python rad_nlp_pipeline.py --input_csv reports.csv --out_jsonl gpt_out.jsonl --deid
```

Output saved as final_extractions_with_preds.csv in root folder

## Method 2: API Server (Experimental)

### 1. Set up SQLite
```bash
python -c "from database import Base, engine; Base.metadata.create_all(bind=engine)"
```
### 2. Run API
```bash
uvicorn api.main:app --reload
```