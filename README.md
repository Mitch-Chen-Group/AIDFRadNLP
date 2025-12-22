# AIDFRadNLP
End-to-end pipeline for batch processing radiology reports using OpenAI API

Tested on Windows 11, Ubuntu 20.04 and OSX

Mitch Chen v1.1

Updated 2025.12.22
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
### 4. Update schema.json based on keywords, output type (boolean if true or false) and criteria 
### 5. Set API key export OPENAI_API_KEY= (Found on [https://platform.openai.com/account/api-keys]); check  echo $OPENAI_API_KEY
### 6. Set model to use in rad_nlp_pipeline.py (MODEL_LLM = "gpt-4o")
### 7. Run pipeline 

```bash
python rad_nlp_pipeline.py --input_csv reports.csv --schema schema.json --out_csv gpt_output.csv  --deid
```

Output saved as gpt_output.csv in root folder. Open with Excel with "delimited" enabled 

## Method 2: API Server (Experimental)

### 1. Set up SQLite
```bash
python -c "from database import Base, engine; Base.metadata.create_all(bind=engine)"
```
### 2. Run API
```bash
uvicorn api.main:app --reload
```
