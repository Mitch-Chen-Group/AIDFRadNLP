# AIDFRadNLP
End-to-end pipeline for batch processing radiology reports using ChatGPT API

Tested on Windows
M Chen v1.0 Updated 2025.11.08

## Usage instruction

### 1. Git Clone repository
git clone 

### 2. Set up environment

```bash
conda create -n AIDFRadNLP python=3.10
conda activate AIDFRadNLP
pip install -r requirements.txt
```
### 3. Reports saved as reports.csv (1 line per report)
### 4. Update schema.json based on keywords and criteria
### 5. Set API key in rad_nlp_pipeline.py (Found on [https://platform.openai.com/account/api-keys])
### 6. Set model to use
### 7. Run pipeline using python rad_nlp_pipeline.py --input_csv reports.csv --out_jsonl gpt_out.jsonl --deid