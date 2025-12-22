# rad_nlp_pipeline.py
"""
End-to-end pipeline:
- load CSV of reports
- optional de-identification
- batch GPT-5 extraction (JSON outputs)
- compute embeddings & train classifier
- apply classifier to full dataset
- export results + simple QA metrics

Usage example: python rad_nlp_pipeline.py --input_csv reports.csv --schema schema.json -out_jsonl gpt_extractions.jsonl --deid
"""
import os
import csv
import time
import json
import re
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier

# OpenAI
from openai import OpenAI

# ---------- CONFIG ----------
MODEL_LLM = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 25
MAX_RETRIES = 5
TEMPERATURE = 0

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rad_nlp")

# ---------- OpenAI Client ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Utilities ----------
def read_reports_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize headers: lowercase and strip spaces
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"report_id", "report_text"} <= set(df.columns):
        raise ValueError("CSV must contain columns: 'report_id', 'report_text'")
    return df

def load_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt_from_schema(schema: Dict[str, Any], report_text: str) -> str:
    fields_json = json.dumps(schema["fields"], indent=2)
    instructions = "\n".join(schema.get("instructions", []))
    extra_notes = "\n".join(schema.get("extra_notes", []))
    return f"""
You are an information extraction system.

Task:
{schema.get("task_description", "")}

Instructions:
{instructions}

Schema (return EXACT JSON with these fields):
{fields_json}

Report:
-----
{report_text}
-----

Notes:
{extra_notes}

Return ONLY valid JSON. Do not add extra keys.
"""

# ---------- De-identification ----------
NAME_RE = re.compile(r"\b([A-Z][a-z]+\s[A-Z][a-z]+)\b")
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")

def deidentify_text(text: str) -> str:
    text = NAME_RE.sub("[NAME]", text)
    text = DATE_RE.sub("[DATE]", text)
    return text

# ---------- Safe JSON parse ----------
def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

# ---------- GPT ----------
@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(10, 30, 200))
def call_gpt(report_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_prompt_from_schema(schema, report_text)
    resp = client.chat.completions.create(
        model=MODEL_LLM,
        messages=[
            {"role": "system", "content": "You are a strict medical information extraction assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=600,
    )
    content = resp.choices[0].message.content
    parsed = safe_parse_json(content)
    if parsed is None:
        raise ValueError("Invalid JSON returned")
    return parsed

# ---------- Embeddings ----------
def get_embedding(text: str) -> List[float]:
    r = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return r.data[0].embedding

# ---------- Flatten record for CSV ----------
def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    flat = {
        "report_id": record["report_id"],
        "report_text": record["report_text"]
    }
    extraction = record.get("extraction", {})
    for k, v in extraction.items():
        if isinstance(v, str) and v.lower() in {"true", "false"}:
            v = v.lower() == "true"
        flat[k] = v
    return flat

def write_csv(out_path: str, records: List[Dict[str, Any]]):
    if not records:
        return
    rows = [flatten_record(r) for r in records]
    extraction_keys = set().union(*(r.keys() for r in rows)) - {"report_id", "report_text"}
    columns = ["report_id", "report_text"] + sorted(extraction_keys)
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)

# ---------- Pipeline ----------
def run_pipeline(input_csv: str, schema_path: str, out_csv: str, deid: bool):
    df = read_reports_csv(input_csv)
    schema = load_schema(schema_path)
    logger.info(f"Loaded {len(df)} reports")

    results = []

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="GPT batches"):
        batch = df.iloc[i:i+BATCH_SIZE]
        for _, row in batch.iterrows():
            text = deidentify_text(row.report_text) if deid else row.report_text
            try:
                extraction = call_gpt(text, schema)
            except Exception as e:
                logger.error(f"GPT failed for report {row.report_id}: {e}")
                extraction = {k: None for k in schema["fields"]}
            results.append({
                "report_id": row.report_id,
                "report_text": text,
                "extraction": extraction
            })
        time.sleep(1)

    write_csv(out_csv, results)
    logger.info(f"Saved extracted CSV to {out_csv}")

    # Embeddings & classifier
    texts = [r["report_text"] for r in results]
    embeddings = [get_embedding(t) for t in tqdm(texts, desc="Embedding")]
    labels = [flatten_record(r) for r in results]

    # Prepare label matrix
    y_labels = []
    for r in labels:
        record = {}
        for k, v in r.items():
            if k in {"report_id", "report_text"}:
                continue
            if v is None:
                record[k] = 0
            elif isinstance(v, bool):
                record[k] = int(v)
            elif isinstance(v, int):
                record[k] = v
        y_labels.append(record)

    if y_labels:
        Y = pd.DataFrame(y_labels)
        Y = Y.dropna(axis=1, how="all")
        if not Y.empty:
            Y = Y.fillna(0).astype(int)
            Y_array = Y.to_numpy(dtype=int)

            # Skip classifier if dataset too small
            if Y_array.shape[0] < 5:
                logger.warning(
                    f"Only {Y_array.shape[0]} samples available. Skipping classifier training."
                )
            else:
                X_tr, X_va, Y_tr, Y_va = train_test_split(
                    np.array(embeddings), Y_array, test_size=0.2, random_state=42
                )
                clf = OneVsRestClassifier(LogisticRegression(max_iter=2000))
                clf.fit(X_tr, Y_tr)
                preds = clf.predict(X_va)
                print(classification_report(Y_va, preds, zero_division=0))

    logger.info("Pipeline completed")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="CSV with report_id,report_text")
    parser.add_argument("--schema", required=True, help="Path to schema.json")
    parser.add_argument("--out_csv", default="gpt_extractions.csv", help="Output CSV path")
    parser.add_argument("--deid", action="store_true", help="De-identify text before extraction")
    args = parser.parse_args()

    run_pipeline(args.input_csv, args.schema, args.out_csv, args.deid)





    
