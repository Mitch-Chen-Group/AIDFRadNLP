# rad_nlp_pipeline.py
"""
End-to-end pipeline:
- load CSV of reports
- optional de-identification
- batch GPT-5 extraction (JSON outputs)
- compute embeddings & train classifier
- apply classifier to full dataset
- export results + simple QA metrics
"""

import os
import csv
import time
import json
import re
import math
import random
import logging
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

# OpenAI SDK (use the OpenAI Python SDK style: `from openai import OpenAI`)
# If your environment has a different client, adapt the calls below.
from openai import OpenAI

# ---------- Configuration ----------
from openai import OpenAI

# Replace with your actual key
api_key = "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"
client = OpenAI(api_key=api_key)

# Test a simple call
response = client.models.list()  # lists available models
print(response)

#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#if not OPENAI_API_KEY:
#    raise RuntimeError("Set OPENAI_API_KEY environment variable with your OpenAI key.")

#client = OpenAI(api_key=OPENAI_API_KEY)  # adapt to your SDK instantiation if needed

MODEL_LLM = "gpt-4o"  # replace if your org uses a different alias
EMBEDDING_MODEL = "text-embedding-3-small"

BATCH_SIZE = 25              # adjust to respect rate limits and concurrency
MAX_RETRIES = 5
TEMPERATURE = 0

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rad_nlp")

# ---------- Utilities ----------
def read_reports_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "report_id" in df.columns and "report_text" in df.columns
    return df

def write_jsonl(out_path: str, records: List[Dict[str,Any]]):
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path: str) -> List[Dict[str,Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

# ---------- Simple de-identification (example, NOT a legal deid) ----------
NAME_RE = re.compile(r"\b([A-Z][a-z]{1,20}\s[A-Z][a-z]{1,20})\b")  # naive
MRN_RE = re.compile(r"\b(MRN|mrn|Patient ID|ID)\s*[:#]?\s*\d+\b")
DATE_RE = re.compile(r"\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](\d{2,4})\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4}\b", flags=re.I)

def deidentify_text(text: str) -> str:
    # WARNING: This is a simple heuristic scrub for demo purposes.
    # For production, use an expert-determined deid pipeline or vendor tool.
    t = NAME_RE.sub("[NAME]", text)
    t = MRN_RE.sub("[ID]", t)
    t = DATE_RE.sub("[DATE]", t)
    # remove email/phone patterns
    t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", t)
    t = re.sub(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "[PHONE]", t)
    return t

# ---------- GPT extraction utilities ----------
PROMPT_TEMPLATE = """You are a radiology report information extraction system.
Extract only the requested fields from the report text. Do NOT add extra fields or invent facts.
Return ONLY valid JSON that matches the schema exactly.

Schema:
{{
  "nodule_present": "true" or "false",
  "largest_nodule_size_mm": number or null,
  "malignancy_suspected": "true" or "false" or "uncertain",
  "follow_up_recommended": "true" or "false",
  "evidence_snippet": "string (exact text supporting the finding)"
}}

Report:
-----
{report_text}
-----

Notes:
- If the report has multiple nodules and lists sizes, return the largest numeric size in mm (if size in cm, convert to mm).
- If size is described as 'subcentimeter' or 'tiny' but no numeric, return null.
- For follow_up_recommended, look for explicit recommendations like 'follow-up CT', 'recommend repeat CT', 'interval CT', 'surveillance'.
- If nothing is mentioned, use 'false' for booleans and null for numeric size.
- Provide the evidence_snippet as a short exact excerpt from the report that justifies the values.
"""

# safe json fix helper
def safe_parse_json(text: str) -> Optional[Dict[str,Any]]:
    # try direct parse
    try:
        return json.loads(text)
    except Exception:
        # try to extract the first {...} substring
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                # try some common fixes:
                t = m.group(0)
                t = t.replace("\n", " ")
                # remove trailing commas
                t = re.sub(r",\s*}", "}", t)
                t = re.sub(r",\s*]", "]", t)
                try:
                    return json.loads(t)
                except Exception:
                    return None
        return None

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=20, max=200))
def call_gpt_for_report(report_text: str) -> Dict[str,Any]:
    prompt = PROMPT_TEMPLATE.format(report_text=report_text)
    # We use the chat completions endpoint style.
    resp = client.chat.completions.create(
        model=MODEL_LLM,
        messages=[{"role":"system", "content": "You are a strict information extraction assistant."},
                  {"role":"user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=400,
    )
    # the SDK may put content in different path; here we follow response structure:
    content = resp.choices[0].message["content"] if hasattr(resp.choices[0].message, "__getitem__") else resp.choices[0].message.content
    parsed = safe_parse_json(content)
    if parsed is None:
        raise ValueError("Failed to parse JSON from model output. Raw output: " + content[:1000])
    return parsed

def convert_size_to_mm(size_text: str) -> Optional[float]:
    # helper to parse sizes like '0.8 cm', '8 mm', '1.2 x 0.8 cm', 'subcentimeter'
    if not size_text: return None
    s = size_text.lower()
    if "subcent" in s or "sub-cent" in s or "tiny" in s:
        return None
    # find numbers
    nums = re.findall(r"(\d+\.?\d*)\s*(mm|cm)?", s)
    if not nums:
        return None
    vals = []
    for num, unit in nums:
        v = float(num)
        if unit == "cm":
            v = v * 10.0
        vals.append(v)
    if vals:
        return max(vals)
    return None

# ---------- Embeddings & classifier ----------
def get_embedding(text: str) -> List[float]:
    # call embeddings endpoint
    r = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return r.data[0].embedding

# Example: map JSON fields to classifier labels (multi-label)
def record_to_labels(json_record: Dict[str,Any]) -> Dict[str,int]:
    return {
        "nodule_present": 1 if str(json_record.get("nodule_present","false")).lower()=="true" else 0,
        "malignancy_suspected_true": 1 if str(json_record.get("malignancy_suspected","false")).lower()=="true" else 0,
        "malignancy_suspected_uncertain": 1 if str(json_record.get("malignancy_suspected","false")).lower()=="uncertain" else 0,
        "follow_up": 1 if str(json_record.get("follow_up_recommended","false")).lower()=="true" else 0
    }

# Train a logistic regression on embeddings
def train_classifier(X_embeds: List[List[float]], y_labels: List[Dict[str,int]]):
    Y = pd.DataFrame(y_labels)
    X = np.array(X_embeds)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_val)
    print("Validation classification report:")
    print(classification_report(Y_val, preds, zero_division=0))
    return clf

# ---------- Orchestration ----------
def run_gpt_batch(df: pd.DataFrame, out_jsonl: str, deid: bool = True):
    results = []
    # iterate in batches
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="GPT batches"):
        batch = df.iloc[i:i+BATCH_SIZE]
        for _, row in batch.iterrows():
            rid = row["report_id"]
            text = row["report_text"]
            if deid:
                text_in = deidentify_text(text)
            else:
                text_in = text
            try:
                parsed = call_gpt_for_report(text_in)
            except Exception as e:
                logger.exception(f"GPT failed for report {rid}: {e}")
                parsed = {
                    "nodule_present": "false",
                    "largest_nodule_size_mm": None,
                    "malignancy_suspected": "false",
                    "follow_up_recommended": "false",
                    "evidence_snippet": ""
                }
            # normalize size: sometimes model returns string with units
            size = parsed.get("largest_nodule_size_mm")
            if isinstance(size, str):
                maybe_size = convert_size_to_mm(size)
                parsed["largest_nodule_size_mm"] = maybe_size
            results.append({
                "report_id": rid,
                "report_text": text if not deid else text_in,
                "extraction": parsed
            })
        # small sleep to avoid bursts; tune for your rate limits
        time.sleep(1.0)
    write_jsonl(out_jsonl, results)
    return results

def build_embeddings_for_dataset(df: pd.DataFrame, text_col="report_text", sample_limit=None):
    embeddings = []
    for i, text in tqdm(enumerate(df[text_col]), total=(len(df) if sample_limit is None else min(len(df), sample_limit)), desc="Embedding"):
        if sample_limit and i >= sample_limit:
            break
        emb = get_embedding(text)
        embeddings.append(emb)
        # small sleep if needed
        time.sleep(0.01)
    return embeddings

def main_pipeline(input_csv: str, out_jsonl: str, do_deid=True, label_csv: Optional[str]=None):
    df = read_reports_csv(input_csv)
    logger.info(f"Loaded {len(df)} reports")

    # 1) Run GPT extraction pass (this produces 'silver' labels)
    logger.info("Running GPT extraction pass...")
    gpt_results = run_gpt_batch(df, out_jsonl, deid=do_deid)

    # 2) Build embeddings for the same reports
    logger.info("Computing embeddings for classifier training...")
    texts = [r["report_text"] for r in gpt_results]
    embeds = []
    for t in tqdm(texts, desc="Embedding all"):
        embeds.append(get_embedding(t))
        time.sleep(0.01)
    # Map GPT outputs -> labels (silver)
    labels = [record_to_labels(r["extraction"]) for r in gpt_results]

    # 3) Train classifier on silver labels (or on human-labeled data if provided)
    logger.info("Training classifier on embeddings (silver labels)...")
    clf = train_classifier(embeds, labels)

    # 4) Apply classifier to all (example: we just re-use the trained clf on embeddings)
    # For demo we reuse the same embeddings computed above.
    X = np.array(embeds)
    preds = clf.predict(X)
    preds_df = pd.DataFrame(preds, columns=["nodule_present_pred", "malignancy_true_pred", "malignancy_uncertain_pred", "follow_up_pred"])
    out_df = pd.DataFrame([{
        "report_id": r["report_id"],
        "report_text": r["report_text"],
        "gpt_extraction": r["extraction"]
    } for r in gpt_results])
    out_df2 = pd.concat([out_df.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)
    out_df2.to_csv("final_extractions_with_preds.csv", index=False)
    logger.info("Exported final_extractions_with_preds.csv")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True, help="CSV with report_id,report_text")
    p.add_argument("--out_jsonl", default="gpt_extractions.jsonl")
    p.add_argument("--deid", action="store_true")
    args = p.parse_args()
    main_pipeline(args.input_csv, args.out_jsonl, do_deid=args.deid)
