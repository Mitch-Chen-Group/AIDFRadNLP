import os
import csv
import time
import json
import re
import asyncio
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from openai import AsyncOpenAI

# ---------------- CONFIG ----------------
MODEL_LLM = "gpt-5.1"
EMBEDDING_MODEL = "text-embedding-3-large"

MAX_CONCURRENCY = 10   # main speed control knob
EMBED_CONCURRENCY = 20

SAVE_EVERY = 25
MAX_INPUT_CHARS = 12000
TEMPERATURE = 0

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("rad_nlp")

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60)

# ---------------- UTIL ----------------
def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=["report_text"])
    df["report_id"] = df["report_id"].astype(str)
    return df

def deidentify(text: str) -> str:
    text = re.sub(r"\b([A-Z][a-z]+\s[A-Z][a-z]+)\b", "[NAME]", text)
    text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DATE]", text)
    return text

def safe_json(text: str):
    try:
        return json.loads(text)
    except:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except:
            return None

def build_prompt(schema, text):
    return f"""
You are a strict medical extraction system.

Schema:
{json.dumps(schema["fields"], indent=2)}

Report:
{text}

Return ONLY JSON.
"""

# ---------------- GPT CALL ----------------
async def call_gpt(semaphore, text, schema):
    async with semaphore:
        text = text[:MAX_INPUT_CHARS]
        prompt = build_prompt(schema, text)

        try:
            resp = await client.responses.create(
                model=MODEL_LLM,
                input=prompt,
                temperature=TEMPERATURE,
                max_output_tokens=600,
            )
            content = resp.output_text
        except Exception as e:
            logger.error(f"GPT error: {e}")
            return {k: None for k in schema["fields"]}

        parsed = safe_json(content)
        if not parsed:
            return {k: None for k in schema["fields"]}

        return parsed

# ---------------- EMBEDDINGS ----------------
async def get_embedding(semaphore, text):
    async with semaphore:
        try:
            r = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return r.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

# ---------------- PIPELINE ----------------
async def run_pipeline(input_csv, schema_path, out_csv, deid):

    df = read_csv(input_csv)
    schema = load_schema(schema_path)

    # resume
    if os.path.exists(out_csv):
        done = pd.read_csv(out_csv)["report_id"].astype(str).tolist()
        df = df[~df["report_id"].isin(done)]

    gpt_sem = asyncio.Semaphore(MAX_CONCURRENCY)
    emb_sem = asyncio.Semaphore(EMBED_CONCURRENCY)

    results = []

    # -------- GPT STAGE --------
    async def process_row(row):
        text = deidentify(row.report_text) if deid else row.report_text

        extraction = await call_gpt(gpt_sem, text, schema)

        return {
            "report_id": row.report_id,
            "report_text": text,
            "extraction": extraction
        }

    tasks = [process_row(row) for _, row in df.iterrows()]

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="GPT"):
        results.append(await f)

    # checkpoint save
    def append_csv(records):
        rows = []
        for r in records:
            flat = {
                "report_id": r["report_id"],
                "report_text": r["report_text"],
                **r["extraction"]
            }
            rows.append(flat)

        df_out = pd.DataFrame(rows)
        header = not os.path.exists(out_csv)
        df_out.to_csv(out_csv, mode="a", header=header, index=False, quoting=csv.QUOTE_ALL)

    append_csv(results)

    # -------- EMBEDDINGS --------
    embed_tasks = [
        get_embedding(emb_sem, r["report_text"])
        for r in results
    ]

    embeddings = []
    for f in tqdm(asyncio.as_completed(embed_tasks), total=len(embed_tasks), desc="Embedding"):
        embeddings.append(await f)

    embeddings = [e for e in embeddings if e is not None]

    # -------- ML (unchanged) --------
    labels = []
    for r in results:
        row = {}
        for k, v in r["extraction"].items():
            if v is None:
                row[k] = 0
            elif isinstance(v, bool):
                row[k] = int(v)
            elif isinstance(v, int):
                row[k] = v
        labels.append(row)

    if labels:
        Y = pd.DataFrame(labels).fillna(0).astype(int).clip(0, 1)

        if len(Y) > 1 and len(embeddings) == len(Y):
            X_tr, X_va, y_tr, y_va = train_test_split(
                np.array(embeddings),
                Y.to_numpy(),
                test_size=0.2,
                random_state=42
            )

            clf = OneVsRestClassifier(LogisticRegression(max_iter=2000))
            clf.fit(X_tr, y_tr)
            _ = clf.predict(X_va)

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--out_csv", default="gpt_output.csv")
    parser.add_argument("--deid", action="store_true")

    args = parser.parse_args()

    asyncio.run(run_pipeline(
        args.input_csv,
        args.schema,
        args.out_csv,
        args.deid
    ))
