import os
import csv
import json
import re
import asyncio
import logging
import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI

MODEL_LLM = "gpt-5.1"
MAX_CONCURRENCY = 10
MAX_INPUT_CHARS = 12000
TEMPERATURE = 0

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("rad_nlp")

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=60
)

def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "report_text" not in df.columns:
        raise ValueError("Missing report_text column")

    if "report_id" not in df.columns:
        raise ValueError("Missing report_id column")

    if "Accession" not in df.columns:
        raise ValueError("Missing Accession column")

    df = df.dropna(subset=["report_text"])
    df["report_id"] = df["report_id"].astype(str)
    df["Accession"] = df["Accession"].astype(str)

    return df

def deidentify(text):
    text = re.sub(r"\b([A-Z][a-z]+\s[A-Z][a-z]+)\b", "[NAME]", text)
    text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DATE]", text)
    return text

def safe_json(text):
    try:
        return json.loads(text)
    except:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                return None
        return None

def build_prompt(schema, text):
    return f"""
You are a strict medical extraction system.

Extract the requested fields from the radiology report.

Schema:
{json.dumps(schema["fields"], indent=2)}

Radiology Report:
{text}

Return ONLY valid JSON.
"""

async def call_gpt(semaphore, text, schema):
    async with semaphore:
        try:
            resp = await client.responses.create(
                model=MODEL_LLM,
                input=build_prompt(schema, text[:MAX_INPUT_CHARS]),
                temperature=TEMPERATURE,
                max_output_tokens=600
            )

            parsed = safe_json(resp.output_text)

            if parsed:
                return parsed

        except Exception as e:
            logger.error(e)

        return {k: None for k in schema["fields"]}

def append_csv(records, out_csv):
    rows = []

    for r in records:
        rows.append({
            "report_id": r["report_id"],
            "Accession": r["Accession"],
            "report_text": r["report_text"],
            **r["extraction"]
        })

    pd.DataFrame(rows).to_csv(
        out_csv,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\"
    )

async def run_pipeline(input_csv, schema_path, out_csv, deid):
    if os.path.exists(out_csv):
        os.remove(out_csv)

    df = read_csv(input_csv)
    schema = load_schema(schema_path)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def process_row(row):
        text = deidentify(row.report_text) if deid else row.report_text

        return {
            "report_id": row.report_id,
            "Accession": row.Accession,
            "report_text": text,
            "extraction": await call_gpt(semaphore, text, schema)
        }

    tasks = [process_row(row) for _, row in df.iterrows()]
    results = []

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting"):
        results.append(await task)

    append_csv(results, out_csv)

    print("Extraction complete.")
    print(f"Processed reports: {len(results)}")
    print(f"Saved to: {out_csv}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--out_csv", default="gpt_output.csv")
    parser.add_argument("--deid", action="store_true")

    args = parser.parse_args()

    asyncio.run(
        run_pipeline(
            args.input_csv,
            args.schema,
            args.out_csv,
            args.deid
        )
    )
