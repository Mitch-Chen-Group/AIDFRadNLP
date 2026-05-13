import os
import csv
import json
import re
import asyncio
import logging

import pandas as pd
from tqdm import tqdm

from openai import AsyncOpenAI

# ---------------- CONFIG ----------------
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

# ---------------- UTIL ----------------
def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_csv(path):
    df = pd.read_csv(path)

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # require report_text
    df = df.dropna(subset=["report_text"])

    # ensure string IDs
    df["report_id"] = df["report_id"].astype(str)

    return df

def deidentify(text: str) -> str:
    """
    Basic PHI masking.
    """

    # names
    text = re.sub(
        r"\b([A-Z][a-z]+\s[A-Z][a-z]+)\b",
        "[NAME]",
        text
    )

    # dates
    text = re.sub(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "[DATE]",
        text
    )

    return text

def safe_json(text: str):
    """
    Safely parse JSON from model output.
    """

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

Extract the requested fields from the radiology report.

Schema:
{json.dumps(schema["fields"], indent=2)}

Radiology Report:
{text}

Return ONLY valid JSON.
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

            return {
                k: None for k in schema["fields"]
            }

        parsed = safe_json(content)

        if not parsed:
            logger.error("Failed to parse JSON.")

            return {
                k: None for k in schema["fields"]
            }

        return parsed

# ---------------- SAVE ----------------
def append_csv(records, out_csv):

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

    df_out.to_csv(
        out_csv,
        mode="a",
        header=header,
        index=False,
        quoting=csv.QUOTE_ALL
    )

# ---------------- PIPELINE ----------------
async def run_pipeline(
    input_csv,
    schema_path,
    out_csv,
    deid
):

    # -------- LOAD DATA --------
    df = read_csv(input_csv)

    schema = load_schema(schema_path)

    # -------- RESUME SUPPORT --------
    if os.path.exists(out_csv):

        done_ids = pd.read_csv(
            out_csv
        )["report_id"].astype(str).tolist()

        df = df[
            ~df["report_id"].isin(done_ids)
        ]

    if len(df) == 0:
        print("No remaining reports to process.")
        return

    # -------- ASYNC LIMITER --------
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # -------- PROCESSING --------
    async def process_row(row):

        text = (
            deidentify(row.report_text)
            if deid
            else row.report_text
        )

        extraction = await call_gpt(
            semaphore,
            text,
            schema
        )

        return {
            "report_id": row.report_id,
            "report_text": text,
            "extraction": extraction
        }

    tasks = [
        process_row(row)
        for _, row in df.iterrows()
    ]

    results = []

    for f in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Extracting"
    ):
        results.append(await f)

    # -------- SAVE RESULTS --------
    append_csv(results, out_csv)

    # -------- DONE --------
    print("\nExtraction complete.")
    print(f"Processed reports: {len(results)}")
    print(f"Saved to: {out_csv}")

# ---------------- CLI ----------------
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to input CSV"
    )

    parser.add_argument(
        "--schema",
        required=True,
        help="Path to JSON schema"
    )

    parser.add_argument(
        "--out_csv",
        default="gpt_output.csv",
        help="Output CSV path"
    )

    parser.add_argument(
        "--deid",
        action="store_true",
        help="Enable de-identification"
    )

    args = parser.parse_args()

    asyncio.run(
        run_pipeline(
            args.input_csv,
            args.schema,
            args.out_csv,
            args.deid
        )
    )
