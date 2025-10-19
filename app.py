import os
import re
import json
import pickle
from pathlib import Path
from datetime import datetime
import threading
import time

import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr
import dateparser
from fastapi import FastAPI
import uvicorn


DATA_URL = "https://raw.githubusercontent.com/lmurayire12/DOMAIN-SPECIFIC-CHATBOT/refs/heads/main/data/personal_transactions%20new.csv"
MODEL_DIR = Path("./saved_models/fine_tuned_t5")  # Optional local fallback
FAISS_INDEX_PATH = "./saved_models/transactions.faiss"
FAISS_META_PATH = "./saved_models/faiss_meta.pkl"

# Preferred Hugging Face model IDs (override via env)
HF_T5_MODEL_ID = os.environ.get("T5_MODEL_ID") or os.environ.get("HF_T5_MODEL_ID") or "google/flan-t5-small"
EMB_MODEL = os.environ.get("SENTENCE_EMB_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")

# Optional shared cache dir (HF Spaces sets HF_HOME)
HF_CACHE_DIR = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")

MAX_LENGTH = 256


print("Loading BudgetBuddy chatbot...", flush=True)


@tf.function(reduce_retracing=True)
def generate_text(input_ids, attention_mask, model):
    return model.generate(input_ids, attention_mask=attention_mask, max_length=128)


def load_model():
    """Load seq2seq model from Hugging Face Hub if possible, else local, else base."""
    # 1) Try Hub (supports large/fine-tuned models)
    try:
        print(f"🔎 Loading model from Hugging Face: {HF_T5_MODEL_ID}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(HF_T5_MODEL_ID, cache_dir=HF_CACHE_DIR)
        # Many checkpoints store PyTorch weights; allow conversion to TF with from_pt=True
        model = TFAutoModelForSeq2SeqLM.from_pretrained(HF_T5_MODEL_ID, from_pt=True, cache_dir=HF_CACHE_DIR)
        print("✓ Loaded model from Hugging Face Hub", flush=True)
        return tokenizer, model
    except Exception as e:
        print(f"⚠ Hub load failed: {e}", flush=True)

    # 2) Try local fine-tuned directory (if present)
    model_path = Path(MODEL_DIR)
    if model_path.exists():
        try:
            print("🔎 Loading model from local directory", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
            try:
                model = TFAutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
            except Exception:
                # If only PyTorch weights present locally
                model = TFAutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR), from_pt=True)
            print("✓ Loaded model from local directory", flush=True)
            return tokenizer, model
        except Exception as e:
            print(f"⚠ Local load failed: {e}", flush=True)

    # 3) Final fallback to a small public checkpoint
    print("⚠ Falling back to base model: google/flan-t5-small", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", cache_dir=HF_CACHE_DIR)
    model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", from_pt=True, cache_dir=HF_CACHE_DIR)
    return tokenizer, model


def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    if "Description" in df.columns:
        df["Description_clean"] = df["Description"].astype(str).str.lower()
        df["Description_clean"] = df["Description_clean"].str.replace("[^a-z0-9 ]", " ", regex=True)
        df["Description_clean"] = df["Description_clean"].str.replace(r"\s+", " ", regex=True).str.strip()

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    return df


def load_faiss():
    if not Path(FAISS_INDEX_PATH).exists():
        print("⚠ FAISS index not found. Building from CSV data...", flush=True)
        return build_faiss_from_data()

    print("✓ Loading FAISS index from local directory", flush=True)
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "rb") as f:
        meta = pickle.load(f)

    sbert = SentenceTransformer(EMB_MODEL)
    return index, meta["texts"], meta["metas"], sbert


def build_faiss_from_data():
    print("Building FAISS index from transaction data...", flush=True)
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    sbert = SentenceTransformer(EMB_MODEL)
    texts = []
    metas = []
    
    for idx, row in df.iterrows():
        date_str = str(row.get('Date', ''))
        amount = row.get('Amount', 0)
        category = row.get('Category', 'Unknown')
        description = str(row.get('Description', ''))
        
        txt = f"DATE: {date_str} AMOUNT: {amount:.2f} CATEGORY: {category} DESCRIPTION: {description}"
        texts.append(txt)
        metas.append({
            'idx': int(idx),
            'date': date_str,
            'amount': float(amount),
            'category': category
        })
    
    embeddings = sbert.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    print(f"✓ Built FAISS index with {len(texts)} transactions", flush=True)
    return index, texts, metas, sbert


# -----------------------------
# Lazy initialization globals
# -----------------------------
tokenizer_global = None
model_global = None
df_global = None
index_global = None
texts_global = None
metas_global = None
sbert_global = None
APP_READY = False


def init_all_heavy_components():
    """Load models, data, and FAISS in the background to avoid startup timeouts."""
    global tokenizer_global, model_global, df_global
    global index_global, texts_global, metas_global, sbert_global, APP_READY

    try:
        print("[init] Loading model...", flush=True)
        tokenizer_global, model_global = load_model()
        print("[init] Model loaded", flush=True)

        print("[init] Loading data...", flush=True)
        df_global = load_data()
        print(f"[init] Data loaded ({len(df_global)} transactions)", flush=True)

        print("[init] Loading/Building FAISS...", flush=True)
        index_global, texts_global, metas_global, sbert_global = load_faiss()
        print("[init] FAISS ready", flush=True)

        APP_READY = True
        print("[init] App is READY ✅", flush=True)
    except Exception as e:
        # Do not crash the server; keep running so port stays open
        print(f"[init] Initialization error: {e}", flush=True)


def is_finance_related(question):
    finance_keywords = [
        "spend", "spent", "budget", "money", "cost", "price", "transaction", 
        "payment", "expense", "income", "category", "total", "amount", "balance",
        "credit", "debit", "purchase", "bought", "paid", "financial", "dollar"
    ]
    return any(keyword in question.lower() for keyword in finance_keywords)


def retrieve_similar(query, index, texts, metas, sbert, k=4):
    q_emb = sbert.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)

    results = []
    for score, i in zip(D[0], I[0]):
        results.append({
            "score": float(score),
            "text": texts[i],
            "meta": metas[i]
        })
    return results


def generate_answer(question, context, tokenizer, model):
    input_text = f"{question} </s> CONTEXT: {context}"
    inputs = tokenizer(input_text, return_tensors="tf", truncation=True, 
                      padding="max_length", max_length=MAX_LENGTH)

    outputs = generate_text(inputs["input_ids"], inputs["attention_mask"], model)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


def parse_time_query(text):
    parsed = dateparser.parse(text, settings={"PREFER_DATES_FROM": "past"})
    if parsed:
        return parsed.year, parsed.month

    year_match = re.search(r"(20\d{2})", text)
    if year_match:
        return int(year_match.group(1)), None

    return None, None


def compute_spending(df, category=None, year=None, month=None):
    filtered = df.copy()

    if category:
        filtered = filtered[filtered["Category"].str.lower() == category.lower()]
    if year:
        filtered = filtered[filtered["Year"] == int(year)]
    if month:
        filtered = filtered[filtered["Month"] == int(month)]

    total = filtered["Amount"].sum()
    return round(float(total), 2), filtered


def is_aggregate_query(text):
    patterns = [r"\bhow much\b", r"\btotal\b", r"\bsum\b", r"\bspent on\b", r"\bspent\b"]
    return any(re.search(pattern, text.lower()) for pattern in patterns)


def answer_question(question, df, index, texts, metas, sbert, tokenizer, model):
    if not is_finance_related(question):
        return "I\'m a finance chatbot. I can help with questions about spending, budgets, and transactions."

    if is_aggregate_query(question):
        categories = df["Category"].dropna().unique().tolist()
        found_category = None

        for cat in categories:
            if cat.lower() in question.lower():
                found_category = cat
                break

        year, month = parse_time_query(question)
        total, subset = compute_spending(df, category=found_category, year=year, month=month)

        if subset.empty:
            retrieved = retrieve_similar(question, index, texts, metas, sbert)
            context = "\n".join([r["text"] for r in retrieved])
            return generate_answer(question, context, tokenizer, model)

        response = f"You spent ${total:.2f}"

        if found_category:
            response += f" on {found_category}"
            if not subset["Description_clean"].empty:
                top_item = subset["Description_clean"].value_counts().idxmax()
                response += f", mostly on {top_item}"

        if year and month:
            month_name = datetime(int(year), int(month), 1).strftime("%B")
            response += f" in {month_name} {year}"
        elif year:
            response += f" in {year}"

        response += "."
        return response

    else:
        retrieved = retrieve_similar(question, index, texts, metas, sbert)
        context = "\n".join([r["text"] for r in retrieved])
        return generate_answer(question, context, tokenizer, model)


def create_interface():
    def chat(question):
        if not question.strip():
            return "Please ask a question about your finances."

        try:
            # Use lazy-loaded globals
            if not APP_READY:
                return "🔄 The app is starting up and loading models (~30-60s on first run). Please try again shortly."

            answer = answer_question(
                question,
                df_global,
                index_global,
                texts_global,
                metas_global,
                sbert_global,
                tokenizer_global,
                model_global,
            )
            return answer
        except Exception as e:
            return f"Error processing your question. Please try rephrasing it."

    interface = gr.Interface(
        fn=chat,
        inputs=gr.Textbox(
            label="Your Question",
            placeholder="Example: How much did I spend on Entertainment in May 2018?",
            lines=2
        ),
        outputs=gr.Textbox(label="Answer", lines=3),
        title="BudgetBuddy - Personal Finance Assistant",
        description="Ask questions about your spending patterns, transactions, and budget insights.",
        examples=[
            ["How much did I spend on Entertainment in May 2018?"],
            ["What are my top spending categories?"],
            ["Why did my spending increase last month?"],
            ["Show me transactions from January 2018"]
        ],
        theme=gr.themes.Soft(),
        css="""
            .gradio-container {max-width: 800px !important}
            #component-0 {text-align: center}
        """
    )

    return interface


def main():
    print("Initializing web server (FastAPI + Gradio mount, lazy loading enabled)...", flush=True)

    # Bind to Render-provided port immediately
    port = int(os.environ.get("PORT", 7860))
    print(f"🔧 Will bind to 0.0.0.0:{port}", flush=True)

    # FastAPI app
    api = FastAPI()

    @api.get("/health")
    def health():
        return {"status": "ok", "ready": APP_READY}

    # Create UI without heavy deps; use globals inside chat()
    gradio_app = create_interface()
    # Mount Gradio at root
    app_mounted = gr.mount_gradio_app(api, gradio_app, path="/")

    # Start background initialization so the port opens quickly
    threading.Thread(target=init_all_heavy_components, daemon=True).start()

    print(f"\n🚀 Starting BudgetBuddy on port {port}...\n", flush=True)
    uvicorn.run(app_mounted, host="0.0.0.0", port=port, log_level="info")
    print(f"\n✅ BudgetBuddy is now running on http://0.0.0.0:{port}\n", flush=True)


if __name__ == "__main__":
    main()
