"""Score one new backstory against a downloaded novel."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backstory_checker.config import BOOKS, EMBED_MODEL, MAX_LENGTH, MODEL_DIR
from backstory_checker.download import ensure_novels
from backstory_checker.index import load_or_build
from backstory_checker.retrieve import model_evidence, top_passages


def predict(book_name: str, backstory: str) -> dict:
    if book_name not in BOOKS:
        known = ", ".join(BOOKS)
        raise ValueError(f"Unknown book {book_name!r}. Known books: {known}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError("No trained model yet. Run python run_pipeline.py first.")

    novel_paths = ensure_novels()
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    index = load_or_build(book_name, novel_paths[book_name], embedder)
    passages, scores = top_passages(backstory.strip(), index, embedder)
    evidence = model_evidence(passages)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    encoded = tokenizer(
        backstory.strip(),
        evidence,
        truncation="only_second",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    label = model.config.id2label[pred_id]
    if isinstance(label, str) and label.isdigit():
        label = {0: "contradict", 1: "consistent"}[pred_id]
    return {
        "label": str(label),
        "prob_contradict": float(probs[0]),
        "prob_consistent": float(probs[1]),
        "passages": passages,
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check one backstory passage against a novel.")
    parser.add_argument("--book", required=True, choices=list(BOOKS))
    parser.add_argument("--text", required=True, help="Backstory passage")
    args = parser.parse_args()
    result = predict(args.book, args.text)
    print(result["label"])
    print(f"contradict {result['prob_contradict']:.3f} | consistent {result['prob_consistent']:.3f}")
    for rank, (passage, score) in enumerate(zip(result["passages"], result["scores"]), start=1):
        preview = passage[:220].replace("\n", " ")
        print(f"\n[{rank}] similarity {score:.3f}\n{preview}")


if __name__ == "__main__":
    main()
