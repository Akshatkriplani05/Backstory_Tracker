"""Fine-tune DeBERTa on backstory passages plus retrieved evidence."""

from __future__ import annotations

import csv
import gc
import inspect
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from backstory_checker.config import (
    BATCH_SIZE,
    CLASSIFIER_MODEL,
    EMBED_MODEL,
    EPOCHS,
    ID2LABEL,
    LABEL2ID,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_DIR,
    MODEL_PASSAGE_WORDS,
    OUTPUT_DIR,
    SEED,
    TEST_SIZE,
    TOP_K,
    TRAIN_CSV,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from backstory_checker.download import ensure_novels
from backstory_checker.index import load_or_build
from backstory_checker.retrieve import attach_evidence


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            label = raw["label"].strip().lower()
            content = raw["content"].strip()
            if label not in LABEL2ID:
                raise ValueError(f"Unknown label {label!r} on id {raw.get('id')}")
            if not content:
                raise ValueError(f"Empty backstory on id {raw.get('id')}")
            rows.append(
                {
                    "id": raw["id"].strip(),
                    "book_name": raw["book_name"].strip(),
                    "char": raw["char"].strip(),
                    "caption": (raw.get("caption") or "").strip(),
                    "content": content,
                    "label": label,
                    "label_id": LABEL2ID[label],
                }
            )
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


class PairDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["backstory"],
            row["evidence"],
            truncation="only_second",
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = row["label_id"]
        return encoded


class CrossEntropyTrainer(Trainer):
    """Binary classification with cross-entropy over the two class logits."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def _metrics(y_true: list[int], y_pred: list[int]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    per_class = {}
    for class_id, name in ID2LABEL.items():
        per_class[name] = {
            "precision": round(float(precision[class_id]), 4),
            "recall": round(float(recall[class_id]), 4),
            "f1": round(float(f1[class_id]), 4),
            "support": int(support[class_id]),
        }
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    correct = int(sum(int(a == b) for a, b in zip(y_true, y_pred)))
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(np.mean(f1)), 4),
        "correct": correct,
        "support": len(y_true),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_labels": ["contradict", "consistent"],
    }


def _print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print(f"  accuracy: {metrics['accuracy']:.3f} ({metrics['correct']}/{metrics['support']})")
    print(f"  macro_f1: {metrics['macro_f1']:.3f}")
    for name, row in metrics["per_class"].items():
        print(
            f"  {name}: precision {row['precision']:.3f}  "
            f"recall {row['recall']:.3f}  f1 {row['f1']:.3f}  n {row['support']}"
        )
    print("  confusion rows are true [contradict, consistent], columns are predicted")
    print(f"  {metrics['confusion_matrix']}")


def _predict(model, tokenizer, rows: list[dict]) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    batches = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start : start + BATCH_SIZE]
            encoded = tokenizer(
                [row["backstory"] for row in batch_rows],
                [row["evidence"] for row in batch_rows],
                truncation="only_second",
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            batches.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(batches)


def _write_predictions(path: Path, rows: list[dict], probabilities: np.ndarray) -> None:
    fieldnames = [
        "id",
        "book_name",
        "char",
        "label",
        "prediction",
        "correct",
        "prob_contradict",
        "prob_consistent",
    ]
    for rank in range(1, 6):
        fieldnames.append(f"sim_{rank}")
        fieldnames.append(f"passage_{rank}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row, probs in zip(rows, probabilities):
            pred_id = int(np.argmax(probs))
            record = {
                "id": row["id"],
                "book_name": row["book_name"],
                "char": row["char"],
                "label": row["label"],
                "prediction": ID2LABEL[pred_id],
                "correct": int(pred_id == row["label_id"]),
                "prob_contradict": f"{probs[0]:.4f}",
                "prob_consistent": f"{probs[1]:.4f}",
            }
            for rank, (passage, score) in enumerate(zip(row["passages"], row["scores"]), start=1):
                record[f"sim_{rank}"] = f"{score:.4f}"
                record[f"passage_{rank}"] = passage
            writer.writerow(record)


def _training_args(train_size: int) -> TrainingArguments:
    total_steps = math.ceil(train_size / BATCH_SIZE) * EPOCHS
    warmup_steps = max(1, round(total_steps * WARMUP_RATIO))
    kwargs = {
        "output_dir": str(OUTPUT_DIR / "trainer"),
        "num_train_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": "linear",
        "optim": "adamw_torch",
        "logging_strategy": "epoch",
        "save_strategy": "no",
        "report_to": "none",
        "seed": SEED,
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 0,
        "use_cpu": True,
    }
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    if "use_cpu" not in signature:
        kwargs.pop("use_cpu")
    return TrainingArguments(**kwargs)


def run() -> dict:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"
    torch.set_num_threads(os.cpu_count() or 4)
    _seed(SEED)
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing labeled file: {TRAIN_CSV}")

    print(
        f"DeBERTa-v3-small | epochs {EPOCHS} | lr {LEARNING_RATE} | "
        f"top {TOP_K} passages | max length {MAX_LENGTH}"
    )
    novel_paths = ensure_novels()
    rows = load_rows(TRAIN_CSV)
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    indexes = {
        book_name: load_or_build(book_name, path, embedder)
        for book_name, path in novel_paths.items()
    }
    attach_evidence(rows, indexes, embedder)
    del embedder
    gc.collect()

    train_rows, test_rows = train_test_split(
        rows,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=[row["label"] for row in rows],
    )
    print(f"Labeled rows: {len(rows)} | train {len(train_rows)} | test {len(test_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_MODEL,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    trainer = CrossEntropyTrainer(
        model=model,
        args=_training_args(len(train_rows)),
        train_dataset=PairDataset(train_rows, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(MODEL_DIR)

    train_probs = _predict(model, tokenizer, train_rows)
    test_probs = _predict(model, tokenizer, test_rows)
    train_metrics = _metrics(
        [row["label_id"] for row in train_rows],
        [int(i) for i in np.argmax(train_probs, axis=1)],
    )
    test_metrics = _metrics(
        [row["label_id"] for row in test_rows],
        [int(i) for i in np.argmax(test_probs, axis=1)],
    )
    _print_metrics("Train", train_metrics)
    _print_metrics("Test", test_metrics)

    losses = [
        {"epoch": entry.get("epoch"), "loss": entry.get("loss")}
        for entry in trainer.state.log_history
        if "loss" in entry
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_predictions(OUTPUT_DIR / "test_predictions.csv", test_rows, test_probs)
    report = {
        "train": train_metrics,
        "test": test_metrics,
        "epoch_losses": losses,
        "train_ids": [row["id"] for row in train_rows],
        "test_ids": [row["id"] for row in test_rows],
        "settings": {
            "classifier": CLASSIFIER_MODEL,
            "embed_model": EMBED_MODEL,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "top_k": TOP_K,
            "words_per_passage_in_model": MODEL_PASSAGE_WORDS,
            "input": "backstory passage + five retrieved novel passages",
        },
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved model to {MODEL_DIR}")
    print(f"Saved metrics to {OUTPUT_DIR / 'metrics.json'}")
    print(f"Saved test predictions to {OUTPUT_DIR / 'test_predictions.csv'}")
    return report
