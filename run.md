# Backstory checker run

This file records how the deleted project was rebuilt and what the training runs did.

The task is binary. A short backstory is either consistent with its novel or it contradicts the novel. A whole book cannot be passed into the classifier, so the book is cut into passages, the five passages closest in meaning to the backstory are retrieved, and DeBERTa then classifies that pair.

Labels: `contradict` = 0, `consistent` = 1.

The classifier input is the backstory text only. Character name and caption are not model features. They are kept on the prediction file so a row can be traced.

## Data

`data/train.csv` has the 80 labeled rows that still exist. Ids run up to 139, so this is a subset of an older set.

| Book | Rows |
|---|---|
| In Search of the Castaways | 49 |
| The Count of Monte Cristo | 31 |

Labels in the full file: 51 consistent, 29 contradict.

Columns: `id`, `book_name`, `char`, `caption`, `content`, `label`. `content` is the backstory. Captions are empty on every Castaways row.

## Package

```
run_pipeline.py
backstory_checker/
  config.py
  download.py
  chunking.py
  index.py
  retrieve.py
  train.py
  predict.py
```

Run from this folder with the local environment:

```
.\.venv\Scripts\python.exe -u run_pipeline.py
```

Score one new backstory after a model exists:

```
.\.venv\Scripts\python.exe -m backstory_checker.predict --book "The Count of Monte Cristo" --text "..."
```

## Step 1. Download the novels

There is no GPU and PyTorch was not installed at the start. A virtual environment was created at `.venv` and the CPU build of PyTorch 2.14.1 was installed, plus Transformers, Sentence-Transformers, and scikit-learn.

Public-domain Project Gutenberg texts were saved under `data/novels/`:

| Book | Ebook | File |
|---|---|---|
| In Search of the Castaways | 2083 | `in_search_of_the_castaways.txt` |
| The Count of Monte Cristo | 1184 | `the_count_of_monte_cristo.txt` |

Gutenberg license text is stripped. The story text starts at chapter 1 of each book (Castaways opens on the yacht on 26 July 1864; Monte Cristo opens in Marseilles on 24 February 1815).

## Step 2. Cut the books into overlapping passages

Each novel is split into windows of 60 words with a 15-word overlap. Windows that still contain the Gutenberg boilerplate are dropped.

| Book | Passages |
|---|---|
| In Search of the Castaways | 3,450 |
| The Count of Monte Cristo | 10,231 |

## Step 3. Embed the passages

Model: `sentence-transformers/all-MiniLM-L6-v2`.

Each passage becomes one vector. The vectors are stored with NumPy so the books are not embedded again:

- `data/cache/in_search_of_the_castaways_embeddings.npy`
- `data/cache/the_count_of_monte_cristo_embeddings.npy`
- matching `*_chunks.json` and `*_meta.json`

Embedding the two books on CPU took a few minutes. Later runs load this cache.

## Step 4. Retrieve the top 5 passages

The backstory is embedded with the same model. Cosine similarity (`sklearn.metrics.pairwise.cosine_similarity`) is computed against every passage in that row's book. The five highest scores are kept.

The classifier does not receive the full 60-word passages. It receives the first 8 words of each of the five, joined as the evidence text. That limit was added after the first training attempt, so five passages could fit in a short sequence on CPU. The full passages are still written to `outputs/test_predictions.csv`.

## Step 5. Split and train

Stratified holdout, seed 42, 20% test:

- train: 64 rows (41 consistent, 23 contradict)
- test: 16 rows (10 consistent, 6 contradict)

Classifier: `microsoft/deberta-v3-small`, a new 2-way head.

The pair is tokenized as backstory plus evidence. If the pair is too long, the backstory is kept and the evidence is truncated.

Loss is cross-entropy over the two logits. Optimizer is AdamW with weight decay 0.01. Batch size 4.

## First training attempt

Settings: 10 epochs, learning rate `1e-5`, max length 512, full retrieved passages.

This machine has no NVIDIA GPU. One update at length 512 took about 3.5 minutes. At 16 updates per epoch that is about 9 hours, so the run was stopped after two steps. No model or score was saved from it.

A short timing check on CPU, batch size 4, came out roughly like this:

| Sequence length | Seconds per update | Hours for 10 epochs |
|---|---|---|
| 96 | 53 | 2.4 |
| 128 | 68 | 3.0 |
| 160 | 84 | 3.7 |
| 192 | 118 | 5.2 |
| 512 | about 210 | about 9 |

Freezing the word embeddings did not speed the step up. The time is in the transformer layers.

## Second training attempt

This is the run that finished. Settings were reduced so it could complete on CPU:

| Setting | Value |
|---|---|
| Epochs | 6 |
| Learning rate | 1e-5 |
| Weight decay | 0.01 |
| Warmup | 6% of steps |
| Batch size | 4 |
| Max length | 96 |
| Words kept from each retrieved passage | 8 |
| Top passages | 5 |
| Updates | 96 (16 per epoch) |

While the machine was awake, an update took about 60 to 90 seconds. The log clock shows about 34 hours because the laptop slept between steps. The process itself exited 0 at 07:27 UTC on 29 October 2026.

Logged training loss and gradient norm:

| Epoch | Loss | Gradient norm |
|---|---|---|
| 1 | 0.391 | NaN |
| 2 | 0 | NaN |
| 3 | 0 | NaN |
| 4 | 0 | NaN |
| 5 | 0 | NaN |
| 6 | 0 | NaN |

The averaged training loss reported at the end was 0.065. Gradients were already NaN in epoch 1, then the loss sat at 0. The weights overflowed. Later epochs were not learning.

## Scores from the finished run

Confusion matrices are rows = true label, columns = predicted label, in the order contradict, consistent.

Train, 64 rows:

|  | Pred contradict | Pred consistent |
|---|---|---|
| True contradict | 23 | 0 |
| True consistent | 41 | 0 |

- accuracy 0.359 (23/64)
- macro F1 0.264
- consistent precision, recall, and F1 are all 0

Test, 16 rows:

|  | Pred contradict | Pred consistent |
|---|---|---|
| True contradict | 6 | 0 |
| True consistent | 10 | 0 |

- accuracy 0.375 (6/16)
- macro F1 0.273
- consistent precision, recall, and F1 are all 0

Every train row and every test row was predicted contradict. The 0.375 test accuracy is the share of the test set that really is contradict (6/16). It is not evidence that the model can tell the two classes apart.

## Files written by the finished run

- `outputs/metrics.json` — scores, epoch losses, the train and test ids, and the settings above
- `outputs/test_predictions.csv` — one row per test backstory, with probabilities and the five full passages
- `outputs/model` — the saved classifier and tokenizer from this broken run

The passage cache under `data/cache/` is still valid and does not depend on the failed fine-tune.

## What this means

Retrieval ran. The books, embeddings, top-5 search, split, and training loop all completed. The fine-tune did not. A learning rate of `1e-5` was meant to allow several epochs without overfitting, but on this CPU run the DeBERTa update produced NaN gradients in the first epoch, and the head collapsed to a single class.

Do not use `outputs/model` for a real check. A new run needs a lower learning rate so the gradients stay finite. The cached passages can be reused.
