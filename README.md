# HODL Project — WhatsApp Marketplace Dataset + Multimodal Pricing Model

This repository contains:

- A **multimodal modeling notebook** (`ModelCode.ipynb`) that builds and fine-tunes a text+image model for **price prediction**.
- A **WhatsApp marketplace dataset** (`dataset_final/`) created from the Sloan Buy/Sell WhatsApp group export.
- The **dataset-generation pipeline** (`HODL_db_generation/`) used to extract structured listings (description/price/image) from WhatsApp chats and PDFs, and to build the final dataset.

---

## Repository structure

```
FinalPricing.ipynb
HODL_db_generation/
  market_extractor.py
  download_docs.py
  build_final_dataset.py
  requirements.txt
```

---

## `FinalPricing.ipynb` — what the notebook does

`FinalPricing.ipynb` documents the end-to-end modeling process:

- **Load a large public dataset in streaming mode** (via Hugging Face `datasets`) to avoid downloading everything at once.
- **Preprocess** examples by:
  - decoding/loading images
  - filtering to rows that include a usable `price` field
  - transforming the target to **log-price** for more stable regression
- **Build a multimodal model**:
  - **Text encoder**: a BERT-family encoder processes the item description.
  - **Image encoder**: a ViT-family encoder processes the item image.
  - The notebook concatenates the pooled embeddings and trains a **regression head** to predict log-price.
- **Train + evaluate**:
  - trains on a train split, evaluates on a test split
  - computes regression metrics including **R²** and **MAPE**
- **Fine-tune on our local WhatsApp dataset**:
  - loads `dataset_final/dataset.csv`
  - resolves `image_ref` to files under `dataset_final/images/`
  - builds `tf.data` pipelines
  - unfreezes encoders and fine-tunes with a **low learning rate** on the small dataset

> Note: The notebook uses both PyTorch checkpoints (via `transformers`) and TensorFlow/Keras (`tf.keras`) for training/evaluation.

---

## `HODL_db_generation/` — how we created the dataset

All dataset creation code lives in `HODL_db_generation/`.

### 1) Extract items from WhatsApp + PDFs (`market_extractor.py`)

`market_extractor.py` implements a **two-pass extraction pipeline** using the OpenAI API:

- **Pass 1 (text-only)**: parse the exported WhatsApp chat text and extract structured sale items (description, price, candidate images) using a strict JSON schema.
- **Pass 2 (multimodal)**: validate which candidate image best matches the item description/price and assign a single `image_ref` (or `null`).

It also supports extracting items from PDFs:

- extracts per-page text
- extracts images from pages
- runs the same two-pass logic (page text → candidate images → select best image)

Outputs are typically written to:

- `HODL_db_generation/output/whatsapp_items.json`
- `HODL_db_generation/output/pdf_items.json`

### 2) Download linked documents (`download_docs.py`)

`download_docs.py` scans the WhatsApp export (`_chat.txt`) for document links (Google Docs/Drive and direct PDFs) and downloads them into:

- `HODL_db_generation/output/downloaded_docs/`

### 3) Build the final dataset (`build_final_dataset.py`)

`build_final_dataset.py` creates `HODL_db_generation/dataset_final/` by:

- combining `output/pdf_items.json` + `output/whatsapp_items.json`
- removing rows missing required fields and de-duplicating
- copying referenced images into `dataset_final/images/`
- validating that images match descriptions (and removing mismatches)
- writing the final CSV (`dataset_final/dataset.csv`)

---

## Running the pipeline (high level)

From `HODL_db_generation/`:

1. Install deps:

```bash
pip install -r requirements.txt
```

2. Set your API key:

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

3. Run extraction:

```bash
python market_extractor.py whatsapp --chat "WhatsApp Chat - Sloan Buy _ Sell 26s + 25s (1)/_chat.txt" --media_dir "WhatsApp Chat - Sloan Buy _ Sell 26s + 25s (1)" --out "output/whatsapp_items.json"
```

4. (Optional) Download linked PDFs:

```bash
python download_docs.py
```

5. Build final dataset:

```bash
python build_final_dataset.py
```

---

## Data, privacy, and licensing note

This repo includes raw WhatsApp export data and media. If you plan to make the repository public, review and redact any personal/sensitive data first.


