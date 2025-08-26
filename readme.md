# Text–Image Matcher (Metric Suite)

A small, model-agnostic toolkit to score how well an image matches a piece of text. It combines four complementary signals and a pluggable **question generation (QG)** step for VQA.

---

## Metrics & Default Models (with rationale)

1) **Joint-Embedding Similarity**
   *What:* cosine between image and text embeddings.
   *Default model:* **CLIP** (OpenAI ViT-B/32 via HuggingFace Transformers).
   *Why:* CLIP is the de facto baseline for global text–image alignment; fast, robust, zero-shot.

2) **Caption → STS Compare**
   *What:* caption the image, then compute semantic similarity between the caption and the input text.
   *Default models:* **BLIP** image-to-text (captioner) + **SentenceTransformer** (e.g., `all-mpnet-base-v2`) for STS.
   *Why:* converts vision to text, then checks textual entailment-ish similarity; catches phrasing nuances CLIP may miss.

3) **Grounded Concept Coverage**
   *What:* detect target entities/attributes/relations mentioned in the text, score soft F1 (precision/recall).
   *Default model:* **OWL-ViT** zero-shot object detection (Transformers pipeline).
   *Why:* encourages *localized* evidence, not just global alignment; useful for object presence and simple relations.

4) **VQA Compliance**
   *What:* ask yes/no questions derived from the text; average the model's “Yes” probabilities.
   *Default model:* **ViLT** (VQA pipeline).
   *QG backends:* `t5` (generative text2text).
   *Why:* targeted checks for counts/attributes/relations (“Are there two agents?”, “Are both wearing glasses?”).

> All models are pluggable via YAML; wrappers normalize their APIs to simple protocols (`encode_*`, `caption`, `similarity`, `detect`, `yesno_prob`).

---

## Resource Footprint & Runtime (Colab, A100)

- **Host RAM:** ~ **3 GB** (models + tokenizer/caches, default config).
- **GPU VRAM:** ~ **3 GB** (mixed precision, single image per step).
- **Wall time:** ~ **≈ 1 minute** to process a small CSV sample on **Colab A100** with defaults.

> Numbers vary with model choice (base vs large), batch size, and whether weights are freshly downloaded or cached.

---

## Quick Start

```bash
pip install -r requirements.txt

python3 matcher.py --config config.yaml \
    --csv "/content/leonardo_challenge/Leonardo Challenge File/challenge_set.csv" \
    --image-col "url" --text-col "caption" --out scores.csv
```
---

## Performance Ideas (parallelism & throughput)

- **Parallel over rows:**
  - Python: `multiprocessing` / `concurrent.futures.ProcessPoolExecutor` (CPU-bound caption/STS)
  - GPU-aware: micro-batch rows and run the **same metric in batches** to maximize GPU utilization.
- **Parallel across metrics:**
  - Run **embedding / caption / detection / VQA** in separate worker threads or processes, then aggregate.
- **Batching:**
  - Use Transformers pipeline `batch_size` where available; for STS, pass a list to `.encode(..)` to get one batched forward.
- **Mixed precision & autocast:** lower VRAM & higher throughput on GPU.
- **Model choices:** swap to smaller checkpoints (e.g., MiniLM for STS) when speed matters.

---

## Deploying with a Message Broker (cloud-native)

**Goal:** scale out scoring jobs across many workers; decouple API from heavy GPU inference.

### Reference Architecture

- **API layer:** FastAPI/Flask service that accepts CSVs or object-store URIs, creates a job, returns a job ID.
- **Broker/Queue:** Redis (Celery/RQ), RabbitMQ, or Kafka for higher throughput & streaming.
- **Workers:** containerized **GPU** workers running this repo; pull jobs, load models once, process batches, push results.
- **Storage:**
  - Inputs & outputs: S3/GCS/Azure Blob
  - Metadata & scores: Postgres/BigQuery/Redshift (or parquet in object storage)
- **Orchestration:**
  - GCP: Cloud Run (for API) + GKE (GPU worker pool)
  - AWS: ECS/EKS with GPU nodes (g4/g5) + SQS/SNS or MSK (Kafka)
  - Azure: AKS + Event Hubs/RabbitMQ
- **Observability:** Prometheus/Grafana, OpenTelemetry traces, structured logs.

### Message Flow

1. Client `POST /jobs` with `config.yaml` + CSV path (or upload to bucket).
2. API enqueues a message `{job_id, csv_uri, cfg_uri}`.
3. Worker pulls message - downloads CSV - loads YAML - runs matcher (batched).
4. Worker writes `scores.csv` to bucket, emits completion message `{job_id, status, output_uri}`.
5. Client polls `GET /jobs/{job_id}` or receives a webhook/callback.

### Reliability & Scale Tips

- **Idempotency keys:** avoid double-processing on retries.
- **Chunking:** split very large CSVs into shards; one message per shard.
- **Backpressure:** use queue TTLs, dead-letter queues, and worker concurrency caps.
- **Model warmup:** pre-load and keep models hot; avoid per-task reloads.

---

## Notes

- For long prompts, CLIP text is **truncated** to its 77-token limit (adapter supports truncation or optional sliding windows to average chunk embeddings).
- VQA QG is fully **generative** (`t5` backend) to avoid hard‑coded templates.
- A simple CSV helper joins **relative image paths** with the CSV’s parent folder and can normalize Leonardo CDN links to local filenames.

---

## License & Credits

- Uses open-source models via Hugging Face. Be mindful of each model’s license for commercial use.
- OpenAI Codex as coding copilot

