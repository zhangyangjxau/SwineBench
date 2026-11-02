#!/usr/bin/env python3
"""Precompute per-base NT-v2 embeddings (last hidden state, token embedding repeated over its k-mer) to H5."""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

LABEL_CHARS = "0123456789ABC"
CHAR_TO_INT = {c: i for i, c in enumerate(LABEL_CHARS)}
DNA_CHARS = set("ACGTN")


def decode_labels(s: str) -> np.ndarray:
    arr = np.empty(len(s), dtype=np.uint8)
    for i, c in enumerate(s):
        v = CHAR_TO_INT.get(c)
        if v is None:
            raise ValueError(f"Unknown label char {c!r}")
        arr[i] = v
    return arr


def iter_jsonl(path: Path):
    with gzip.open(path, "rt") as fh:
        for idx, line in enumerate(fh):
            rec = json.loads(line)
            if rec.get("schema") != "bend13":
                raise ValueError(f"{path}:{idx} expected schema=bend13, got {rec.get('schema')!r}")
            seq = rec["sequence"].upper()
            labels = decode_labels(rec["labels"])
            if len(seq) != labels.shape[0]:
                raise ValueError(f"{path}:{idx} sequence/label length mismatch")
            yield idx, rec, seq, labels


class BatchedEncoder:
    def __init__(self, model_name: str, device: torch.device,
                 max_chunk_bp: int, gpu_batch: int, dtype: str):
        self.device = device
        self.max_chunk_bp = max_chunk_bp
        self.gpu_batch = gpu_batch
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden_size = int(self.model.config.hidden_size)
        # bp length and special-token flag per token id
        vocab = self.tokenizer.get_vocab()
        max_id = max(vocab.values())
        self._token_bp = np.zeros(max_id + 1, dtype=np.int32)
        for tok, tid in vocab.items():
            self._token_bp[tid] = sum(1 for c in tok.upper() if c in DNA_CHARS)
        self._is_special = np.zeros(max_id + 1, dtype=bool)
        for sid in self.tokenizer.all_special_ids:
            if sid is not None and 0 <= sid <= max_id:
                self._is_special[sid] = True

    def _chunks_of_sequence(self, seq: str) -> List[str]:
        return [seq[i:i + self.max_chunk_bp] for i in range(0, len(seq), self.max_chunk_bp)]

    @torch.inference_mode()
    def _encode_one_batch(self, chunks: List[str]) -> List[np.ndarray]:
        enc = self.tokenizer(chunks, return_tensors="pt", padding=True,
                             add_special_tokens=True, truncation=False)
        enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
        out = self.model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1]  # (B, T, D)
        if self.dtype == "float16":
            hidden = hidden.half()
        elif self.dtype == "bfloat16":
            hidden = hidden.bfloat16()
        input_ids = enc["input_ids"].cpu().numpy()
        attn = enc["attention_mask"].cpu().numpy()
        hidden_cpu = hidden.cpu().numpy()
        results: List[np.ndarray] = []
        for b, chunk in enumerate(chunks):
            mask = (attn[b] == 1) & (~self._is_special[input_ids[b]])
            valid_positions = np.where(mask)[0]
            token_ids = input_ids[b, valid_positions]
            repeats = self._token_bp[token_ids]
            if repeats.sum() != len(chunk):
                raise RuntimeError(f"bp length mismatch: tokens repeat to {repeats.sum()} vs chunk {len(chunk)}")
            emb = np.repeat(hidden_cpu[b, valid_positions], repeats, axis=0)
            results.append(emb)
        return results

    def encode_records(self, sequences: List[str], log_cb=None) -> List[np.ndarray]:
        all_chunks: List[Tuple[int, int, str]] = []
        for r, seq in enumerate(sequences):
            for ci, chunk in enumerate(self._chunks_of_sequence(seq)):
                all_chunks.append((r, ci, chunk))
        order = sorted(range(len(all_chunks)), key=lambda i: len(all_chunks[i][2]), reverse=True)
        chunk_emb: List[Optional[np.ndarray]] = [None] * len(all_chunks)
        for bstart in range(0, len(order), self.gpu_batch):
            batch_idx = order[bstart: bstart + self.gpu_batch]
            batch_chunks = [all_chunks[i][2] for i in batch_idx]
            embs = self._encode_one_batch(batch_chunks)
            for slot, emb in zip(batch_idx, embs):
                chunk_emb[slot] = emb
            if log_cb is not None:
                log_cb(bstart + len(batch_idx), len(order))
        per_rec: List[List[np.ndarray]] = [[] for _ in sequences]
        for (r, _, _), emb in sorted(zip(all_chunks, chunk_emb), key=lambda t: (t[0][0], t[0][1])):
            per_rec[r].append(emb)
        return [np.concatenate(parts, axis=0) for parts in per_rec]


def precompute_split(enc: BatchedEncoder, jsonl_path: Path, out_path: Path,
                     compression: str, log_every: int, record_batch: int):
    t0 = time.time()
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    records_iter = list(iter_jsonl(jsonl_path))
    total = len(records_iter)
    print(json.dumps({"event": "split_start", "split": jsonl_path.stem,
                      "n_records": total}), flush=True)
    with h5py.File(tmp, "w") as h5:
        h5.attrs["schema"] = "bend13"
        h5.attrs["num_classes"] = 13
        h5.attrs["hidden_size"] = enc.hidden_size
        h5.attrs["dtype"] = enc.dtype
        h5.attrs["source"] = str(jsonl_path)
        h5.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        h5.attrs["gpu_batch"] = enc.gpu_batch
        h5.attrs["record_batch"] = record_batch
        emb_grp = h5.create_group("embeddings")
        lab_grp = h5.create_group("labels")
        meta_grp = h5.create_group("meta")
        lengths: List[int] = []
        ids: List[str] = []
        done = 0
        for b0 in range(0, total, record_batch):
            batch = records_iter[b0: b0 + record_batch]
            seqs = [seq for _, _, seq, _ in batch]
            embs = enc.encode_records(seqs)
            for (idx, rec, seq, labels), emb in zip(batch, embs):
                if emb.shape[0] != labels.shape[0]:
                    raise RuntimeError(f"{jsonl_path}:{idx} emb len {emb.shape[0]} != label len {labels.shape[0]}")
                emb_grp.create_dataset(str(idx), data=emb, compression=compression)
                lab_grp.create_dataset(str(idx), data=labels, compression=compression)
                lengths.append(int(labels.shape[0]))
                ids.append(rec["id"])
            done += len(batch)
            if log_every and (done % log_every == 0 or done == total):
                elapsed = time.time() - t0
                vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(json.dumps({"event": "encode_progress", "split": jsonl_path.stem,
                                  "n": done, "total": total,
                                  "elapsed_sec": round(elapsed, 1),
                                  "sec_per_record": round(elapsed / done, 3),
                                  "peak_vram_gb": round(vram, 2)}), flush=True)
        meta_grp.create_dataset("lengths", data=np.asarray(lengths, dtype=np.int32))
        meta_grp.create_dataset("ids", data=np.asarray(ids, dtype=h5py.string_dtype()))
        h5.attrs["num_records"] = done
        h5.attrs["complete"] = True
    tmp.rename(out_path)
    print(json.dumps({"event": "split_done", "split": jsonl_path.stem,
                      "out": str(out_path), "n": done,
                      "elapsed_sec": round(time.time() - t0, 1)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--max-chunk-bp", type=int, default=12000)
    ap.add_argument("--gpu-batch", type=int, default=16,
                    help="Chunks per GPU forward pass")
    ap.add_argument("--record-batch", type=int, default=64,
                    help="Records to load+encode before writing to H5")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--compression", default="lzf")
    ap.add_argument("--log-every", type=int, default=64)
    args = ap.parse_args()

    data_dir = Path(args.data_dir); out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available; falling back to CPU. This will be very slow.", flush=True)
    enc = BatchedEncoder(args.model_name, device, args.max_chunk_bp, args.gpu_batch, args.dtype)
    print(json.dumps({"event": "encoder_ready", "model": args.model_name,
                      "hidden_size": enc.hidden_size, "device": str(device),
                      "gpu_batch": args.gpu_batch, "max_chunk_bp": args.max_chunk_bp,
                      "dtype": args.dtype}), flush=True)
    for split in args.splits:
        jsonl = data_dir / f"{split}.jsonl.gz"
        out = out_dir / f"{split}.h5"
        if not jsonl.exists():
            raise FileNotFoundError(jsonl)
        precompute_split(enc, jsonl, out, args.compression, args.log_every, args.record_batch)


if __name__ == "__main__":
    main()
