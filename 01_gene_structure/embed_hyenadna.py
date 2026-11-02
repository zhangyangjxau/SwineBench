#!/usr/bin/env python3
"""Precompute per-base HyenaDNA embeddings (single-nucleotide tokens, structural special tokens removed) to H5."""
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
from transformers import AutoConfig, AutoModel, AutoTokenizer

LABEL_CHARS = "0123456789ABC"
LABEL_NAMES = [
    "NC",
    "CDS_F", "UTR5_F", "UTR3_F", "DF", "IF", "AF",
    "CDS_R", "UTR5_R", "UTR3_R", "DR", "IR", "AR",
]
CHAR_TO_INT = {c: i for i, c in enumerate(LABEL_CHARS)}
VALID_DNA = set("ACGTN")


def decode_labels(s: str) -> np.ndarray:
    arr = np.empty(len(s), dtype=np.uint8)
    for i, c in enumerate(s):
        try:
            arr[i] = CHAR_TO_INT[c]
        except KeyError as exc:
            raise ValueError(f"unknown label char {c!r}") from exc
    return arr


def clean_sequence(seq: str) -> str:
    seq = seq.upper()
    return "".join(c if c in VALID_DNA else "N" for c in seq)


def iter_jsonl(path: Path):
    with gzip.open(path, "rt") as fh:
        for idx, line in enumerate(fh):
            rec = json.loads(line)
            if rec.get("schema") != "bend13":
                raise ValueError(f"{path}:{idx} expected schema=bend13, got {rec.get('schema')!r}")
            seq = clean_sequence(rec["sequence"])
            labels = decode_labels(rec["labels"])
            if len(seq) != labels.shape[0]:
                raise ValueError(f"{path}:{idx} sequence/label length mismatch")
            yield idx, rec, seq, labels


def infer_token_limit(tokenizer, config, override: int) -> int:
    if override and override > 0:
        return int(override)
    candidates: List[int] = []
    tok_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_limit, int) and 0 < tok_limit < 10**8:
        candidates.append(tok_limit)
    for attr in ("max_seq_len", "max_position_embeddings", "max_sequence_length"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            candidates.append(value)
    return min(candidates) if candidates else 1024


def infer_hidden_size(config) -> int:
    for attr in ("d_model", "hidden_size", "n_embd"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return int(value)
    raise ValueError("Could not infer hidden size from HyenaDNA config")


class BatchedHyenaDNAEncoder:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        target_chunk_bp: int,
        max_tokens: int,
        gpu_batch: int,
        storage_dtype: str,
        forward_dtype: str,
    ):
        self.model_name = model_name
        self.device = device
        self.target_chunk_bp = int(target_chunk_bp)
        self.gpu_batch = int(gpu_batch)
        self.storage_dtype = storage_dtype
        self.forward_dtype = forward_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # HyenaDNA is causal and ignores attention masks: pad on the right so padding never precedes real bases
        self.tokenizer.padding_side = "right"
        assert self.tokenizer.padding_side == "right"
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.max_tokens = infer_token_limit(self.tokenizer, self.config, max_tokens)
        self.hidden_size = infer_hidden_size(self.config)

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        unk_id = self.tokenizer.unk_token_id
        structural_ids = set()
        for name in (
            "bos_token_id", "eos_token_id", "sep_token_id",
            "cls_token_id", "pad_token_id", "mask_token_id",
        ):
            sid = getattr(self.tokenizer, name, None)
            if sid is not None and sid != unk_id:
                structural_ids.add(int(sid))
        for sid in getattr(self.tokenizer, "all_special_ids", []):
            if sid is not None and sid != unk_id:
                tok = self.tokenizer.convert_ids_to_tokens(int(sid))
                if tok in {"[BOS]", "[EOS]", "[SEP]", "[CLS]", "[PAD]", "[MASK]", "<s>", "</s>", "<pad>"}:
                    structural_ids.add(int(sid))
        self.structural_special_ids = np.asarray(sorted(structural_ids), dtype=np.int64)
        self.pad_token_id = self.tokenizer.pad_token_id

    def _tokenized_len(self, chunk: str) -> int:
        return len(self.tokenizer(chunk, add_special_tokens=True, truncation=False)["input_ids"])

    def _split_chunk_to_fit(self, chunk: str) -> List[str]:
        if not chunk:
            return []
        stack = [chunk]
        out: List[str] = []
        while stack:
            cur = stack.pop()
            tok_len = self._tokenized_len(cur)
            if tok_len <= self.max_tokens:
                out.append(cur)
                continue
            if len(cur) == 1:
                raise RuntimeError(
                    f"single-base chunk tokenized to {tok_len} tokens, exceeding max_tokens={self.max_tokens}"
                )
            mid = len(cur) // 2
            stack.append(cur[mid:])
            stack.append(cur[:mid])
        return out

    def chunks_of_sequence(self, seq: str) -> List[str]:
        rough = [seq[i:i + self.target_chunk_bp] for i in range(0, len(seq), self.target_chunk_bp)]
        chunks: List[str] = []
        for chunk in rough:
            chunks.extend(self._split_chunk_to_fit(chunk))
        if sum(len(c) for c in chunks) != len(seq):
            raise RuntimeError("internal chunking error: chunks do not reconstruct sequence length")
        return chunks

    def _autocast_context(self):
        if self.device.type != "cuda" or self.forward_dtype == "float32":
            return torch.autocast(device_type="cuda", enabled=False)
        if self.forward_dtype == "float16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        if self.forward_dtype == "bfloat16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        raise ValueError(f"unsupported forward_dtype={self.forward_dtype}")

    @torch.inference_mode()
    def _encode_one_batch(self, chunks: List[str]) -> Tuple[List[np.ndarray], int]:
        batch = self.tokenizer(
            chunks,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            truncation=False,
        )
        input_ids_cpu = batch["input_ids"].detach().cpu().numpy()
        if input_ids_cpu.shape[1] > self.max_tokens:
            raise RuntimeError(
                f"tokenized batch length {input_ids_cpu.shape[1]} exceeds max_tokens={self.max_tokens}; "
                "chunk splitting should have prevented this"
            )

        if "attention_mask" in batch:
            attn_cpu = batch["attention_mask"].detach().cpu().numpy().astype(bool)
        elif self.pad_token_id is None:
            attn_cpu = np.ones_like(input_ids_cpu, dtype=bool)
        else:
            attn_cpu = input_ids_cpu != int(self.pad_token_id)

        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        with self._autocast_context():
            out = self.model(input_ids)
        hidden = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
        hidden = hidden.detach().float().cpu().numpy()

        if self.structural_special_ids.size:
            is_structural = np.isin(input_ids_cpu, self.structural_special_ids)
        else:
            is_structural = np.zeros_like(input_ids_cpu, dtype=bool)

        results: List[np.ndarray] = []
        for b, chunk in enumerate(chunks):
            active = attn_cpu[b] & (~is_structural[b])
            valid_positions = np.where(active)[0]
            if valid_positions.size != len(chunk):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids_cpu[b, : min(30, input_ids_cpu.shape[1])].tolist())
                raise RuntimeError(
                    f"HyenaDNA per-base length mismatch: active_tokens={valid_positions.size}, "
                    f"chunk_bp={len(chunk)}, first_tokens={tokens}, "
                    f"structural_ids={self.structural_special_ids.tolist()}"
                )

            emb = hidden[b, valid_positions, :]
            if not np.isfinite(emb).all():
                n_nan = int(np.isnan(emb).sum())
                n_inf = int(np.isinf(emb).sum())
                raise FloatingPointError(f"non-finite HyenaDNA embedding: nan={n_nan}, inf={n_inf}")
            if self.storage_dtype == "float16":
                emb = emb.astype(np.float16, copy=False)
            elif self.storage_dtype in {"float32", "bfloat16"}:
                emb = emb.astype(np.float32, copy=False)
            else:
                raise ValueError(f"unsupported storage_dtype={self.storage_dtype}")
            results.append(emb)
        return results, int(input_ids_cpu.shape[1])

    def encode_records(self, sequences: List[str]) -> Tuple[List[np.ndarray], int, int]:
        all_chunks: List[Tuple[int, int, str]] = []
        for rec_idx, seq in enumerate(sequences):
            for chunk_idx, chunk in enumerate(self.chunks_of_sequence(seq)):
                all_chunks.append((rec_idx, chunk_idx, chunk))

        order = sorted(range(len(all_chunks)), key=lambda i: len(all_chunks[i][2]), reverse=True)
        chunk_emb: List[Optional[np.ndarray]] = [None] * len(all_chunks)
        max_seen_tokens = 0
        for b0 in range(0, len(order), self.gpu_batch):
            batch_idx = order[b0:b0 + self.gpu_batch]
            batch_chunks = [all_chunks[i][2] for i in batch_idx]
            embs, tok_len = self._encode_one_batch(batch_chunks)
            max_seen_tokens = max(max_seen_tokens, tok_len)
            for slot, emb in zip(batch_idx, embs):
                chunk_emb[slot] = emb

        per_record: List[List[np.ndarray]] = [[] for _ in sequences]
        for (rec_idx, _, _), emb in sorted(zip(all_chunks, chunk_emb), key=lambda t: (t[0][0], t[0][1])):
            if emb is None:
                raise RuntimeError("internal encoding error: missing chunk embedding")
            per_record[rec_idx].append(emb)
        return [np.concatenate(parts, axis=0) for parts in per_record], len(all_chunks), max_seen_tokens


def h5_is_complete(path: Path, expected_model: str, expected_storage_dtype: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with h5py.File(path, "r") as h5:
            return (
                bool(h5.attrs.get("complete", False))
                and h5.attrs.get("schema") == "bend13"
                and h5.attrs.get("encoder") == "hyenadna_per_base"
                and h5.attrs.get("model_name") == expected_model
                and h5.attrs.get("storage_dtype") == expected_storage_dtype
            )
    except OSError:
        return False


def precompute_split(
    enc: BatchedHyenaDNAEncoder,
    jsonl_path: Path,
    out_path: Path,
    compression: str,
    log_every: int,
    record_batch: int,
    overwrite: bool,
):
    if h5_is_complete(out_path, enc.model_name, enc.storage_dtype) and not overwrite:
        print(json.dumps({"event": "split_skip", "split": jsonl_path.stem, "out": str(out_path)}), flush=True)
        return

    t0 = time.time()
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    records = list(iter_jsonl(jsonl_path))
    total = len(records)
    print(json.dumps({"event": "split_start", "split": jsonl_path.stem, "n_records": total}), flush=True)

    with h5py.File(tmp, "w") as h5:
        h5.attrs["schema"] = "bend13"
        h5.attrs["num_classes"] = 13
        h5.attrs["hidden_size"] = enc.hidden_size
        h5.attrs["dtype"] = enc.storage_dtype
        h5.attrs["storage_dtype"] = enc.storage_dtype
        h5.attrs["forward_dtype"] = enc.forward_dtype
        h5.attrs["model_name"] = enc.model_name
        h5.attrs["encoder"] = "hyenadna_per_base"
        h5.attrs["label_chars"] = LABEL_CHARS
        h5.attrs["label_names"] = json.dumps(LABEL_NAMES)
        h5.attrs["source"] = str(jsonl_path)
        h5.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        h5.attrs["target_chunk_bp"] = enc.target_chunk_bp
        h5.attrs["max_tokens"] = enc.max_tokens
        h5.attrs["gpu_batch"] = enc.gpu_batch
        h5.attrs["padding_side"] = enc.tokenizer.padding_side
        h5.attrs["record_batch"] = record_batch
        h5.attrs["structural_special_ids"] = json.dumps(enc.structural_special_ids.tolist())

        emb_grp = h5.create_group("embeddings")
        lab_grp = h5.create_group("labels")
        meta_grp = h5.create_group("meta")
        lengths: List[int] = []
        ids: List[str] = []
        chroms: List[str] = []
        starts: List[int] = []
        ends: List[int] = []
        total_chunks = 0
        max_seen_tokens = 0
        done = 0

        for b0 in range(0, total, record_batch):
            batch = records[b0:b0 + record_batch]
            seqs = [seq for _, _, seq, _ in batch]
            embs, n_chunks, batch_max_tokens = enc.encode_records(seqs)
            total_chunks += n_chunks
            max_seen_tokens = max(max_seen_tokens, batch_max_tokens)

            for (idx, rec, seq, labels), emb in zip(batch, embs):
                if emb.shape[0] != labels.shape[0]:
                    raise RuntimeError(f"{jsonl_path}:{idx} emb len {emb.shape[0]} != label len {labels.shape[0]}")
                emb_grp.create_dataset(str(idx), data=emb, compression=compression)
                lab_grp.create_dataset(str(idx), data=labels, compression=compression)
                lengths.append(int(labels.shape[0]))
                ids.append(str(rec.get("id", idx)))
                chroms.append(str(rec.get("chrom", "")))
                starts.append(int(rec.get("start0", -1)))
                ends.append(int(rec.get("end0", -1)))

            done += len(batch)
            if log_every and (done % log_every == 0 or done == total):
                elapsed = time.time() - t0
                peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                print(json.dumps({
                    "event": "encode_progress",
                    "split": jsonl_path.stem,
                    "n": done,
                    "total": total,
                    "chunks": total_chunks,
                    "max_seen_tokens": max_seen_tokens,
                    "elapsed_sec": round(elapsed, 1),
                    "sec_per_record": round(elapsed / max(done, 1), 3),
                    "peak_vram_gb": round(peak_vram, 2),
                }), flush=True)

        meta_grp.create_dataset("lengths", data=np.asarray(lengths, dtype=np.int32))
        meta_grp.create_dataset("ids", data=np.asarray(ids, dtype=h5py.string_dtype()))
        meta_grp.create_dataset("chroms", data=np.asarray(chroms, dtype=h5py.string_dtype()))
        meta_grp.create_dataset("start0", data=np.asarray(starts, dtype=np.int64))
        meta_grp.create_dataset("end0", data=np.asarray(ends, dtype=np.int64))
        h5.attrs["num_records"] = done
        h5.attrs["num_chunks"] = total_chunks
        h5.attrs["max_seen_tokens"] = max_seen_tokens
        h5.attrs["complete"] = True

    tmp.rename(out_path)
    print(json.dumps({
        "event": "split_done",
        "split": jsonl_path.stem,
        "out": str(out_path),
        "n": done,
        "chunks": total_chunks,
        "max_seen_tokens": max_seen_tokens,
        "elapsed_sec": round(time.time() - t0, 1),
    }), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default="LongSafari/hyenadna-medium-450k-seqlen-hf")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--target-chunk-bp", type=int, default=30000)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--gpu-batch", type=int, default=2)
    ap.add_argument("--record-batch", type=int, default=8)
    ap.add_argument("--storage-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--forward-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--compression", default="lzf")
    ap.add_argument("--log-every", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available; this will be very slow.", flush=True)

    enc = BatchedHyenaDNAEncoder(
        model_name=args.model_name,
        device=device,
        target_chunk_bp=args.target_chunk_bp,
        max_tokens=args.max_tokens,
        gpu_batch=args.gpu_batch,
        storage_dtype=args.storage_dtype,
        forward_dtype=args.forward_dtype,
    )
    print(json.dumps({
        "event": "encoder_ready",
        "model": args.model_name,
        "hidden_size": enc.hidden_size,
        "device": str(device),
        "target_chunk_bp": enc.target_chunk_bp,
        "max_tokens": enc.max_tokens,
        "gpu_batch": enc.gpu_batch,
        "storage_dtype": enc.storage_dtype,
        "forward_dtype": enc.forward_dtype,
        "structural_special_ids": enc.structural_special_ids.tolist(),
    }), flush=True)

    for split in args.splits:
        jsonl = data_dir / f"{split}.jsonl.gz"
        out = out_dir / f"{split}.h5"
        if not jsonl.exists():
            raise FileNotFoundError(jsonl)
        precompute_split(
            enc=enc,
            jsonl_path=jsonl,
            out_path=out,
            compression=args.compression,
            log_every=args.log_every,
            record_batch=args.record_batch,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
