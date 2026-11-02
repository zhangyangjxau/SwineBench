#!/usr/bin/env python3
"""Precompute per-base Evo2 embeddings (one layer, optional fixed-seed Gaussian random projection) to H5."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch

LABEL_CHARS = "0123456789ABC"
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


def get_pad_id(tokenizer) -> int:
    for name in ("pad_id", "pad_token_id", "pad"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int):
            return int(value)
    return 0


def load_evo2(model_name: str, local_path: Optional[str]):
    try:
        from evo2 import Evo2
    except Exception as exc:
        raise RuntimeError(
            "Could not import Evo2. Use an environment with evo2, vortex, "
            "flash-attn, CUDA torch, h5py, and numpy installed."
        ) from exc
    if local_path:
        return Evo2(model_name, local_path=local_path)
    return Evo2(model_name)


class Evo2PerBaseEncoder:
    def __init__(
        self,
        model_name: str,
        local_path: Optional[str],
        layer_name: str,
        device: torch.device,
        target_chunk_bp: int,
        gpu_batch: int,
        storage_dtype: str,
        project_dim: int,
        projection_seed: int,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Evo2 precompute; current torch reports cuda=False")
        if gpu_batch != 1:
            print(
                json.dumps({
                    "event": "warn",
                    "message": "Evo2 configs usually set max_batch_size=1; gpu_batch>1 may fail.",
                    "gpu_batch": gpu_batch,
                }),
                flush=True,
            )
        self.model_name = model_name
        self.layer_name = layer_name
        self.device = device
        self.target_chunk_bp = int(target_chunk_bp)
        self.gpu_batch = int(gpu_batch)
        self.storage_dtype = storage_dtype
        self.project_dim = int(project_dim)
        self.projection_seed = int(projection_seed)
        self.projection = None

        self.evo2 = load_evo2(model_name, local_path)
        self.tokenizer = self.evo2.tokenizer
        self.pad_id = get_pad_id(self.tokenizer)
        self.max_seqlen = int(getattr(getattr(self.evo2.model, "config", object()), "max_seqlen", 0) or 0)

        probe = self._encode_one_batch(["ACGT"])
        self.source_hidden_size = int(probe[0].shape[1])
        if self.project_dim > 0:
            rng = np.random.default_rng(self.projection_seed)
            proj = rng.standard_normal((self.source_hidden_size, self.project_dim), dtype=np.float32)
            proj /= math.sqrt(float(self.project_dim))
            self.projection = torch.tensor(proj, dtype=torch.float32, device=self.device)
            self.hidden_size = self.project_dim
        else:
            self.projection = None
            self.hidden_size = self.source_hidden_size

    def chunks_of_sequence(self, seq: str) -> List[str]:
        if self.target_chunk_bp <= 0:
            chunks = [seq]
        else:
            chunks = [seq[i:i + self.target_chunk_bp] for i in range(0, len(seq), self.target_chunk_bp)]
        if self.max_seqlen and max((len(c) for c in chunks), default=0) > self.max_seqlen:
            raise RuntimeError(
                f"chunk length exceeds Evo2 max_seqlen={self.max_seqlen}; "
                f"set --target-chunk-bp <= {self.max_seqlen}"
            )
        return chunks

    def _tokenize_batch(self, chunks: List[str]) -> Tuple[torch.Tensor, List[int]]:
        token_lists = []
        lengths = []
        for chunk in chunks:
            toks = self.tokenizer.tokenize(chunk)
            toks = [int(x) for x in toks]
            if len(toks) != len(chunk):
                raise RuntimeError(f"Evo2 tokenizer length mismatch: {len(toks)} tokens vs {len(chunk)} bp")
            token_lists.append(toks)
            lengths.append(len(toks))
        max_len = max(lengths)
        padded = [toks + [self.pad_id] * (max_len - len(toks)) for toks in token_lists]
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        return input_ids, lengths

    @torch.inference_mode()
    def _encode_one_batch(self, chunks: List[str]) -> List[np.ndarray]:
        input_ids, lengths = self._tokenize_batch(chunks)
        _, emb_dict = self.evo2(
            input_ids,
            return_embeddings=True,
            layer_names=[self.layer_name],
        )
        if self.layer_name not in emb_dict:
            raise KeyError(f"Evo2 layer {self.layer_name!r} not returned")
        emb = emb_dict[self.layer_name]
        if emb.ndim != 3:
            raise RuntimeError(f"expected 3D embedding tensor, got shape={tuple(emb.shape)}")
        if emb.shape[0] != len(chunks) and emb.shape[1] == len(chunks):
            emb = emb.transpose(0, 1)
        if emb.shape[0] != len(chunks):
            raise RuntimeError(f"cannot interpret Evo2 embedding shape={tuple(emb.shape)}")

        out: List[np.ndarray] = []
        for b, n in enumerate(lengths):
            x = emb[b, :n].detach().float()
            if self.projection is not None:
                x = x @ self.projection
            arr = x.cpu().numpy()
            if arr.shape[0] != len(chunks[b]):
                raise RuntimeError(f"embedding length {arr.shape[0]} != chunk length {len(chunks[b])}")
            if not np.isfinite(arr).all():
                raise FloatingPointError("non-finite Evo2 embedding")
            if self.storage_dtype == "float16":
                arr = arr.astype(np.float16, copy=False)
            elif self.storage_dtype == "float32":
                arr = arr.astype(np.float32, copy=False)
            else:
                raise ValueError(f"unsupported storage_dtype={self.storage_dtype}")
            out.append(arr)
        del input_ids, emb
        torch.cuda.empty_cache()
        return out

    def encode_records(self, sequences: List[str]) -> Tuple[List[np.ndarray], int]:
        all_chunks: List[Tuple[int, int, str]] = []
        for rec_idx, seq in enumerate(sequences):
            for chunk_idx, chunk in enumerate(self.chunks_of_sequence(seq)):
                all_chunks.append((rec_idx, chunk_idx, chunk))

        order = sorted(range(len(all_chunks)), key=lambda i: len(all_chunks[i][2]), reverse=True)
        chunk_emb: List[Optional[np.ndarray]] = [None] * len(all_chunks)
        for b0 in range(0, len(order), self.gpu_batch):
            batch_idx = order[b0:b0 + self.gpu_batch]
            embs = self._encode_one_batch([all_chunks[i][2] for i in batch_idx])
            for slot, emb in zip(batch_idx, embs):
                chunk_emb[slot] = emb

        per_rec: List[List[np.ndarray]] = [[] for _ in sequences]
        for i, (rec_idx, _, _) in enumerate(all_chunks):
            emb = chunk_emb[i]
            if emb is None:
                raise RuntimeError("missing chunk embedding")
            per_rec[rec_idx].append(emb)
        return [np.concatenate(parts, axis=0) for parts in per_rec], len(all_chunks)


def h5_complete(path: Path, args) -> bool:
    try:
        with h5py.File(path, "r") as h5:
            return (
                bool(h5.attrs.get("complete", False))
                and h5.attrs.get("schema") == "bend13"
                and h5.attrs.get("encoder") == "evo2_char_per_base"
                and h5.attrs.get("model_name") == args.model_name
                and h5.attrs.get("layer_name") == args.layer_name
                and int(h5.attrs.get("project_dim", 0)) == int(args.project_dim)
            )
    except Exception:
        return False


def precompute_split(enc: Evo2PerBaseEncoder, jsonl_path: Path, out_path: Path, args) -> None:
    if out_path.exists() and not args.overwrite and h5_complete(out_path, args):
        print(json.dumps({"event": "split_skip", "split": jsonl_path.name, "out": str(out_path)}), flush=True)
        return

    records = list(iter_jsonl(jsonl_path))
    total = len(records)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    if out_path.exists() and args.overwrite:
        out_path.unlink()

    t0 = time.time()
    print(json.dumps({"event": "split_start", "split": jsonl_path.stem, "n_records": total}), flush=True)
    with h5py.File(tmp, "w") as h5:
        h5.attrs["schema"] = "bend13"
        h5.attrs["num_classes"] = 13
        h5.attrs["encoder"] = "evo2_char_per_base"
        h5.attrs["model_name"] = args.model_name
        h5.attrs["layer_name"] = args.layer_name
        h5.attrs["source_hidden_size"] = enc.source_hidden_size
        h5.attrs["hidden_size"] = enc.hidden_size
        h5.attrs["project_dim"] = args.project_dim
        h5.attrs["projection_seed"] = args.projection_seed
        h5.attrs["target_chunk_bp"] = args.target_chunk_bp
        h5.attrs["dtype"] = args.storage_dtype
        h5.attrs["storage_dtype"] = args.storage_dtype
        h5.attrs["source"] = str(jsonl_path)
        h5.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        emb_grp = h5.create_group("embeddings")
        lab_grp = h5.create_group("labels")
        meta_grp = h5.create_group("meta")
        lengths, ids = [], []
        done, chunks_done = 0, 0
        for b0 in range(0, total, args.record_batch):
            batch = records[b0:b0 + args.record_batch]
            seqs = [seq for _, _, seq, _ in batch]
            embs, n_chunks = enc.encode_records(seqs)
            chunks_done += n_chunks
            for (idx, rec, seq, labels), emb in zip(batch, embs):
                if emb.shape[0] != labels.shape[0]:
                    raise RuntimeError(f"{rec['id']}: emb len {emb.shape[0]} != label len {labels.shape[0]}")
                emb_grp.create_dataset(str(idx), data=emb, compression=args.compression)
                lab_grp.create_dataset(str(idx), data=labels, compression=args.compression)
                lengths.append(int(labels.shape[0]))
                ids.append(rec["id"])
            done += len(batch)
            if args.log_every and (done % args.log_every == 0 or done == total):
                elapsed = time.time() - t0
                peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                print(json.dumps({
                    "event": "encode_progress",
                    "split": jsonl_path.stem,
                    "n": done,
                    "total": total,
                    "chunks": chunks_done,
                    "elapsed_sec": round(elapsed, 1),
                    "sec_per_record": round(elapsed / max(done, 1), 3),
                    "peak_vram_gb": round(peak, 2),
                }), flush=True)
        meta_grp.create_dataset("lengths", data=np.asarray(lengths, dtype=np.int32))
        meta_grp.create_dataset("ids", data=np.asarray(ids, dtype=h5py.string_dtype()))
        h5.attrs["num_records"] = done
        h5.attrs["complete"] = True
    tmp.rename(out_path)
    print(json.dumps({"event": "split_done", "out": str(out_path), "n": done, "elapsed_sec": round(time.time() - t0, 1)}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default="evo2_7b")
    ap.add_argument("--local-path", default=None, help="optional local Evo2 checkpoint path")
    ap.add_argument("--layer-name", default="blocks.20.mlp.l3")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--target-chunk-bp", type=int, default=14000)
    ap.add_argument("--gpu-batch", type=int, default=1)
    ap.add_argument("--record-batch", type=int, default=1)
    ap.add_argument("--storage-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--project-dim", type=int, default=0, help="0 stores raw Evo2 hidden states")
    ap.add_argument("--projection-seed", type=int, default=20260526)
    ap.add_argument("--compression", default="lzf")
    ap.add_argument("--log-every", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    enc = Evo2PerBaseEncoder(
        model_name=args.model_name,
        local_path=args.local_path,
        layer_name=args.layer_name,
        device=device,
        target_chunk_bp=args.target_chunk_bp,
        gpu_batch=args.gpu_batch,
        storage_dtype=args.storage_dtype,
        project_dim=args.project_dim,
        projection_seed=args.projection_seed,
    )
    print(json.dumps({
        "event": "encoder_ready",
        "model": args.model_name,
        "layer_name": args.layer_name,
        "source_hidden_size": enc.source_hidden_size,
        "hidden_size": enc.hidden_size,
        "project_dim": args.project_dim,
        "device": str(device),
        "target_chunk_bp": args.target_chunk_bp,
        "max_seqlen": enc.max_seqlen,
        "storage_dtype": args.storage_dtype,
    }), flush=True)
    for split in args.splits:
        precompute_split(
            enc,
            Path(args.data_dir) / f"{split}.jsonl.gz",
            Path(args.output_dir) / f"{split}.h5",
            args,
        )


if __name__ == "__main__":
    main()
