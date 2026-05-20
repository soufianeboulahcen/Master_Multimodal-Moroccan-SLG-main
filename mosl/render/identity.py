"""Phase C — face identity extraction & preservation (identity.py).

Encodes a person's identity from a folder of photos into a reusable embedding
that Phase D feeds to a diffusion pipeline (InstantID / IP-Adapter FaceID) so
the SAME person stays consistent across every generated pose and frame.

Scope (deliberately narrow)
---------------------------
Phase C does ONE thing: turn target photos into an identity embedding + face
crops + metadata. It does not touch pose extraction (Phase B) and does not run
diffusion (Phase D). Identity and pose are orthogonal inputs that meet only at
Phase D, linked by `identity_id`.

Encoder
-------
Faces are encoded with **insightface** (ArcFace, `antelopev2` pack) — the exact
recognition model both InstantID and IP-Adapter FaceID consume, so the 512-d
embedding produced here drops straight into either pipeline.

For IP-Adapter FaceID *Plus*, a CLIP image embedding is also needed. Phase C
saves the aligned face crop; Phase D computes the CLIP embedding from that crop
using the encoder that ships with the adapter. This keeps Phase C free of the
heavy CLIP/diffusers dependency.

Output layout (under <out_root>, default outputs/identity/)
    identity_embeddings/<identity_id>.npz     fused + per-image ArcFace embeddings
    identity_metadata/<identity_id>.json      identity metadata + QC metrics
    aligned/<identity_id>/                    aligned crops, face crops, viz
    aligned/<identity_id>/reference_face.png  best crop for Phase D

Dependencies (DGX):
    pip install insightface onnxruntime-gpu opencv-python pyyaml

NOTE: written for DGX execution; not run in this environment (no GPU /
insightface). The fusion / consistency / IO logic is unit-checked. Validate
the GPU path first with:  python -m mosl.render.identity --check

Usage
-----
    python -m mosl.render.identity --input photos/omar/
    python -m mosl.render.identity --input photos/omar/ --identity-id omar --viz
    python -m mosl.render.identity --config identity_config.yaml
    python -m mosl.render.identity --batch photos/          # all sub-folders
    python -m mosl.render.identity --write-default-config identity_config.yaml
    python -m mosl.render.identity --list                   # show saved identities
    python -m mosl.render.identity --check
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("mosl.render.identity")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EMBED_DIM = 512

__all__ = [
    "IdentityConfig", "load_config",
    "FaceRecord", "FusedIdentity",
    "FaceAnalyzer",
    "load_images", "detect_and_align_face", "extract_identity_features",
    "fuse_multi_image_identity", "save_embeddings", "load_embeddings",
    "process_batch", "list_identities", "visualize_alignment",
    "build_identity",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IdentityConfig:
    """All tunables for one identity-extraction run.

    Loaded from YAML or JSON (--config); CLI flags override the file.
    YAML is preferred on DGX; JSON is accepted for backward compatibility.
    """
    # I/O
    input_dir: str = ""
    identity_id: str = ""           # default: input folder name
    out_root: Path = ROOT / "outputs" / "identity"

    # insightface backend
    det_model: str = "antelopev2"   # InstantID default; buffalo_l is lighter
    model_root: str = "~/.insightface"
    det_size: int = 640
    device: str = "cuda"            # cuda | cpu

    # face crop geometry
    crop_size: int = 512            # square edge for the Phase D reference crop
    crop_margin: float = 0.40       # extra context around the face bbox

    # quality gates
    min_det_score: float = 0.50     # reject weak detections
    outlier_threshold: float = 0.50 # drop images below this cosine-to-mean
    consistency_warn: float = 0.55  # warn if mean pairwise cosine below this

    # selection strategy when multiple faces are in one image
    face_select: str = "largest"    # largest | most_confident

    # output options
    save_viz: bool = False          # write detection-overlay images
    save_aligned_grid: bool = False # write a contact-sheet of all aligned crops

    def __post_init__(self) -> None:
        self.out_root = Path(self.out_root).expanduser()
        if not self.identity_id and self.input_dir:
            self.identity_id = Path(self.input_dir).resolve().name

    # Directories that Phase D and downstream code read from.
    @property
    def embeddings_dir(self) -> Path:
        return self.out_root / "identity_embeddings"

    @property
    def metadata_dir(self) -> Path:
        return self.out_root / "identity_metadata"

    @property
    def aligned_dir(self) -> Path:
        return self.out_root / "aligned" / self.identity_id


def _load_yaml_or_json(path: Path) -> dict:
    """Load a YAML or JSON config file. YAML requires pyyaml."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # noqa: PLC0415
            return yaml.safe_load(text) or {}
        except ImportError as exc:
            raise RuntimeError(
                "pyyaml is required for YAML configs: pip install pyyaml"
            ) from exc
    return json.loads(text)


def load_config(path: Path | None, args: argparse.Namespace) -> IdentityConfig:
    """Resolve config with precedence: CLI flags > config file > dataclass defaults."""
    data: dict = {}
    if path is not None:
        if not path.is_file():
            raise SystemExit(f"config file not found: {path}")
        data.update(_load_yaml_or_json(path))

    if getattr(args, "input", None):
        data["input_dir"] = args.input

    # Only apply CLI overrides that were explicitly provided (not None).
    cli_map = {
        "identity_id": getattr(args, "identity_id", None),
        "out_root":    getattr(args, "out_root", None),
        "det_model":   getattr(args, "det_model", None),
        "device":      getattr(args, "device", None),
        "crop_size":   getattr(args, "crop_size", None),
        "save_viz":    True if getattr(args, "viz", False) else None,
    }
    data.update({k: v for k, v in cli_map.items() if v is not None})

    valid = {f.name for f in fields(IdentityConfig)}
    return IdentityConfig(**{k: v for k, v in data.items() if k in valid})


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class FaceRecord:
    """One detected face and its identity features."""
    image_name: str
    bbox: np.ndarray            # (4,)  x1,y1,x2,y2
    kps: np.ndarray             # (5,2) five-point landmarks
    det_score: float
    embedding: np.ndarray       # (512,) raw ArcFace embedding
    normed_embedding: np.ndarray  # (512,) L2-normalized
    aligned_crop: np.ndarray    # (112,112,3) ArcFace-aligned, BGR
    face_crop: np.ndarray       # (crop_size,crop_size,3) padded crop, BGR
    detection_viz: np.ndarray | None = None  # source image w/ bbox+kps overlay


@dataclass
class FusedIdentity:
    """The fused identity built from one or more FaceRecords."""
    identity_id: str
    fused: np.ndarray           # (512,) normalized — the primary identity embedding
    fused_raw: np.ndarray       # (512,) fused raw (un-normalized) embeddings
    per_image: np.ndarray       # (N,512) normalized per-image embeddings
    used: list[bool]            # which images survived outlier filtering
    sims_to_fused: list[float]  # per-image cosine similarity to the fused embedding
    reference_kps: np.ndarray   # (5,2) kps of the highest-confidence image


# ---------------------------------------------------------------------------
# insightface backend
# ---------------------------------------------------------------------------

class FaceAnalyzer:
    """Thin wrapper over insightface FaceAnalysis.

    Isolated so the import error message and provider selection live in one
    place and the rest of the module stays testable without a GPU.
    """

    def __init__(self, cfg: IdentityConfig) -> None:
        try:
            from insightface.app import FaceAnalysis  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "insightface is not installed. On the DGX run:\n"
                "    pip install insightface onnxruntime-gpu opencv-python"
            ) from exc

        if cfg.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        log.info("loading insightface '%s' (device=%s)", cfg.det_model, cfg.device)
        self._app = FaceAnalysis(
            name=cfg.det_model,
            root=str(Path(cfg.model_root).expanduser()),
            providers=providers,
        )
        self._app.prepare(ctx_id=ctx_id, det_size=(cfg.det_size, cfg.det_size))

    def get(self, image_bgr: np.ndarray) -> list:
        """Run detection + recognition on one BGR image."""
        return self._app.get(image_bgr)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_images(input_dir: Path) -> list[tuple[str, np.ndarray]]:
    """Load every supported image in a folder as (filename, BGR ndarray).

    Args:
        input_dir: Folder containing face photos.

    Returns:
        List of (filename, BGR image array) tuples, sorted by filename.

    Raises:
        RuntimeError: If the folder is missing or contains no readable images.
    """
    import cv2  # noqa: PLC0415

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise RuntimeError(f"input folder not found: {input_dir}")

    files = sorted(p for p in input_dir.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise RuntimeError(
            f"no images ({sorted(IMAGE_EXTS)}) found in {input_dir}"
        )

    images: list[tuple[str, np.ndarray]] = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            log.warning("unreadable image skipped: %s", p.name)
            continue
        images.append((p.name, img))

    if not images:
        raise RuntimeError(f"no readable images in {input_dir}")

    log.info("loaded %d image(s) from %s", len(images), input_dir)
    return images


def _square_face_crop(
    img: np.ndarray, bbox: np.ndarray, size: int, margin: float
) -> np.ndarray:
    """Square, edge-padded crop centred on the face bbox, resized to `size`."""
    import cv2  # noqa: PLC0415

    h, w = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) * (0.5 + margin)
    xa, ya = int(round(cx - half)), int(round(cy - half))
    xb, yb = int(round(cx + half)), int(round(cy + half))
    pad_l = max(0, -xa)
    pad_t = max(0, -ya)
    pad_r = max(0, xb - w)
    pad_b = max(0, yb - h)
    crop = img[max(0, ya):min(h, yb), max(0, xa):min(w, xb)]
    if crop.size == 0:
        crop = img
    if pad_l or pad_t or pad_r or pad_b:
        crop = np.pad(crop, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def detect_and_align_face(
    name: str,
    img: np.ndarray,
    analyzer: FaceAnalyzer,
    cfg: IdentityConfig,
) -> FaceRecord | None:
    """Detect, select, align, and ArcFace-encode the target face in one image.

    When multiple faces are present, the selection strategy is controlled by
    ``cfg.face_select`` (``"largest"`` or ``"most_confident"``).

    Args:
        name: Image filename (used only for log messages).
        img:  BGR image array.
        analyzer: Loaded FaceAnalyzer instance.
        cfg:  Active IdentityConfig.

    Returns:
        FaceRecord on success, None if no usable face was found.
    """
    from insightface.utils import face_align  # noqa: PLC0415

    faces = [
        f for f in analyzer.get(img)
        if float(getattr(f, "det_score", 0.0)) >= cfg.min_det_score
    ]
    if not faces:
        log.warning("%s: no face above det_score %.2f", name, cfg.min_det_score)
        return None

    if cfg.face_select == "most_confident":
        face = max(faces, key=lambda f: float(f.det_score))
    else:  # largest (default)
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

    if len(faces) > 1:
        log.info(
            "%s: %d faces found, kept the %s one",
            name, len(faces), cfg.face_select,
        )

    emb = np.asarray(getattr(face, "embedding", None), dtype=np.float32)
    if emb is None or emb.size != EMBED_DIM:
        log.warning(
            "%s: detector pack '%s' has no recognition model — "
            "use antelopev2 or buffalo_l",
            name, cfg.det_model,
        )
        return None

    normed = np.asarray(face.normed_embedding, dtype=np.float32)
    # ArcFace-aligned 112×112 crop — the canonical input for InstantID
    aligned = face_align.norm_crop(img, landmark=face.kps, image_size=112)
    # Larger padded crop for IP-Adapter FaceID Plus CLIP encoding (Phase D)
    crop = _square_face_crop(img, face.bbox, cfg.crop_size, cfg.crop_margin)
    viz = _draw_detection(img, face.bbox, face.kps) if cfg.save_viz else None

    return FaceRecord(
        image_name=name,
        bbox=np.asarray(face.bbox, dtype=np.float32),
        kps=np.asarray(face.kps, dtype=np.float32),
        det_score=float(face.det_score),
        embedding=emb,
        normed_embedding=normed,
        aligned_crop=aligned,
        face_crop=crop,
        detection_viz=viz,
    )


def extract_identity_features(
    images: list[tuple[str, np.ndarray]],
    analyzer: FaceAnalyzer,
    cfg: IdentityConfig,
) -> list[FaceRecord]:
    """Batch: run detect_and_align_face over every input image.

    One bad image never aborts the batch — it is logged and skipped.

    Args:
        images:   Output of load_images().
        analyzer: Loaded FaceAnalyzer instance.
        cfg:      Active IdentityConfig.

    Returns:
        List of FaceRecord for every image where a face was successfully encoded.
    """
    records: list[FaceRecord] = []
    for name, img in images:
        try:
            rec = detect_and_align_face(name, img, analyzer, cfg)
        except Exception as exc:  # noqa: BLE001
            log.error("%s: face extraction failed: %s", name, exc)
            rec = None
        if rec is not None:
            records.append(rec)

    log.info("encoded %d/%d image(s)", len(records), len(images))
    return records


def fuse_multi_image_identity(
    records: list[FaceRecord],
    cfg: IdentityConfig,
) -> FusedIdentity:
    """Fuse per-image embeddings into one stable identity embedding.

    Strategy:
    1. Compute a provisional mean of all normalized embeddings.
    2. Drop images whose cosine similarity to the provisional mean is below
       ``cfg.outlier_threshold`` (different person, bad crop, heavy occlusion).
    3. Re-average the survivors and L2-normalize the result.

    If every image is flagged as an outlier (e.g. threshold set too high),
    the single image with the highest similarity to the provisional mean is
    kept so the function never returns an empty identity.

    Args:
        records: Output of extract_identity_features().
        cfg:     Active IdentityConfig.

    Returns:
        FusedIdentity with the normalized fused embedding and QC fields.

    Raises:
        RuntimeError: If records is empty.
    """
    if not records:
        raise RuntimeError("no faces were encoded — cannot build an identity")

    per_image = np.stack([r.normed_embedding for r in records])  # (N, 512)
    raw = np.stack([r.embedding for r in records])               # (N, 512)

    # Provisional mean for outlier detection
    provisional = per_image.mean(axis=0)
    provisional /= np.linalg.norm(provisional) + 1e-9
    sims_to_mean = per_image @ provisional  # (N,)

    used = [bool(s >= cfg.outlier_threshold) for s in sims_to_mean]
    if not any(used):
        # All flagged — keep the single best rather than fail
        best_idx = int(np.argmax(sims_to_mean))
        used[best_idx] = True

    for rec, keep, sim in zip(records, used, sims_to_mean):
        if not keep:
            log.warning(
                "dropped %s from fusion (cosine-to-mean %.3f < %.2f)",
                rec.image_name, sim, cfg.outlier_threshold,
            )

    keep_idx = [i for i, k in enumerate(used) if k]
    fused = per_image[keep_idx].mean(axis=0)
    fused /= np.linalg.norm(fused) + 1e-9
    fused_raw = raw[keep_idx].mean(axis=0)

    sims_to_fused = [float(e @ fused) for e in per_image]
    best_rec = records[int(np.argmax([r.det_score for r in records]))]

    log.info("fused identity from %d/%d image(s)", len(keep_idx), len(records))
    return FusedIdentity(
        identity_id=cfg.identity_id,
        fused=fused.astype(np.float32),
        fused_raw=fused_raw.astype(np.float32),
        per_image=per_image.astype(np.float32),
        used=used,
        sims_to_fused=sims_to_fused,
        reference_kps=best_rec.kps,
    )


def _consistency(per_image: np.ndarray) -> dict:
    """Mean and min pairwise cosine similarity — a QC signal for identity stability.

    Values close to 1.0 mean all photos show the same person in similar
    conditions. Values below ~0.5 suggest mixed identities or heavy variation.
    """
    n = len(per_image)
    if n < 2:
        return {"mean_pairwise_cosine": 1.0, "min_pairwise_cosine": 1.0, "n_images": n}
    sims = [
        float(per_image[i] @ per_image[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return {
        "mean_pairwise_cosine": round(float(np.mean(sims)), 4),
        "min_pairwise_cosine": round(float(np.min(sims)), 4),
        "n_images": n,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_embeddings(
    fused: FusedIdentity,
    records: list[FaceRecord],
    cfg: IdentityConfig,
) -> dict:
    """Write the embedding NPZ, metadata JSON, and face crops to disk.

    Output directories follow the spec:
        <out_root>/identity_embeddings/<identity_id>.npz
        <out_root>/identity_metadata/<identity_id>.json
        <out_root>/aligned/<identity_id>/

    Args:
        fused:   Output of fuse_multi_image_identity().
        records: Output of extract_identity_features().
        cfg:     Active IdentityConfig.

    Returns:
        The metadata dict that was written to JSON (useful for logging / tests).
    """
    import cv2  # noqa: PLC0415

    for d in (cfg.embeddings_dir, cfg.metadata_dir, cfg.aligned_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- embedding NPZ -------------------------------------------------------
    emb_path = cfg.embeddings_dir / f"{cfg.identity_id}.npz"
    np.savez(
        emb_path,
        fused=fused.fused,
        fused_raw=fused.fused_raw,
        per_image=fused.per_image,
        reference_kps=fused.reference_kps,
    )

    # --- face crops ----------------------------------------------------------
    best_i = int(np.argmax([r.det_score for r in records]))
    for i, rec in enumerate(records):
        stem = Path(rec.image_name).stem
        cv2.imwrite(
            str(cfg.aligned_dir / f"{stem}_aligned112.png"), rec.aligned_crop
        )
        cv2.imwrite(
            str(cfg.aligned_dir / f"{stem}_facecrop.png"), rec.face_crop
        )
        if i == best_i:
            # reference_face.png is the canonical input for Phase D
            cv2.imwrite(
                str(cfg.aligned_dir / "reference_face.png"), rec.face_crop
            )
        if cfg.save_viz and rec.detection_viz is not None:
            cv2.imwrite(
                str(cfg.aligned_dir / f"{stem}_detect.png"), rec.detection_viz
            )

    # --- optional aligned-crop contact sheet ---------------------------------
    if cfg.save_aligned_grid and records:
        _save_aligned_grid(records, cfg.aligned_dir)

    # --- QC ------------------------------------------------------------------
    consistency = _consistency(fused.per_image)
    if consistency["mean_pairwise_cosine"] < cfg.consistency_warn:
        log.warning(
            "LOW identity consistency (mean pairwise cosine %.3f < %.2f) — "
            "photos may show different people or heavy pose variation; "
            "expect weaker identity preservation downstream",
            consistency["mean_pairwise_cosine"], cfg.consistency_warn,
        )

    # --- metadata JSON -------------------------------------------------------
    metadata = {
        "identity_id": cfg.identity_id,
        "created": date.today().isoformat(),
        "input_dir": cfg.input_dir,
        "n_images": len(records),
        "n_used_in_fusion": int(sum(fused.used)),
        "embedding_dim": EMBED_DIM,
        "detector": f"insightface/{cfg.det_model}",
        "det_size": [cfg.det_size, cfg.det_size],
        "fusion": {
            "method": "mean of L2-normalized ArcFace embeddings, outlier-filtered",
            "outlier_threshold": cfg.outlier_threshold,
        },
        "consistency": consistency,
        "images": [
            {
                "name": r.image_name,
                "det_score": round(r.det_score, 4),
                "bbox": [round(float(v), 1) for v in r.bbox],
                "cosine_to_fused": round(fused.sims_to_fused[i], 4),
                "used": fused.used[i],
            }
            for i, r in enumerate(records)
        ],
        "reference_kps": fused.reference_kps.tolist(),
        "files": {
            "embeddings": str(emb_path.relative_to(cfg.out_root)),
            "crops_dir": str(cfg.aligned_dir.relative_to(cfg.out_root)),
            "reference_face": f"aligned/{cfg.identity_id}/reference_face.png",
        },
        # Downstream usage notes — consumed by Phase D to select the right path
        "usage": {
            "instantid": (
                "Load `fused` (512-d normalized) as the face embedding. "
                "Use `reference_kps` to build the 5-point landmark image for "
                "IdentityNet. Per-frame face control comes from Phase B DWPose, "
                "not from this embedding."
            ),
            "ip_adapter_faceid": (
                "Use `fused` (normalized) as `faceid_embeds`. "
                "Compute the FaceID-Plus CLIP embedding from `reference_face.png` "
                "in Phase D using the adapter's own image encoder."
            ),
            "flux_pulid": (
                "PuLID-FLUX accepts the same ArcFace embedding. "
                "Pass `fused` directly; no CLIP embedding needed."
            ),
        },
        "integration": {
            "phase_b": "DWPose keypoints linked at Phase D by identity_id",
            "phase_d": "SDXL/FLUX via InstantID or IP-Adapter FaceID",
            "phase_e": "MimicMotion / AnimateDiff video generation",
        },
    }

    meta_path = cfg.metadata_dir / f"{cfg.identity_id}.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("saved identity '%s' -> %s", cfg.identity_id, emb_path)
    return metadata


def load_embeddings(
    identity_id: str,
    out_root: Path,
    as_torch: bool = False,
) -> dict:
    """Load a saved identity for Phase D.

    Args:
        identity_id: The identity name used when save_embeddings() was called.
        out_root:    Same root used during extraction.
        as_torch:    If True, return torch.Tensor instead of numpy arrays.
                     InstantID and IP-Adapter ingest tensors directly.

    Returns:
        Dict with keys: identity_id, fused, fused_raw, per_image,
        reference_kps, metadata.

    Raises:
        FileNotFoundError: If the NPZ file does not exist.
    """
    out_root = Path(out_root).expanduser()
    emb_path = out_root / "identity_embeddings" / f"{identity_id}.npz"
    meta_path = out_root / "identity_metadata" / f"{identity_id}.json"

    if not emb_path.is_file():
        raise FileNotFoundError(f"no identity embedding at {emb_path}")

    npz = np.load(emb_path)
    result: dict = {
        "identity_id": identity_id,
        "fused": npz["fused"],
        "fused_raw": npz["fused_raw"],
        "per_image": npz["per_image"],
        "reference_kps": npz["reference_kps"],
        "metadata": (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.is_file()
            else {}
        ),
        # Convenience: path to the reference face crop for Phase D CLIP encoding
        "reference_face_path": (
            out_root / "aligned" / identity_id / "reference_face.png"
        ),
    }

    if as_torch:
        import torch  # noqa: PLC0415
        for k in ("fused", "fused_raw", "per_image", "reference_kps"):
            result[k] = torch.from_numpy(np.ascontiguousarray(result[k]))

    return result


# ---------------------------------------------------------------------------
# Batch processing & discovery
# ---------------------------------------------------------------------------

def process_batch(
    batch_root: Path,
    cfg: IdentityConfig,
    *,
    skip_existing: bool = True,
) -> list[dict]:
    """Process every sub-folder of batch_root as a separate identity.

    Each sub-folder name becomes the identity_id. Useful when you have a
    directory of per-person photo folders:

        photos/
            omar/   photo1.jpg  photo2.jpg
            fatima/ photo1.jpg

    Args:
        batch_root:     Directory whose immediate sub-folders are identities.
        cfg:            Base config; input_dir and identity_id are overridden
                        per-folder.
        skip_existing:  Skip folders that already have a saved embedding.

    Returns:
        List of metadata dicts for every successfully processed identity.
    """
    batch_root = Path(batch_root)
    if not batch_root.is_dir():
        raise RuntimeError(f"batch root not found: {batch_root}")

    subdirs = sorted(p for p in batch_root.iterdir() if p.is_dir())
    if not subdirs:
        raise RuntimeError(f"no sub-folders found in {batch_root}")

    log.info("batch: found %d identity folder(s) in %s", len(subdirs), batch_root)
    results: list[dict] = []
    analyzer: FaceAnalyzer | None = None  # shared across identities

    for folder in subdirs:
        identity_id = folder.name
        emb_path = cfg.out_root / "identity_embeddings" / f"{identity_id}.npz"
        if skip_existing and emb_path.is_file():
            log.info("skipping '%s' — embedding already exists", identity_id)
            continue

        # Clone config with per-identity overrides
        import dataclasses  # noqa: PLC0415
        per_cfg = dataclasses.replace(
            cfg, input_dir=str(folder), identity_id=identity_id
        )

        try:
            if analyzer is None:
                analyzer = FaceAnalyzer(per_cfg)
            images = load_images(folder)
            records = extract_identity_features(images, analyzer, per_cfg)
            fused = fuse_multi_image_identity(records, per_cfg)
            metadata = save_embeddings(fused, records, per_cfg)
            results.append(metadata)
            log.info("batch: processed '%s'", identity_id)
        except Exception as exc:  # noqa: BLE001
            log.error("batch: FAILED '%s': %s", identity_id, exc)

    log.info("batch: done — %d/%d identities processed", len(results), len(subdirs))
    return results


def list_identities(out_root: Path) -> list[dict]:
    """Return a summary of all saved identities under out_root.

    Reads the metadata JSON for each saved identity and returns a list of
    lightweight summary dicts. Useful for Phase D to discover available
    identities without loading the full NPZ files.

    Args:
        out_root: Same root used during extraction.

    Returns:
        List of dicts with keys: identity_id, n_images, n_used_in_fusion,
        consistency, created, files.
    """
    out_root = Path(out_root).expanduser()
    meta_dir = out_root / "identity_metadata"
    if not meta_dir.is_dir():
        return []

    summaries: list[dict] = []
    for meta_path in sorted(meta_dir.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            summaries.append({
                "identity_id": meta.get("identity_id", meta_path.stem),
                "n_images": meta.get("n_images", 0),
                "n_used_in_fusion": meta.get("n_used_in_fusion", 0),
                "consistency": meta.get("consistency", {}),
                "created": meta.get("created", ""),
                "files": meta.get("files", {}),
            })
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read metadata %s: %s", meta_path.name, exc)

    return summaries


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _draw_detection(
    img: np.ndarray, bbox: np.ndarray, kps: np.ndarray
) -> np.ndarray:
    """Overlay the face bbox (green) and five landmarks (red) on the source image."""
    import cv2  # noqa: PLC0415

    viz = img.copy()
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for kx, ky in kps:
        cv2.circle(
            viz, (int(round(float(kx))), int(round(float(ky)))), 3, (0, 0, 255), -1
        )
    return viz


def _save_aligned_grid(records: list[FaceRecord], out_dir: Path) -> None:
    """Write a contact sheet of all 112×112 aligned crops side-by-side."""
    import cv2  # noqa: PLC0415

    crops = [r.aligned_crop for r in records]
    n = len(crops)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    h, w = 112, 112
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, crop in enumerate(crops):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = crop
    cv2.imwrite(str(out_dir / "aligned_grid.png"), grid)


def visualize_alignment(
    records: list[FaceRecord],
    out_dir: Path,
    *,
    show: bool = False,
) -> Path:
    """Write a visualization of face alignment results for QC.

    Produces a side-by-side grid: original crop (with bbox/kps overlay) next
    to the 112×112 ArcFace-aligned crop. One row per image.

    Args:
        records: Output of extract_identity_features().
        out_dir: Directory to write the visualization PNG.
        show:    If True, attempt to display with cv2.imshow (DGX: use False).

    Returns:
        Path to the written PNG.
    """
    import cv2  # noqa: PLC0415

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_h, cell_w = 224, 224
    n = len(records)
    canvas = np.zeros((n * cell_h, cell_w * 2, 3), dtype=np.uint8)

    for i, rec in enumerate(records):
        y0 = i * cell_h

        # Left: face crop (resized to cell)
        left = cv2.resize(rec.face_crop, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        # Overlay det_score text
        cv2.putText(
            left, f"{rec.image_name}  score={rec.det_score:.2f}",
            (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
        )
        canvas[y0:y0 + cell_h, :cell_w] = left

        # Right: ArcFace-aligned 112×112 crop, upscaled
        right = cv2.resize(rec.aligned_crop, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            right, "ArcFace aligned 112px",
            (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1,
        )
        canvas[y0:y0 + cell_h, cell_w:] = right

    out_path = out_dir / "alignment_viz.png"
    cv2.imwrite(str(out_path), canvas)
    log.info("alignment visualization -> %s", out_path)

    if show:
        cv2.imshow("Phase C — alignment QC", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_identity(cfg: IdentityConfig) -> dict:
    """Full Phase C pipeline for one identity folder.

    Runs: load_images -> FaceAnalyzer -> extract_identity_features ->
          fuse_multi_image_identity -> save_embeddings.

    Args:
        cfg: Fully populated IdentityConfig.

    Returns:
        The metadata dict written to identity_metadata/<identity_id>.json.
    """
    if not cfg.input_dir:
        raise SystemExit("no --input folder given")

    t0 = time.time()
    images = load_images(Path(cfg.input_dir).expanduser())
    analyzer = FaceAnalyzer(cfg)
    records = extract_identity_features(images, analyzer, cfg)
    fused = fuse_multi_image_identity(records, cfg)
    metadata = save_embeddings(fused, records, cfg)

    if cfg.save_viz:
        visualize_alignment(records, cfg.aligned_dir)

    log.info(
        "identity '%s' built in %.1fs — embedding: %s",
        cfg.identity_id, time.time() - t0,
        cfg.embeddings_dir / f"{cfg.identity_id}.npz",
    )
    return metadata


def _run_check(cfg: IdentityConfig) -> int:
    """Validate that the insightface backend loads and runs — no data needed."""
    log.info("environment check: loading insightface backend ...")
    try:
        analyzer = FaceAnalyzer(cfg)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    faces = analyzer.get(np.zeros((640, 640, 3), dtype=np.uint8))
    log.info("backend OK — inference ran (faces on blank frame: %d)", len(faces))
    return 0


def _write_default_config(path: Path) -> int:
    """Write a YAML config template to path."""
    path = Path(path)
    suffix = path.suffix.lower()
    cfg = IdentityConfig()

    if suffix in {".yaml", ".yml"}:
        lines = [
            "# Phase C — identity extraction config",
            "# All fields are optional; CLI flags override this file.",
            "",
            f"input_dir: photos/<person>/          # folder of face photos",
            f"identity_id: ''                       # default: folder name",
            f"out_root: {cfg.out_root}",
            "",
            f"det_model: {cfg.det_model}            # antelopev2 | buffalo_l",
            f"model_root: {cfg.model_root}",
            f"det_size: {cfg.det_size}",
            f"device: {cfg.device}                  # cuda | cpu",
            "",
            f"crop_size: {cfg.crop_size}",
            f"crop_margin: {cfg.crop_margin}",
            "",
            f"min_det_score: {cfg.min_det_score}",
            f"outlier_threshold: {cfg.outlier_threshold}",
            f"consistency_warn: {cfg.consistency_warn}",
            f"face_select: {cfg.face_select}        # largest | most_confident",
            "",
            f"save_viz: false",
            f"save_aligned_grid: false",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        # JSON fallback
        template = {
            "input_dir": "photos/<person>/",
            "identity_id": "",
            "out_root": str(cfg.out_root),
            "det_model": cfg.det_model,
            "model_root": cfg.model_root,
            "det_size": cfg.det_size,
            "device": cfg.device,
            "crop_size": cfg.crop_size,
            "crop_margin": cfg.crop_margin,
            "min_det_score": cfg.min_det_score,
            "outlier_threshold": cfg.outlier_threshold,
            "consistency_warn": cfg.consistency_warn,
            "face_select": cfg.face_select,
            "save_viz": False,
            "save_aligned_grid": False,
        }
        path.write_text(json.dumps(template, indent=2), encoding="utf-8")

    log.info("wrote default config template -> %s", path)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", metavar="DIR",
                    help="folder of target face photos (single identity)")
    ap.add_argument("--batch", metavar="DIR",
                    help="root folder; each sub-folder is one identity")
    ap.add_argument("--identity-id",
                    help="identity name (default: input folder name)")
    ap.add_argument("--out-root",
                    help="output root (default: outputs/identity)")
    ap.add_argument("--config", metavar="PATH",
                    help="YAML or JSON config file (CLI flags override it)")
    ap.add_argument("--det-model", choices=["antelopev2", "buffalo_l"],
                    help="insightface model pack")
    ap.add_argument("--device", choices=["cuda", "cpu"],
                    help="inference device")
    ap.add_argument("--crop-size", type=int,
                    help="square face-crop edge (pixels)")
    ap.add_argument("--viz", action="store_true",
                    help="save detection-overlay and alignment visualization images")
    ap.add_argument("--list", action="store_true",
                    help="list saved identities and exit")
    ap.add_argument("--check", action="store_true",
                    help="validate the insightface environment and exit")
    ap.add_argument("--write-default-config", metavar="PATH",
                    help="write a config template (YAML or JSON) to PATH and exit")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.write_default_config:
        return _write_default_config(Path(args.write_default_config))

    cfg = load_config(Path(args.config) if args.config else None, args)

    if args.list:
        identities = list_identities(cfg.out_root)
        if not identities:
            print(f"no identities found under {cfg.out_root}")
        else:
            print(json.dumps(identities, indent=2, ensure_ascii=False))
        return 0

    if args.check:
        return _run_check(cfg)

    if args.batch:
        try:
            results = process_batch(Path(args.batch), cfg)
            print(f"batch complete: {len(results)} identities processed")
            return 0
        except (RuntimeError, FileNotFoundError) as exc:
            log.error("%s", exc)
            return 1

    if not cfg.input_dir:
        ap.error(
            "no input — pass --input <folder>, --batch <root>, "
            "--list, --check, or --config"
        )

    try:
        build_identity(cfg)
    except (RuntimeError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
