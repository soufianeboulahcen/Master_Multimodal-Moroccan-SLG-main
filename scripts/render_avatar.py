"""Phase G — end-to-end avatar video generation.

One command: identity photos + an Arabic sign (or a driving video) → a finished,
identity-preserving photorealistic avatar video. Orchestrates the `mosl/render/`
phases C → D → E → F; each stage is skipped if its output already exists
(unless --force).

    photos ─► [C identity] ─► [D keyframe] ─┐
                                            ├─► [E MimicMotion] ─► [F polish] ─► MP4
    Arabic word / sign video ───────────────┘

Text resolution
---------------
`--text <Arabic word>` is looked up in `data/labels.csv` (NFC-normalized) to
find the matching MoSL sign clip, which then drives MimicMotion. Per
`docs/RESULTS.md`, retrieving the real clip beats the SignLLM model's generated
pose, so retrieval is the motion source here.

NOTE: needs the DGX render environment (see mosl/render/SETUP_DGX.md) and the
MoSL dataset present under data/raw/ for --text lookups.

Usage
-----
    # first run for a person — build identity from photos, then a sign by text
    python scripts/render_avatar.py --photos photos/omar/ --text "الأذان"

    # identity already built — reuse it, drive with an explicit video
    python scripts/render_avatar.py --identity-id omar --driving-video clip.mp4

    python scripts/render_avatar.py --identity-id omar --text "سلام" --upscale
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosl.render.identity import IdentityConfig, build_identity  # noqa: E402
from mosl.render.keyframe import KeyframeConfig, generate_keyframe  # noqa: E402
from mosl.render.temporal import TemporalConfig, polish_video  # noqa: E402
from mosl.render.video import VideoConfig, render_video  # noqa: E402

log = logging.getLogger("render_avatar")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def resolve_text_to_clip(word: str) -> Path:
    """Map an Arabic word to its MoSL sign clip via data/labels.csv."""
    labels = ROOT / "data" / "labels.csv"
    if not labels.is_file():
        raise SystemExit(f"label table not found: {labels}")
    target = _nfc(word)
    exact, stripped = [], []
    with open(labels, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if _nfc(row["word_arabic"]) == target:
                exact.append(row)
            elif _nfc(row.get("word_arabic_stripped", "")) == target:
                stripped.append(row)
    rows = exact or stripped
    if not rows:
        raise SystemExit(f"no MoSL sign found for text: {word!r}")
    if exact and len(exact) > 1:
        log.warning("%d clips match %r — using the first (variant %s)",
                    len(exact), word, rows[0].get("variant") or "—")
    if not exact and stripped:
        log.warning("no diacritic-exact match for %r — fell back to a "
                    "stripped-form match (may be a different sign)", word)
    clip = ROOT / rows[0]["relative_path"]
    if not clip.is_file():
        raise SystemExit(
            f"matched clip is not on disk: {clip}\n"
            "    the MoSL dataset must be present under data/raw/")
    return clip


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--identity-id", help="identity name (default: photos folder name)")
    ap.add_argument("--photos", help="folder of target photos (builds the identity)")
    motion = ap.add_mutually_exclusive_group(required=True)
    motion.add_argument("--text", help="Arabic sign word — looked up in labels.csv")
    motion.add_argument("--driving-video", help="an explicit driving sign clip")
    ap.add_argument("--backend", choices=["mimicmotion", "animatediff"],
                    default="mimicmotion",
                    help="video generation backend (default: mimicmotion)")
    ap.add_argument("--upscale", action="store_true", help="enable Phase F upscaling")
    ap.add_argument("--interp-backend", choices=["ffmpeg", "rife"], default="ffmpeg",
                    help="frame interpolation backend for Phase F (default: ffmpeg)")
    ap.add_argument("--force", action="store_true",
                    help="re-run stages even if their output exists")
    ap.add_argument("--status", action="store_true",
                    help="show which stages are complete for this identity and exit")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    identity_id = args.identity_id or (Path(args.photos).name if args.photos else "")
    if not identity_id:
        ap.error("pass --identity-id or --photos")

    # ---- status check mode ----------------------------------------------
    if args.status:
        emb_path = ROOT / "outputs" / "identity" / "identity_embeddings" / f"{identity_id}.npz"
        kf_path  = ROOT / "outputs" / "keyframes" / identity_id / "keyframe.png"
        log.info("Status for identity '%s':", identity_id)
        log.info("  Phase C (identity embedding): %s", "✓" if emb_path.is_file() else "✗ missing")
        log.info("  Phase D (keyframe):           %s", "✓" if kf_path.is_file() else "✗ missing")
        return 0

    # ---- Stage 1/4 — identity -------------------------------------------
    emb = ROOT / "outputs" / "identity" / "identity_embeddings" / f"{identity_id}.npz"
    if args.photos and (args.force or not emb.is_file()):
        log.info("STAGE 1/4 — identity: encoding from %s", args.photos)
        build_identity(IdentityConfig(input_dir=args.photos,
                                      identity_id=identity_id))
    elif emb.is_file():
        log.info("STAGE 1/4 — identity: reusing %s", emb.name)
    else:
        ap.error(f"no identity '{identity_id}' — pass --photos to build it")

    # ---- Stage 2/4 — keyframe -------------------------------------------
    keyframe = ROOT / "outputs" / "keyframes" / identity_id / "keyframe.png"
    if args.force or not keyframe.is_file():
        log.info("STAGE 2/4 — keyframe: generating reference image")
        generate_keyframe(KeyframeConfig(identity_id=identity_id))
    else:
        log.info("STAGE 2/4 — keyframe: reusing %s", keyframe.name)

    # ---- resolve the driving motion -------------------------------------
    if args.text:
        driving = resolve_text_to_clip(args.text)
        log.info("text %r -> %s", args.text, driving.name)
    else:
        driving = Path(args.driving_video).expanduser()
        if not driving.is_file():
            ap.error(f"driving video not found: {driving}")

    # ---- Stage 3/4 — MimicMotion ----------------------------------------
    log.info("STAGE 3/4 — video: MimicMotion")
    v_manifest = render_video(VideoConfig(identity_id=identity_id), driving)
    raw = (ROOT / "outputs" / "avatar_video" / identity_id
           / v_manifest["output_video"])

    # ---- Stage 4/4 — temporal polish ------------------------------------
    log.info("STAGE 4/4 — temporal: deflicker + interpolate%s",
             " + upscale" if args.upscale else "")
    f_manifest = polish_video(TemporalConfig(upscale=args.upscale), raw)
    final = ROOT / "outputs" / "avatar_final" / f_manifest["output_video"]

    log.info("=" * 60)
    log.info("DONE — avatar video: %s", final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
