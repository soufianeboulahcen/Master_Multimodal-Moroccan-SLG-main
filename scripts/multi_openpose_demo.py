"""Multi-video OpenPose-style pose tracking demo.

For each input clip produces up to 6 output variants:
  1. overlay      — skeleton drawn on original frame
  2. skeleton     — skeleton on pure black background
  3. neon         — glowing neon skeleton on black (Gaussian blur glow)
  4. heatmap      — confidence heatmap blended over original
  5. slowmo       — 0.5× speed overlay (frame duplication)
  6. studio       — skeleton on gradient studio background

Plus a mosaic MP4 that tiles all 6 variants side-by-side.
Per-frame OpenPose-compatible JSON is written for every clip.

Usage (pass absolute paths for Arabic filenames):
    python scripts/multi_openpose_demo.py --clips /abs/path/v1.mp4 /abs/path/v2.mp4 \\
        --out-dir demo_output

    # auto-pick N clips per category:
    python scripts/multi_openpose_demo.py --auto 2 --out-dir demo_output
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base
from scipy.ndimage import gaussian_filter

ROOT          = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "holistic_landmarker.task"
DATA_DIR      = ROOT / "data" / "vedios-dataset"

# ── Colour palettes ───────────────────────────────────────────────────────────
# OpenPose rainbow (BGR)
RAINBOW = [
    (  0,215,255),(  0,255,170),(  0,255, 85),(  0,255,  0),
    (  0,170,255),(  0, 85,255),(  0,  0,255),( 85,  0,255),
    (170,  0,255),(255,  0,255),(255,  0,170),(255,  0, 85),
    (255,  0,  0),(255, 85,  0),(255,170,  0),(255,255,  0),
    (170,255,  0),( 85,255,  0),
]
# Neon glow colours per limb (BGR)
NEON = [
    (255,255,  0),(  0,255,255),(  0,255,  0),(255,  0,255),
    (255,128,  0),(  0,128,255),(255,  0,128),(128,255,  0),
    (  0,255,128),(128,  0,255),(255,255,128),(128,255,255),
    (255,128,255),(  0,200,255),(200,  0,255),(255,200,  0),
    (  0,255,200),(200,255,  0),
]

# COCO-18 body edges (a, b, colour_index)
BODY_EDGES = [
    (0,1,0),(1,2,1),(2,3,2),(3,4,3),(1,5,4),(5,6,5),(6,7,6),
    (1,8,7),(8,9,8),(9,10,9),(1,11,10),(11,12,11),(12,13,12),
    (0,14,13),(14,16,14),(0,15,15),(15,17,16),
]
_FINGERS = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
HAND_EDGES = [(a,b) for f in _FINGERS for a,b in zip(f,f[1:])]

LHAND_COL = (255,220,  0)
RHAND_COL = (  0,  0,255)
FACE_COL  = (180,180,180)
JR, BT    = 4, 2          # joint radius, bone thickness

# MediaPipe → COCO-18
_PL = mp_vision.PoseLandmark
MP2C = {
    0:_PL.NOSE,
    2:_PL.RIGHT_SHOULDER, 3:_PL.RIGHT_ELBOW, 4:_PL.RIGHT_WRIST,
    5:_PL.LEFT_SHOULDER,  6:_PL.LEFT_ELBOW,  7:_PL.LEFT_WRIST,
    8:_PL.RIGHT_HIP,  9:_PL.RIGHT_KNEE,  10:_PL.RIGHT_ANKLE,
    11:_PL.LEFT_HIP, 12:_PL.LEFT_KNEE,  13:_PL.LEFT_ANKLE,
    14:_PL.RIGHT_EYE, 15:_PL.LEFT_EYE,
    16:_PL.RIGHT_EAR, 17:_PL.LEFT_EAR,
}

# ── Extraction ────────────────────────────────────────────────────────────────
def extr_body(lms, W, H):
    k = np.zeros((18,3), np.float32)
    if lms is None: return k
    for ci,mi in MP2C.items():
        lm = lms[mi]; k[ci]=[lm.x*W,lm.y*H,float(getattr(lm,'visibility',1.))]
    rs,ls=k[2],k[5]
    if rs[2]>.1 and ls[2]>.1: k[1]=[(rs[0]+ls[0])/2,(rs[1]+ls[1])/2,(rs[2]+ls[2])/2]
    return k

def extr_hand(lms, W, H):
    k=np.zeros((21,3),np.float32)
    if lms is None: return k
    for i,lm in enumerate(lms): k[i]=[lm.x*W,lm.y*H,1.]
    return k

def extr_face(lms, W, H):
    if lms is None: return np.zeros((0,3),np.float32)
    return np.array([[lm.x*W,lm.y*H,1.] for lm in lms],np.float32)

# ── Drawing primitives ────────────────────────────────────────────────────────
def _pt(k,i): return (int(k[i,0]),int(k[i,1]))

def draw_body(canvas, k, palette=RAINBOW, thr=.25, jr=JR, bt=BT):
    for a,b,ci in BODY_EDGES:
        if k[a,2]>thr and k[b,2]>thr:
            cv2.line(canvas,_pt(k,a),_pt(k,b),palette[ci%len(palette)],bt,cv2.LINE_AA)
    for i in range(18):
        if k[i,2]>thr:
            cv2.circle(canvas,_pt(k,i),jr,palette[i%len(palette)],-1,cv2.LINE_AA)
            cv2.circle(canvas,_pt(k,i),jr+1,(0,0,0),1,cv2.LINE_AA)

def draw_hand(canvas, k, col, thr=.1, jr=3, bt=1):
    for a,b in HAND_EDGES:
        if k[a,2]>thr and k[b,2]>thr:
            cv2.line(canvas,_pt(k,a),_pt(k,b),col,bt,cv2.LINE_AA)
    for i in range(21):
        if k[i,2]>thr:
            cv2.circle(canvas,_pt(k,i),jr,col,-1,cv2.LINE_AA)

def draw_face(canvas, k):
    for i in range(len(k)):
        if k[i,2]>.1:
            cv2.circle(canvas,(int(k[i,0]),int(k[i,1])),1,FACE_COL,-1,cv2.LINE_AA)

def draw_all(canvas, body, lh, rh, face, palette=RAINBOW):
    draw_face(canvas,face)
    draw_body(canvas,body,palette)
    draw_hand(canvas,lh,LHAND_COL)
    draw_hand(canvas,rh,RHAND_COL)

# ── Render variants ───────────────────────────────────────────────────────────
def render_overlay(bgr, body, lh, rh, face):
    c=bgr.copy(); draw_all(c,body,lh,rh,face); return c

def render_skeleton(bgr, body, lh, rh, face):
    c=np.zeros_like(bgr); draw_all(c,body,lh,rh,face); return c

def render_neon(bgr, body, lh, rh, face):
    """Glowing neon skeleton: draw thick lines, Gaussian blur, add sharp lines."""
    H,W=bgr.shape[:2]
    base=np.zeros_like(bgr,np.float32)
    # thick glow layer
    tmp=np.zeros_like(bgr)
    draw_body(tmp,body,NEON,jr=8,bt=6)
    draw_hand(tmp,lh,(255,255,0),jr=5,bt=4)
    draw_hand(tmp,rh,(0,100,255),jr=5,bt=4)
    # blur each channel independently for glow
    for c in range(3):
        base[:,:,c]=gaussian_filter(tmp[:,:,c].astype(np.float32),sigma=4)
    base=np.clip(base,0,255).astype(np.uint8)
    # sharp lines on top
    draw_body(base,body,NEON,jr=3,bt=2)
    draw_hand(base,lh,(255,255,0),jr=2,bt=1)
    draw_hand(base,rh,(0,100,255),jr=2,bt=1)
    draw_face(base,face)
    return base

def render_heatmap(bgr, body, lh, rh, face):
    """Confidence heatmap blended over original frame."""
    H,W=bgr.shape[:2]
    heat=np.zeros((H,W),np.float32)
    all_kpts=np.vstack([body,lh,rh])
    for i in range(len(all_kpts)):
        x,y,c=all_kpts[i]
        if c>.1:
            xi,yi=int(x),int(y)
            if 0<=xi<W and 0<=yi<H:
                cv2.circle(heat,(xi,yi),20,float(c),-1)
    heat=gaussian_filter(heat,sigma=12)
    if heat.max()>0: heat/=heat.max()
    heat_bgr=cv2.applyColorMap((heat*255).astype(np.uint8),cv2.COLORMAP_JET)
    blended=cv2.addWeighted(bgr,0.55,heat_bgr,0.45,0)
    # draw skeleton on top
    draw_body(blended,body,RAINBOW,jr=3,bt=2)
    draw_hand(blended,lh,LHAND_COL,jr=2,bt=1)
    draw_hand(blended,rh,RHAND_COL,jr=2,bt=1)
    return blended

def render_studio(bgr, body, lh, rh, face):
    """Skeleton on a dark gradient studio background."""
    H,W=bgr.shape[:2]
    # vertical gradient: dark charcoal top → slightly lighter bottom
    grad=np.zeros((H,W,3),np.uint8)
    for row in range(H):
        v=int(18+22*(row/H))
        grad[row,:]=v
    draw_all(grad,body,lh,rh,face,RAINBOW)
    # subtle vignette
    Y,X=np.ogrid[:H,:W]
    cx,cy=W//2,H//2
    dist=np.sqrt(((X-cx)/cx)**2+((Y-cy)/cy)**2)
    vign=np.clip(1-0.5*dist,0,1).astype(np.float32)
    for c in range(3):
        grad[:,:,c]=(grad[:,:,c]*vign).astype(np.uint8)
    return grad

# ── JSON ──────────────────────────────────────────────────────────────────────
def to_flat(k): return [round(float(v),4) for v in k.flatten()]

def make_json(idx,body,lh,rh,face,W,H):
    return {"version":1.3,"frame_index":idx,"image_size":{"width":W,"height":H},
            "people":[{"person_id":[-1],
                "pose_keypoints_2d":to_flat(body),
                "face_keypoints_2d":to_flat(face),
                "hand_left_keypoints_2d":to_flat(lh),
                "hand_right_keypoints_2d":to_flat(rh),
                "pose_keypoints_3d":[],"face_keypoints_3d":[],
                "hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}]}

# ── Mosaic builder ────────────────────────────────────────────────────────────
VARIANT_LABELS = ["overlay","skeleton","neon","heatmap","slowmo","studio"]

def make_mosaic(frames_dict, W, H):
    """Tile 6 variant frames into a 3×2 grid with labels."""
    cols,rows=3,2
    pad=4; lh_px=22
    cell_w,cell_h=W,H
    out_w=cols*cell_w+(cols+1)*pad
    out_h=rows*(cell_h+lh_px)+(rows+1)*pad
    canvas=np.full((out_h,out_w,3),20,np.uint8)
    for idx,(label,frame) in enumerate(frames_dict.items()):
        r,c=divmod(idx,cols)
        x0=pad+c*(cell_w+pad); y0=pad+r*(cell_h+lh_px+pad)
        # label bar
        cv2.rectangle(canvas,(x0,y0),(x0+cell_w,y0+lh_px),(40,40,40),-1)
        cv2.putText(canvas,label,(x0+6,y0+15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(220,220,220),1,cv2.LINE_AA)
        # frame
        if frame is not None:
            canvas[y0+lh_px:y0+lh_px+cell_h, x0:x0+cell_w]=frame
    return canvas

# ── Per-video pipeline ────────────────────────────────────────────────────────
def process_clip(video_path: Path, out_dir: Path, model_path: Path,
                 write_json=True) -> dict:
    stem=video_path.stem
    clip_dir=out_dir/stem
    clip_dir.mkdir(parents=True,exist_ok=True)
    json_dir=clip_dir/"keypoints"
    if write_json: json_dir.mkdir(exist_ok=True)

    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open: {video_path}")
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    def mkw(name,fps_=None,size=None):
        return cv2.VideoWriter(str(clip_dir/name),fourcc,fps_ or fps,size or (W,H))

    # mosaic is 3×2 grid
    cols,rows=3,2; pad=4; lh_px=22
    mos_w=cols*W+(cols+1)*pad; mos_h=rows*(H+lh_px)+(rows+1)*pad

    writers={
        "overlay":  mkw(f"{stem}_overlay.mp4"),
        "skeleton": mkw(f"{stem}_skeleton.mp4"),
        "neon":     mkw(f"{stem}_neon.mp4"),
        "heatmap":  mkw(f"{stem}_heatmap.mp4"),
        "slowmo":   mkw(f"{stem}_slowmo.mp4", fps_=fps/2),
        "studio":   mkw(f"{stem}_studio.mp4"),
        "mosaic":   mkw(f"{stem}_mosaic.mp4", size=(mos_w,mos_h)),
    }

    opts=mp_vision.HolisticLandmarkerOptions(
        base_options=mp_base.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_suppression_threshold=0.5,
        min_pose_landmarks_confidence=0.5,
        min_face_detection_confidence=0.5,
        min_face_suppression_threshold=0.5,
        min_face_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
    )
    lmk=mp_vision.HolisticLandmarker.create_from_options(opts)

    idx=0; n_det=0
    while True:
        ok,bgr=cap.read()
        if not ok: break
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        res=lmk.detect_for_video(mp_img,int(idx*1000/fps))

        body=extr_body(res.pose_landmarks,W,H)
        lh  =extr_hand(res.left_hand_landmarks,W,H)
        rh  =extr_hand(res.right_hand_landmarks,W,H)
        face=extr_face(res.face_landmarks,W,H)
        if body.max()>0: n_det+=1

        if write_json:
            with open(json_dir/f"{idx:06d}_keypoints.json","w") as f:
                json.dump(make_json(idx,body,lh,rh,face,W,H),f,separators=(",",":"))

        variants={
            "overlay":  render_overlay(bgr,body,lh,rh,face),
            "skeleton": render_skeleton(bgr,body,lh,rh,face),
            "neon":     render_neon(bgr,body,lh,rh,face),
            "heatmap":  render_heatmap(bgr,body,lh,rh,face),
            "slowmo":   render_overlay(bgr,body,lh,rh,face),
            "studio":   render_studio(bgr,body,lh,rh,face),
        }
        for name,frame in variants.items():
            writers[name].write(frame)
            if name=="slowmo":          # duplicate frame for 0.5× speed
                writers[name].write(frame)

        mosaic=make_mosaic(variants,W,H)
        writers["mosaic"].write(mosaic)

        idx+=1
        if idx%25==0:
            print(f"    [{100*idx/max(n_tot,1):5.1f}%] {idx}/{n_tot}",flush=True)

    cap.release(); lmk.close()
    for w in writers.values(): w.release()

    outputs={k:str(clip_dir/f"{stem}_{k}.mp4") for k in writers}
    return {"video":str(video_path),"stem":stem,"frames":idx,"fps":fps,
            "resolution":f"{W}x{H}","body_det_rate":f"{100*n_det/max(idx,1):.1f}%",
            "outputs":outputs,
            "json_dir":str(json_dir) if write_json else None,
            "json_files":idx if write_json else 0}

# ── Auto clip selection ───────────────────────────────────────────────────────
def auto_pick(n_per_cat: int) -> list[Path]:
    picks=[]
    for cat in sorted(os.listdir(str(DATA_DIR))):
        d=DATA_DIR/cat
        if not d.is_dir(): continue
        files=sorted((os.path.getsize(str(d/f)),f)
                     for f in os.listdir(str(d)) if f.endswith(".mp4"))
        for _,f in files[:n_per_cat]:
            picks.append((d/f).resolve())
    return picks

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips",   nargs="*", default=[],
                    help="Absolute paths to input .mp4 files")
    ap.add_argument("--auto",    type=int, default=0,
                    help="Auto-pick N clips per category from the MoSL dataset")
    ap.add_argument("--out-dir", default="demo_output")
    ap.add_argument("--model",   default=str(DEFAULT_MODEL))
    ap.add_argument("--no-json", action="store_true")
    args=ap.parse_args()

    model=Path(args.model)
    if not model.exists():
        print(f"[error] model not found: {model}",file=sys.stderr); return 1

    videos=[]
    if args.clips:
        videos=[Path(c) for c in args.clips]
    if args.auto>0:
        videos+=auto_pick(args.auto)
    if not videos:
        print("[error] provide --clips or --auto N",file=sys.stderr); return 1

    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    print(f"Processing {len(videos)} clip(s) → {out_dir}/")
    print(f"  variants: overlay, skeleton, neon, heatmap, slowmo, studio, mosaic")
    print(f"  json: {'no' if args.no_json else 'yes'}\n")

    summaries=[]; t0=time.perf_counter()
    for i,vp in enumerate(videos,1):
        if not vp.exists():
            print(f"[{i}/{len(videos)}] SKIP (not found): {vp}"); continue
        print(f"[{i}/{len(videos)}] {vp.name}")
        try:
            s=process_clip(vp,out_dir,model,write_json=not args.no_json)
            summaries.append(s)
            print(f"  ✓ {s['frames']} frames  body_det={s['body_det_rate']}")
            for k,p in s["outputs"].items():
                sz=os.path.getsize(p) if os.path.exists(p) else 0
                print(f"    [{k:<8}] {Path(p).name}  ({sz//1024} KB)")
            if s["json_dir"]:
                print(f"    [json    ] {s['json_dir']}/ ({s['json_files']} files)")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ✗ FAILED: {e}",file=sys.stderr)
        print()

    elapsed=time.perf_counter()-t0
    print(f"Done — {len(summaries)}/{len(videos)} succeeded in {elapsed:.1f}s")
    sp=out_dir/"summary.json"
    with open(sp,"w") as f: json.dump(summaries,f,indent=2,ensure_ascii=False)
    print(f"Summary → {sp}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
