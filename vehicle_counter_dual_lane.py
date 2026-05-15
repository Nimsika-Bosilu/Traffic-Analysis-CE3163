"""
CE3163 - Traffic Data Collection
Vehicle Detection, Classification & Counting — DUAL LANE (Up/Down)
Objectives:
  1. Automated Vehicle Detection & Classification (YOLO + BoT-SORT)
  2. Traffic Parameter Extraction (counts, flow rates, speed distributions)
  3. Homography Calibration & Perspective Correction using real-world GCPs
Model  : YOLOv8x
Tracker: BoT-SORT  (built into Ultralytics >= 8.1)
"""

import math
import statistics
import numpy as np
import cv2
import cvzone
import csv
from pathlib import Path
from collections import defaultdict
from datetime import timedelta
from ultralytics import YOLO
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.series import DataPoint

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — update paths / coordinates as needed
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH    = "Transport.mp4"
MASK_PATH     = "mask.png"
MODEL_PATH    = "../Yolo-Weights/yolo26l.pt"
OUTPUT_VIDEO  = "output_counted.mp4"   # set None to skip saving annotated video
CONF_THRESH   = 0.30

# ── Preview window ────────────────────────────────────────────────────────────
# Set SHOW_PREVIEW = False to run headless (no window) — much faster.
# The annotated output video and all Excel/image files are still saved.
# Set SHOW_PREVIEW = True when you want to watch processing live.
SHOW_PREVIEW  = False   # ← change to True to see the live window

# ── Device (GPU / CPU) ────────────────────────────────────────────────────────
# Auto-detects NVIDIA CUDA GPU. Set manually if needed:
#   "cuda"   → force NVIDIA GPU
#   "cpu"    → force CPU
#   "mps"    → Apple Silicon GPU
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_dev_info = f"  ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""
print(f"[DEVICE] Using: {DEVICE}{_dev_info}")

# ── COUNTING lines (one per lane) ────────────────────────────────────────────
# Each line: [x1, y1, x2, y2]  — horizontal within its own lane.
# LINE_UP   spans the UP-direction carriageway  (x: 463 → 980,  y=700)
# LINE_DOWN spans the DOWN-direction carriageway (x: 1020→ 1703, y=699)
# These lines are in SEPARATE lanes divided by a 1.0 m median.
# A vehicle in the UP lane ONLY crosses LINE_UP.
# A vehicle in the DOWN lane ONLY crosses LINE_DOWN.
# When a vehicle's bounding-box centre (cx, cy) passes through its lane's
# counting line (within ±LINE_TOLERANCE pixels), it is counted once.
LINE_UP       = [463,  700,  980,  700]  # UP-lane counting line   (y=700)
LINE_DOWN     = [1020, 699, 1703,  699]  # DOWN-lane counting line (y=699)
LINE_TOLERANCE = 20                       # ± pixels tolerance

# ── SPEED TRAP lines (TWO per lane, bracketing the counting line) ─────────────
# Speed is measured by timing a vehicle between TRAP_A (upstream) and TRAP_B
# (downstream) within the SAME lane. The real-world distance between the two
# trap lines (SPEED_TRAP_DIST_M) must be measured on-site.
#
# Layout (UP lane, side view along road):
#   ─── SPEED_TRAP_UP_A ───  (y = 685, upstream  — 15 px above counting line)
#         ↕  SPEED_TRAP_DIST_M metres  (measure on-site)
#   ─── LINE_UP (counting) ─ (y = 700)
#         ↕
#   ─── SPEED_TRAP_UP_B ───  (y = 715, downstream — 15 px below counting line)
#
# Same pattern for the DOWN lane (counting line y=699, gap ±15 px).
#
SPEED_TRAP_UP_A   = [463,  685,  980,  685]  # UP   upstream   trap line (y=685)
SPEED_TRAP_UP_B   = [463,  715,  980,  715]  # UP   downstream trap line (y=715)
SPEED_TRAP_DOWN_A = [1020, 684, 1703,  684]  # DOWN upstream   trap line (y=684)
SPEED_TRAP_DOWN_B = [1020, 714, 1703,  714]  # DOWN downstream trap line (y=714)

# Real-world distance between TRAP_A and TRAP_B in each lane (metres).
# *** Replace with your on-site physical measurement ***
SPEED_TRAP_DIST_M = 10.0   # metres between TRAP_A and TRAP_B


# ── Vehicle classes (COCO IDs → assignment classification) ───────────────────
# COCO classes remapped to match assignment vehicle categories:
#   motorbike(3)→motorcycle, car(2)→car, truck(7)→light/heavy goods,
#   bus(5)→bus, bicycle(1)→bicycle (three-wheelers not in COCO → remapped)
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ── Extended class labels for report (assignment-specified categories) ─────────
# Maps detected class → assignment report label
CLASS_REPORT_LABEL = {
    "bicycle":    "Bicycle",
    "car":        "Car",
    "motorcycle": "Motorcycle",
    "bus":        "Bus",
    "truck":      "Truck / LGV / HGV",
}

BOTSORT_CFG = dict(
    tracker = "botsort.yaml",
    persist = True,
    conf    = CONF_THRESH,
    iou     = 0.45,
    max_det = 300,
)

INTERVALS = {"1min": 60, "5min": 300, "10min": 600}

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA SITE ASSESSMENT  (documented from field observation)
# ─────────────────────────────────────────────────────────────────────────────
# Mounting position : Center-mounted directly over the road median
#                     (pedestrian overpass / gantry / traffic-light pole)
# Measured height   : ~1.5 m above the top of the median barrier
# Pitch (tilt)      : Slight downward,  ~5° – 10°  →  use 7.5° (midpoint)
#                     (horizon visible at approx. y = 550–600 in 1080p frame)
# Yaw / Roll        : Effectively 0° — camera points straight along road axis
#                     and horizon is perfectly level.
#
# Road geometry (measured / standard Sri Lankan dimensions):
#   UP   lane width  : 7.0 m  (pixel span 463 → 990,  Δ = 527 px)
#   DOWN lane width  : 7.0 m  (pixel span 1020 → 1703, Δ = 683 px)
#   Median width     : 1.0 m  (pixel gap 990 → 1020,  Δ = 30 px)   ← MEASURED
#   Total road width : 15.0 m  (7.0 + 1.0 + 7.0)
#
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_HEIGHT_M   = 1.5     # measured height above median barrier (metres)
CAMERA_PITCH_DEG  = 7.5     # estimated downward tilt (degrees)
LANE_WIDTH_M      = 7.0     # each carriageway lane width (metres)
MEDIAN_WIDTH_M    = 1.0     # center median island width (metres)  ← MEASURED

# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 3 — HOMOGRAPHY / PERSPECTIVE CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
# Unified World Coordinate System
# ─────────────────────────────────────────────────────────────────────────────
#   Origin  : Median centre at LINE_DOWN level
#   X-axis  : Across road  (negative = UP lane side, positive = DOWN lane side)
#   Y-axis  : Along road   (Y=0 at LINE_DOWN / BM-3,BM-4;  Y=+1.5 m at
#             LINE_UP / BM-1,BM-2  — measured from perspective projection)
#
#   Benchmark layout:
#
#       BM-1 ────────────── BM-2        ← y=700 (further from camera, Y=+1.5 m)
#   Unified World Coordinate System (physically measured on-site)
#   ─────────────────────────────────────────────────────────────────────────
#   Origin  : BM-3  (inner-left edge of DOWN lane at LINE_DOWN)
#   X-axis  : Across road (0 m at BM-3, increasing toward DOWN-lane outer edge)
#   Y-axis  : Along road  (0 m at LINE_DOWN / BM-3,BM-4;
#                          10 m at LINE_UP  / BM-1,BM-2  — physically measured)
#
#   Benchmark layout (top = further from camera):
#
#       BM-1 (7.5m,10m) ────── BM-2 (7.0m,10m)   ← pixel y=688  LINE_UP
#       UP lane                 DOWN lane
#       BM-3 (0.0m, 0m) ────── BM-4 (7.0m, 0m)   ← pixel y=869  LINE_DOWN
#
#   Note: X=0 at BM-3 (median edge / left of DOWN lane).
#         X=7.0 m is DOWN-lane outer edge (BM-4) and UP-lane inner edge (BM-2).
#         X=7.5 m is UP-lane outer edge (BM-1) — slight offset measured on-site.
#         Y=10.0 m is the physically measured longitudinal distance between
#         LINE_DOWN (y=869 px) and LINE_UP (y=688 px).
#
# ─────────────────────────────────────────────────────────────────────────────
GCP_PIXEL_POINTS = np.float32([
    [581, 688],   # BM-1: outer-left edge of UP lane   at LINE_UP   (y=688)
    [943, 688],   # BM-2: inner-right edge of UP lane  at LINE_UP   (median edge)
    [  0, 869],   # BM-3: inner-left edge of DOWN lane at LINE_DOWN (origin, y=869)
    [843, 869],   # BM-4: outer-right edge of DOWN lane at LINE_DOWN
])

# Real-world positions — physically measured on-site (metres)
GCP_WORLD_POINTS = np.float32([
    [7.5, 10.0],  # BM-1: UP-lane outer edge,  10.0 m ahead of baseline
    [7.0, 10.0],  # BM-2: UP-lane inner edge,  10.0 m ahead of baseline
    [0.0,  0.0],  # BM-3: DOWN-lane inner edge, at baseline (origin)
    [7.0,  0.0],  # BM-4: DOWN-lane outer edge, at baseline
])

# Physically measured longitudinal distance between LINE_UP and LINE_DOWN (m)
DETECTION_ZONE_M = 10.0

# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 2 — SPEED ESTIMATION  (two-speed-trap-line method per lane)
# ─────────────────────────────────────────────────────────────────────────────
# Speed is timed between TRAP_A (upstream) and TRAP_B (downstream) lines
# within each lane. SPEED_TRAP_DIST_M is the physically measured distance
# between those two lines along the road surface.

COLORS = {
    "car":        (255, 200,   0),
    "truck":      (  0, 165, 255),
    "bus":        (  0, 255, 128),
    "motorcycle": (255,   0, 255),
    "bicycle":    (128, 255, 255),
    "default":    (200, 200, 200),
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_color(cls_name):
    return COLORS.get(cls_name, COLORS["default"])

def hms(sec):
    return str(timedelta(seconds=int(sec)))

def crosses_line(cx, cy, line):
    """Return True when (cx, cy) is within LINE_TOLERANCE of the line segment."""
    lx1, ly1, lx2, ly2 = line
    return lx1 < cx < lx2 and ly1 - LINE_TOLERANCE < cy < ly2 + LINE_TOLERANCE

# ── Homography (Objective 3) ──────────────────────────────────────────────────
def compute_homography():
    """Compute the perspective homography matrix from GCP pixel→world points."""
    H, status = cv2.findHomography(GCP_PIXEL_POINTS, GCP_WORLD_POINTS, cv2.RANSAC, 5.0)
    if H is None:
        print("[WARN] Homography failed — using pixel distances (no correction).")
    return H

def pixel_to_world(H, px, py):
    """Project a pixel point (px, py) to real-world metres using homography H."""
    if H is None:
        return float(px), float(py)
    pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H)
    return float(world[0][0][0]), float(world[0][0][1])

def world_distance_m(H, px1, py1, px2, py2):
    """Real-world distance in metres between two pixel points."""
    wx1, wy1 = pixel_to_world(H, px1, py1)
    wx2, wy2 = pixel_to_world(H, px2, py2)
    return math.hypot(wx2 - wx1, wy2 - wy1)

# ── Speed helpers (Objective 2) ───────────────────────────────────────────────
def estimate_speed_kmh(H, entry_px, entry_py, exit_px, exit_py, elapsed_sec):
    """Estimate speed in km/h using homography-corrected distance and time."""
    if elapsed_sec <= 0:
        return None
    dist_m = world_distance_m(H, entry_px, entry_py, exit_px, exit_py)
    if dist_m < 0.5:   # too small, likely noise
        return None
    speed_ms  = dist_m / elapsed_sec
    return round(speed_ms * 3.6, 2)   # m/s → km/h

# ── Annotated GCP benchmark image (Objective 3 report requirement) ─────────────
def export_gcp_annotated_image(video_path, out_path="gcp_benchmark.jpg"):
    """
    Objective 3 — Extract first frame and overlay all GCP benchmark markers
    with correct world coordinates for the report Methodology section.
    """
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("[WARN] Cannot read video for GCP annotation.")
        return

    # ── Benchmark labels, world coords, descriptions, colours ─────────────────
    bm_info = [
        ("BM-1", GCP_WORLD_POINTS[0], "UP-lane outer edge  (Y=+3.0 m)",  (0,   255, 255)),
        ("BM-2", GCP_WORLD_POINTS[1], "UP-lane inner / median edge",      (255, 128,   0)),
        ("BM-3", GCP_WORLD_POINTS[2], "DOWN-lane inner / median edge",    (0,   255,   0)),
        ("BM-4", GCP_WORLD_POINTS[3], "DOWN-lane outer edge (Y= 0.0 m)",  (255,   0, 255)),
    ]

    for i, (lbl, wpt, desc, col) in enumerate(bm_info):
        px = int(GCP_PIXEL_POINTS[i][0])
        py = int(GCP_PIXEL_POINTS[i][1])
        wx, wy = float(wpt[0]), float(wpt[1])

        # marker circle
        cv2.circle(frame, (px, py), 12, col, -1)
        cv2.circle(frame, (px, py), 14, (0,  0, 0), 2)

        # benchmark ID
        cv2.putText(frame, lbl,
                    (px + 16, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, col, 2, cv2.LINE_AA)

        # world coordinates
        cv2.putText(frame, f"X={wx:+.2f}m  Y={wy:.1f}m",
                    (px + 16, py + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # description
        cv2.putText(frame, desc,
                    (px + 16, py + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # ── Detection zone boundary ────────────────────────────────────────────────
    pts = GCP_PIXEL_POINTS.astype(int)
    cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (0, 200, 255), 2)  # LINE_UP
    cv2.line(frame, tuple(pts[2]), tuple(pts[3]), (0, 200, 255), 2)  # LINE_DOWN
    cv2.line(frame, tuple(pts[0]), tuple(pts[2]), (0, 200, 255), 2)  # left edge
    cv2.line(frame, tuple(pts[1]), tuple(pts[3]), (0, 200, 255), 2)  # right edge

    # ── Dimension labels on zone edges ────────────────────────────────────────
    # UP lane width arrow label
    mid_up_x = (pts[0][0] + pts[1][0]) // 2
    cv2.putText(frame, f"UP lane = {LANE_WIDTH_M} m",
                (mid_up_x - 60, pts[0][1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

    # DOWN lane width arrow label
    mid_dn_x = (pts[2][0] + pts[3][0]) // 2
    cv2.putText(frame, f"DOWN lane = {LANE_WIDTH_M} m",
                (mid_dn_x - 70, pts[2][1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

    # Longitudinal depth label
    mid_left_y = (pts[0][1] + pts[2][1]) // 2
    cv2.putText(frame, f"{DETECTION_ZONE_M} m",
                (pts[0][0] - 75, mid_left_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

    # ── Camera info panel (bottom-left) ───────────────────────────────────────
    info_lines = [
        f"Camera height : ~{CAMERA_HEIGHT_M} m",
        f"Pitch (tilt)  : ~{CAMERA_PITCH_DEG} deg downward",
        f"Yaw / Roll    :  0 deg",
        f"Median width  :  {MEDIAN_WIDTH_M} m",
        f"Zone depth    :  {DETECTION_ZONE_M} m (LINE_UP to LINE_DOWN)",
        f"Total road    :  {LANE_WIDTH_M*2 + MEDIAN_WIDTH_M} m",
    ]
    panel_x, panel_y = 20, frame.shape[0] - 170
    cv2.rectangle(frame, (panel_x - 8, panel_y - 22),
                  (panel_x + 400, panel_y + len(info_lines) * 22 + 4),
                  (30, 30, 30), -1)
    for k, txt in enumerate(info_lines):
        cv2.putText(frame, txt,
                    (panel_x, panel_y + k * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 230, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, frame)
    print(f"[GCP] Annotated benchmark image saved \u2192 {out_path}")



def crosses_line(cx, cy, line):
    """Return True when point (cx,cy) is within LINE_TOLERANCE of the line segment."""
    lx1, ly1, lx2, ly2 = line
    return lx1 < cx < lx2 and ly1 - LINE_TOLERANCE < cy < ly2 + LINE_TOLERANCE


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)   # move model weights to GPU (or CPU)
    print(f"[MODEL] Loaded {MODEL_PATH} on {DEVICE}")
    cap   = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mask = cv2.imread(MASK_PATH)
    if mask is None:
        print("[WARN] mask.png not found – processing full frame.")
        mask = np.ones((height, width, 3), dtype=np.uint8) * 255
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height))

    writer = None
    if OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # ── Objective 3: Compute homography from GCPs ─────────────────────────────
    H = compute_homography()
    print(f"[HOMO] Homography matrix computed from {len(GCP_PIXEL_POINTS)} GCPs.")
    export_gcp_annotated_image(VIDEO_PATH)   # saves gcp_benchmark.jpg for report

    # ── Perspective-corrected trap-line distances ─────────────────────────────
    # Linear scale (px/m = const) is WRONG near y=688 (far end of frame).
    # At the far end, perspective compresses more road into fewer pixels, so
    # 1 px ≈ much more than the average 0.055 m/px.
    # Model: world_Y = A / (y_pixel - HORIZON_Y) + B  (hyperbolic / perspective)
    # Fitted using the two GCP reference rows (y=688→Y=10m, y=869→Y=0m).
    HORIZON_Y  = 580.0   # ← estimated horizon row (adjust ±20 px if speed drifts)
    gcp_y_up   = float(GCP_PIXEL_POINTS[0][1])   # y=688  →  world Y=10 m
    gcp_y_dn   = float(GCP_PIXEL_POINTS[2][1])   # y=869  →  world Y=0 m
    # Solve for A, B:
    A_p = DETECTION_ZONE_M * (gcp_y_up - HORIZON_Y) * (gcp_y_dn - HORIZON_Y) \
          / (gcp_y_dn - gcp_y_up)
    B_p = -A_p / (gcp_y_dn - HORIZON_Y)

    def py_to_worldY(py):
        return A_p / (py - HORIZON_Y) + B_p

    TRAP_DIST_UP   = abs(py_to_worldY(SPEED_TRAP_UP_A[1])   - py_to_worldY(SPEED_TRAP_UP_B[1]))
    TRAP_DIST_DOWN = abs(py_to_worldY(SPEED_TRAP_DOWN_A[1]) - py_to_worldY(SPEED_TRAP_DOWN_B[1]))
    print(f"[SPEED] Perspective trap dist — UP: {TRAP_DIST_UP:.3f} m  "
          f"DOWN: {TRAP_DIST_DOWN:.3f} m  (horizon_y={HORIZON_Y})")

    # ── Per-lane state ────────────────────────────────────────────────────────
    # lanes: "up" | "down"
    counted   = {"up": set(), "down": set()}
    per_class = {"up": defaultdict(int), "down": defaultdict(int)}
    # crossing_log: list of (video_time_sec, lane, class_name)
    crossing_log = []
    # track -> which lane it already crossed (avoid double-count)
    crossed_lane = {}

    # ── Objective 2: Speed tracking state (two-trap-line method) ───────────────
    # trap_a_hit[tid] = (video_time, lane, cls_name)  — set when entry trap crossed
    trap_a_hit  = {}   # tid -> (time, lane, cls_name)
    speed_log   = []   # list of (lane, cls_name, speed_kmh)
    live_speed  = {}   # tid -> latest speed_kmh for HUD / bounding-box label

    frame_no = 0
    if SHOW_PREVIEW:
        print("[INFO] Starting. Press Q in the preview window to quit.")
    else:
        print("[INFO] Running headless. Press Ctrl+C at any time to stop and save outputs.")

    try:
      while True:
        success, frame = cap.read()
        if not success:
            break

        video_time = frame_no / fps
        img_region = cv2.bitwise_and(frame, mask)

        results = model.track(
            img_region,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
            device=DEVICE,
            **BOTSORT_CFG,
        )

        if results[0].boxes.id is not None:
            boxes     = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss      = results[0].boxes.cls.cpu().numpy().astype(int)
            confs     = results[0].boxes.conf.cpu().numpy()

            for box, tid, cls_idx, conf in zip(boxes, track_ids, clss, confs):
                x1, y1, x2, y2 = box
                cls_name = VEHICLE_CLASSES.get(cls_idx, "unknown")
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w  = x2 - x1
                h  = y2 - y1
                color = get_color(cls_name)

                # ── Objective 2: Speed trap-line crossing ────────────────────
                # Direction-aware trap order:
                #   UP   lane → vehicles move bottom→top (y decreasing)
                #              TRAP_UP_B  (y=715) hit FIRST, TRAP_UP_A  (y=685) SECOND
                #   DOWN lane → vehicles move top→bottom (y increasing)
                #              TRAP_DOWN_A(y=684) hit FIRST, TRAP_DOWN_B(y=714) SECOND
                for lane_tag, entry_trap, exit_trap in [
                    ("up",   SPEED_TRAP_UP_B,   SPEED_TRAP_UP_A),   # B first!
                    ("down", SPEED_TRAP_DOWN_A, SPEED_TRAP_DOWN_B), # A first
                ]:
                    # Entry trap crossed → start the clock
                    if crosses_line(cx, cy, entry_trap) and tid not in trap_a_hit:
                        trap_a_hit[tid] = (video_time, lane_tag, cls_name)

                    # Exit trap crossed → finalise speed using homography distance
                    if crosses_line(cx, cy, exit_trap) and tid in trap_a_hit:
                        t_entry, l_entry, c_entry = trap_a_hit[tid]
                        elapsed = video_time - t_entry
                        if elapsed > 0:
                            dist_m    = TRAP_DIST_UP if l_entry == "up" else TRAP_DIST_DOWN
                            speed_kmh = round((dist_m / elapsed) * 3.6, 2)
                            speed_log.append((l_entry, c_entry, speed_kmh))
                            live_speed[tid] = speed_kmh
                        del trap_a_hit[tid]   # reset for next timing

                # ── Line-crossing logic ───────────────────────────────────────
                for lane, line in [("up", LINE_UP), ("down", LINE_DOWN)]:
                    if crosses_line(cx, cy, line):
                        if crossed_lane.get(tid) != lane:   # first cross for this lane
                            crossed_lane[tid] = lane
                            if tid not in counted[lane]:
                                counted[lane].add(tid)
                                per_class[lane][cls_name] += 1
                                crossing_log.append((video_time, lane, cls_name))
                            color = (0, 255, 0)             # flash green

                # ── Draw bounding box ─────────────────────────────────────────
                cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=2, colorR=color)

                # Class + track ID label (at box top-left)
                cvzone.putTextRect(
                    frame, f"{cls_name} #{tid}",
                    (max(0, x1), max(56, y1)),
                    scale=0.60, thickness=1, offset=3,
                )

                # Speed label — prominently ABOVE the bounding box
                if tid in live_speed:
                    spd_txt = f"{live_speed[tid]:.1f} km/h"
                    # dark background pill
                    (tw, th), _ = cv2.getTextSize(
                        spd_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    bx1 = max(0, x1)
                    by1 = max(0, y1 - th - 14)
                    cv2.rectangle(frame, (bx1 - 2, by1),
                                  (bx1 + tw + 6, by1 + th + 8),
                                  (20, 20, 20), -1)
                    cv2.putText(frame, spd_txt,
                                (bx1 + 2, by1 + th + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 255, 180), 2, cv2.LINE_AA)

                cv2.circle(frame, (cx, cy), 5, color, cv2.FILLED)


        # ── Draw counting lines ───────────────────────────────────────────
        lux1,luy1,lux2,luy2 = LINE_UP
        ldx1,ldy1,ldx2,ldy2 = LINE_DOWN
        cv2.line(frame, (lux1,luy1), (lux2,luy2), (0, 100, 255), 3)
        cv2.line(frame, (ldx1,ldy1), (ldx2,ldy2), (255, 100,   0), 3)
        cvzone.putTextRect(frame, "COUNT UP",   (lux1, luy1-8), scale=0.6, thickness=1)
        cvzone.putTextRect(frame, "COUNT DOWN", (ldx1, ldy1-8), scale=0.6, thickness=1)

        # ── Draw speed trap lines ─────────────────────────────────────────
        for trap, col, lbl in [
            (SPEED_TRAP_UP_A,   (0, 220, 100), "UP TRAP-A"),
            (SPEED_TRAP_UP_B,   (0, 180,  80), "UP TRAP-B"),
            (SPEED_TRAP_DOWN_A, (220, 180, 0), "DN TRAP-A"),
            (SPEED_TRAP_DOWN_B, (180, 140, 0), "DN TRAP-B"),
        ]:
            cv2.line(frame, (trap[0], trap[1]), (trap[2], trap[3]), col, 1)
            cvzone.putTextRect(frame, lbl, (trap[0], trap[1]-6),
                               scale=0.45, thickness=1, offset=2)

        # ── TOP SPEED BANNER ─────────────────────────────────────────────────
        all_spds  = [s for _, _, s in speed_log]
        up_spds   = [s for ln, _, s in speed_log if ln == "up"]
        down_spds = [s for ln, _, s in speed_log if ln == "down"]
        avg_all   = statistics.mean(all_spds)  if all_spds  else None
        avg_up    = statistics.mean(up_spds)   if up_spds   else None
        avg_down  = statistics.mean(down_spds) if down_spds else None

        # semi-transparent dark bar across full top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 56), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # timestamp (left)
        cv2.putText(frame, f"CE3163  |  {hms(video_time)}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (200, 230, 255), 2, cv2.LINE_AA)

        # per-lane speed (centre)
        spd_txt = (
            f"UP avg: {avg_up:.1f} km/h" if avg_up else "UP avg: --"
        ) + "    " + (
            f"DOWN avg: {avg_down:.1f} km/h" if avg_down else "DOWN avg: --"
        ) + "    " + (
            f"ALL avg: {avg_all:.1f} km/h  [n={len(all_spds)}]" if avg_all else "ALL avg: --"
        )
        cv2.putText(frame, spd_txt,
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (80, 255, 160), 2, cv2.LINE_AA)

        # last measured speed (top-right)
        if speed_log:
            last_lane, last_cls, last_spd = speed_log[-1]
            rtxt = f"Last: {last_cls} ({last_lane.upper()}) = {last_spd:.1f} km/h"
            rtw  = cv2.getTextSize(rtxt, cv2.FONT_HERSHEY_SIMPLEX, 0.57, 2)[0][0]
            cv2.putText(frame, rtxt, (width - rtw - 12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 220, 50), 2, cv2.LINE_AA)

        # ── SIDE HUD (counts, below banner) ──────────────────────────────────
        total_up   = len(counted["up"])
        total_down = len(counted["down"])
        avg_str    = f"{avg_all:.1f}" if avg_all else "N/A"

        hud = [
            "-- UP lane --",
            *[f"  {k}: {v}" for k, v in sorted(per_class['up'].items())],
            f"  TOTAL UP: {total_up}",
            "-- DOWN lane --",
            *[f"  {k}: {v}" for k, v in sorted(per_class['down'].items())],
            f"  TOTAL DOWN: {total_down}",
            "-- Speed --",
            f"  Samples : {len(all_spds)}",
            f"  Avg     : {avg_str} km/h",
        ]
        for i, txt in enumerate(hud):
            cvzone.putTextRect(frame, txt, (10, 68 + i * 26),
                               scale=0.60, thickness=1, offset=4)

        frame_no += 1
        if writer:
            writer.write(frame)

        # ── Live preview (only when SHOW_PREVIEW = True) ──────────────────────
        if SHOW_PREVIEW:
            cv2.imshow("CE3163 \u2014 Dual Lane Counter (BoT-SORT)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quit by user.")
                break
        else:
            # Headless mode: print progress every 500 frames
            if frame_no % 500 == 0:
                print(f"[INFO] Frame {frame_no}  |  time {hms(video_time)}"
                      f"  |  UP={len(counted['up'])}  DOWN={len(counted['down'])}"
                      f"  |  speed samples={len(speed_log)}")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected — saving outputs and exiting...")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    # ── Export ────────────────────────────────────────────────────────────────
    export_excel(crossing_log, per_class, speed_log)

    print("\n========== FINAL SUMMARY ==========")
    for lane in ("up", "down"):
        print(f"\n  [{lane.upper()} LANE]")
        for cls, cnt in sorted(per_class[lane].items()):
            print(f"    {cls:12s}: {cnt}")
        print(f"    {'TOTAL':12s}: {len(counted[lane])}")

    all_spds = [s for _, _, s in speed_log]
    if all_spds:
        print(f"\n  [SPEED — ALL LANES]  ({len(all_spds)} samples)")
        print(f"    Mean   : {statistics.mean(all_spds):.2f} km/h")
        if len(all_spds) > 1:
            print(f"    Std Dev: {statistics.stdev(all_spds):.2f} km/h")
        print(f"    Min    : {min(all_spds):.2f} km/h")
        print(f"    Max    : {max(all_spds):.2f} km/h")
    print("====================================\n")


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL EXPORT
# ─────────────────────────────────────────────────────────────────────────────

HDR_FILL  = PatternFill("solid", start_color="1F4E79")
HDR_FONT  = Font(bold=True, color="FFFFFF", name="Arial", size=10)
SUB_FILL  = PatternFill("solid", start_color="2E75B6")
DATA_FONT = Font(name="Arial", size=10)
ALT_FILL  = PatternFill("solid", start_color="D6E4F0")
THIN      = Side(style="thin", color="AAAAAA")
BORDER    = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER    = Alignment(horizontal="center", vertical="center")

def style_header(cell, sub=False):
    cell.font      = HDR_FONT
    cell.fill      = SUB_FILL if sub else HDR_FILL
    cell.alignment = CENTER
    cell.border    = BORDER

def style_data(cell, alt=False):
    cell.font      = DATA_FONT
    cell.alignment = CENTER
    cell.border    = BORDER
    if alt:
        cell.fill = ALT_FILL

def auto_width(ws):
    for col in ws.columns:
        max_len = max((len(str(c.value or "")) for c in col), default=8)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 30)


def build_timeseries_sheet(wb, sheet_name, crossing_log, lane, duration, label):
    ws = wb.create_sheet(sheet_name)
    cls_list = list(VEHICLE_CLASSES.values())
    max_time = crossing_log[-1][0] if crossing_log else 0

    headers = ["Interval Start", "Interval End"] + \
              [f"{c} Count" for c in cls_list] + \
              ["Total Count"] + \
              [f"{c} Flow (veh/hr)" for c in cls_list] + \
              ["Total Flow (veh/hr)"]

    ws.row_dimensions[1].height = 22
    for col, h in enumerate(headers, 1):
        cell = ws.cell(1, col, h)
        style_header(cell)

    row = 2
    t   = 0
    while t < max_time:
        bucket = defaultdict(int)
        for ts, ln, cls in crossing_log:
            if ln == lane and t <= ts < t + duration:
                bucket[cls] += 1
        total = sum(bucket.values())
        flow  = {c: round(bucket[c] * 3600 / duration) for c in cls_list}

        values = [hms(t), hms(t + duration)] + \
                 [bucket[c] for c in cls_list] + \
                 [total] + \
                 [flow[c] for c in cls_list] + \
                 [round(total * 3600 / duration)]

        alt = (row % 2 == 0)
        for col, val in enumerate(values, 1):
            cell = ws.cell(row, col, val)
            style_data(cell, alt)
        row += 1
        t   += duration

    # Totals row
    ws.cell(row, 1, "TOTAL").font = Font(bold=True, name="Arial", size=10)
    ws.cell(row, 1).fill          = HDR_FILL
    ws.cell(row, 1).alignment     = CENTER
    ws.cell(row, 1).border        = BORDER
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)

    for col_idx, cls in enumerate(cls_list, 3):
        ws.cell(row, col_idx, f"=SUM({get_column_letter(col_idx)}2:{get_column_letter(col_idx)}{row-1})")
        ws.cell(row, col_idx).font = Font(bold=True, name="Arial", size=10)
        ws.cell(row, col_idx).fill = HDR_FILL
        ws.cell(row, col_idx).font = Font(bold=True, color="FFFFFF", name="Arial")
        ws.cell(row, col_idx).alignment = CENTER
        ws.cell(row, col_idx).border = BORDER

    total_col = 2 + len(cls_list) + 1
    ws.cell(row, total_col, f"=SUM({get_column_letter(total_col)}2:{get_column_letter(total_col)}{row-1})")
    ws.cell(row, total_col).font = Font(bold=True, color="FFFFFF", name="Arial")
    ws.cell(row, total_col).fill = HDR_FILL
    ws.cell(row, total_col).alignment = CENTER
    ws.cell(row, total_col).border = BORDER

    auto_width(ws)
    return ws


def build_composition_sheet(wb, per_class):
    ws = wb.create_sheet("Vehicle Composition")
    cls_list = list(VEHICLE_CLASSES.values())

    headers = ["Vehicle Class", "Up Count", "Down Count", "Total Count",
               "Up %", "Down %", "Total %"]
    ws.row_dimensions[1].height = 22
    for col, h in enumerate(headers, 1):
        style_header(ws.cell(1, col, h))

    data_start = 2
    for row, cls in enumerate(cls_list, data_start):
        alt = (row % 2 == 0)
        up  = per_class["up"].get(cls, 0)
        dn  = per_class["down"].get(cls, 0)
        vals = [cls, up, dn,
                f"=B{row}+C{row}",
                f"=IF(SUM(B{data_start}:B{data_start+len(cls_list)-1})=0,0,B{row}/SUM(B{data_start}:B{data_start+len(cls_list)-1}))",
                f"=IF(SUM(C{data_start}:C{data_start+len(cls_list)-1})=0,0,C{row}/SUM(C{data_start}:C{data_start+len(cls_list)-1}))",
                f"=IF(SUM(D{data_start}:D{data_start+len(cls_list)-1})=0,0,D{row}/SUM(D{data_start}:D{data_start+len(cls_list)-1}))"]
        for col, val in enumerate(vals, 1):
            cell = ws.cell(row, col, val)
            style_data(cell, alt)

    # Total row
    tr = data_start + len(cls_list)
    ws.cell(tr, 1, "TOTAL").font = Font(bold=True, name="Arial", size=10)
    for col in range(1, 8):
        ws.cell(tr, col).fill   = HDR_FILL
        ws.cell(tr, col).font   = Font(bold=True, color="FFFFFF", name="Arial")
        ws.cell(tr, col).alignment = CENTER
        ws.cell(tr, col).border = BORDER
    for col in [2, 3, 4]:
        ltr = get_column_letter(col)
        ws.cell(tr, col, f"=SUM({ltr}{data_start}:{ltr}{tr-1})")
    ws.cell(tr, 5, f"=SUM(E{data_start}:E{tr-1})")
    ws.cell(tr, 6, f"=SUM(F{data_start}:F{tr-1})")
    ws.cell(tr, 7, f"=SUM(G{data_start}:G{tr-1})")

    # Format % columns
    for row in range(data_start, tr + 1):
        for col in [5, 6, 7]:
            ws.cell(row, col).number_format = "0.00%"

    # Bar chart — counts per class
    chart = BarChart()
    chart.type    = "col"
    chart.title   = "Vehicle Count by Class and Lane"
    chart.y_axis.title = "Count"
    chart.x_axis.title = "Vehicle Class"
    chart.style  = 10
    chart.width  = 18
    chart.height = 12

    cats = Reference(ws, min_col=1, min_row=data_start, max_row=tr - 1)
    for col, lane_label in [(2, "Up Lane"), (3, "Down Lane")]:
        data = Reference(ws, min_col=col, min_row=1, max_row=tr - 1)
        series = BarChart()
        chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    ws.add_chart(chart, f"I2")

    auto_width(ws)
    return ws


def build_summary_sheet(wb, per_class, crossing_log):
    ws = wb.active
    ws.title = "Summary"

    ws["A1"] = "CE3163 — Traffic Survey Summary"
    ws["A1"].font = Font(bold=True, size=14, name="Arial", color="1F4E79")
    ws.merge_cells("A1:F1")
    ws["A1"].alignment = CENTER

    ws["A2"] = "Generated by automated YOLO + BoT-SORT pipeline"
    ws["A2"].font = Font(italic=True, size=10, name="Arial", color="666666")
    ws.merge_cells("A2:F2")
    ws["A2"].alignment = CENTER

    headers = ["Metric", "Up Lane", "Down Lane", "Combined"]
    for col, h in enumerate(headers, 1):
        style_header(ws.cell(4, col, h))

    cls_list = list(VEHICLE_CLASSES.values())
    rows_data = []
    for cls in cls_list:
        up  = per_class["up"].get(cls, 0)
        dn  = per_class["down"].get(cls, 0)
        rows_data.append([cls.capitalize(), up, dn, up + dn])

    total_up   = sum(per_class["up"].values())
    total_down = sum(per_class["down"].values())
    rows_data.append(["TOTAL", total_up, total_down, total_up + total_down])

    for i, row_vals in enumerate(rows_data):
        r   = 5 + i
        alt = (i % 2 == 0)
        bold = (i == len(rows_data) - 1)
        for col, val in enumerate(row_vals, 1):
            cell = ws.cell(r, col, val)
            style_data(cell, alt)
            if bold:
                cell.font = Font(bold=True, name="Arial", size=10)
                cell.fill = PatternFill("solid", start_color="BDD7EE")

    auto_width(ws)


def build_speed_sheet(wb, speed_log):
    """
    Objective 2 — Speed Distribution Sheet.
    Section A: Per-lane, per-class speed statistics (count, mean, std, min, max, 85th pctile).
    Section B: Speed histogram (binned counts) per lane with bar chart.
    """
    ws = wb.create_sheet("Speed Distribution")
    cls_list = list(VEHICLE_CLASSES.values())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A — Descriptive Statistics
    # ══════════════════════════════════════════════════════════════════════════
    ws["A1"] = "Speed Distribution by Lane and Vehicle Class (km/h)"
    ws["A1"].font = Font(bold=True, size=12, name="Arial", color="1F4E79")
    ws.merge_cells("A1:H1")
    ws["A1"].alignment = CENTER

    stat_headers = ["Lane", "Vehicle Class", "Count",
                    "Mean (km/h)", "Std Dev (km/h)", "Min (km/h)", "Max (km/h)",
                    "85th Pctile (km/h)"]
    ws.row_dimensions[2].height = 22
    for col, h in enumerate(stat_headers, 1):
        style_header(ws.cell(2, col, h))

    row = 3
    for lane in ("up", "down"):
        for cls in cls_list:
            speeds = [s for ln, c, s in speed_log if ln == lane and c == cls]
            if not speeds:
                continue
            speeds_sorted = sorted(speeds)
            count  = len(speeds)
            mean_v = round(statistics.mean(speeds), 2)
            std_v  = round(statistics.stdev(speeds), 2) if count > 1 else 0.0
            min_v  = round(min(speeds), 2)
            max_v  = round(max(speeds), 2)
            p85_idx = min(int(0.85 * count), count - 1)
            p85_v  = round(speeds_sorted[p85_idx], 2)
            alt = (row % 2 == 0)
            for col, val in enumerate(
                    [lane.upper(), CLASS_REPORT_LABEL.get(cls, cls),
                     count, mean_v, std_v, min_v, max_v, p85_v], 1):
                style_data(ws.cell(row, col, val), alt)
            row += 1

    # Overall combined row
    all_spds = [s for _, _, s in speed_log]
    if all_spds:
        all_sorted = sorted(all_spds)
        p85_idx = min(int(0.85 * len(all_spds)), len(all_spds) - 1)
        for col, val in enumerate(
                ["ALL", "All Classes",
                 len(all_spds),
                 round(statistics.mean(all_spds), 2),
                 round(statistics.stdev(all_spds), 2) if len(all_spds) > 1 else 0.0,
                 round(min(all_spds), 2),
                 round(max(all_spds), 2),
                 round(all_sorted[p85_idx], 2)], 1):
            cell = ws.cell(row, col, val)
            cell.font = Font(bold=True, name="Arial", size=10, color="FFFFFF")
            cell.fill = HDR_FILL
            cell.alignment = CENTER
            cell.border = BORDER

    auto_width(ws)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION B — Speed Histogram (binned counts per lane)
    # Required by Objective 2: "speed distribution (histogram) per lane"
    # ══════════════════════════════════════════════════════════════════════════
    BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 9999)]
    BIN_LABELS = ["0–20", "20–40", "40–60", "60–80", "80–100", "100+"]

    # Leave 2 blank rows after Section A then start histogram
    hist_start_row = row + 3
    ws.cell(hist_start_row, 1, "Speed Histogram — Vehicle Count per Speed Band (km/h)")
    ws.cell(hist_start_row, 1).font = Font(bold=True, size=11, name="Arial", color="1F4E79")
    ws.merge_cells(
        start_row=hist_start_row, start_column=1,
        end_row=hist_start_row, end_column=4)
    ws.cell(hist_start_row, 1).alignment = CENTER

    hist_header_row = hist_start_row + 1
    for col, h in enumerate(["Speed Band (km/h)", "UP Lane", "DOWN Lane", "Combined"], 1):
        style_header(ws.cell(hist_header_row, col, h))

    hist_data_start = hist_header_row + 1
    for i, (lo, hi) in enumerate(BINS):
        r = hist_data_start + i
        up_cnt   = sum(1 for ln, _, s in speed_log if ln == "up"   and lo <= s < hi)
        down_cnt = sum(1 for ln, _, s in speed_log if ln == "down" and lo <= s < hi)
        alt = (i % 2 == 0)
        for col, val in enumerate(
                [BIN_LABELS[i], up_cnt, down_cnt, up_cnt + down_cnt], 1):
            style_data(ws.cell(r, col, val), alt)

    # Bar chart for histogram
    hist_chart = BarChart()
    hist_chart.type   = "col"
    hist_chart.title  = "Speed Distribution Histogram by Lane"
    hist_chart.y_axis.title = "Vehicle Count"
    hist_chart.x_axis.title = "Speed Band (km/h)"
    hist_chart.style  = 10
    hist_chart.width  = 20
    hist_chart.height = 14

    cats = Reference(ws,
                     min_col=1, min_row=hist_data_start,
                     max_row=hist_data_start + len(BINS) - 1)
    for col_idx, lane_lbl in [(2, "UP Lane"), (3, "DOWN Lane")]:
        data = Reference(ws,
                         min_col=col_idx, min_row=hist_header_row,
                         max_row=hist_data_start + len(BINS) - 1)
        hist_chart.add_data(data, titles_from_data=True)
    hist_chart.set_categories(cats)
    ws.add_chart(hist_chart, f"F{hist_start_row}")

    return ws



def build_camera_info_sheet(wb):
    """
    Methodology sheet — Camera setup, GCP coordinates, road geometry.
    Satisfies Objective 3 reporting requirement.
    """
    ws = wb.create_sheet("Camera & Site Info")

    def sec(r, title):
        ws.cell(r, 1, title).font = Font(bold=True, size=11, name="Arial", color="1F4E79")
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=4)
        ws.cell(r, 1).fill = PatternFill("solid", start_color="D6E4F0")

    def row2(r, key, val):
        ws.cell(r, 1, key).font  = Font(bold=True, name="Arial", size=10)
        ws.cell(r, 2, val).font  = Font(name="Arial", size=10)
        ws.cell(r, 1).border = ws.cell(r, 2).border = BORDER

    sec(1, "CE3163 — Camera & Site Survey Details")
    row2(2, "Camera mounting",       "Centre-mounted over road median (overpass / gantry)")
    row2(3, "Estimated height (m)",  f"{CAMERA_HEIGHT_M} m above median barrier")
    row2(4, "Pitch (tilt)",          f"~{CAMERA_PITCH_DEG}° downward")
    row2(5, "Yaw / Roll",            "0° (aligned with road axis, horizon level)")
    row2(6, "Lane width (each)",     f"{LANE_WIDTH_M} m  (standard Sri Lankan)")
    row2(7, "Median width",          f"{MEDIAN_WIDTH_M} m  (measured on-site)")
    row2(8, "Total road width",      f"{LANE_WIDTH_M*2 + MEDIAN_WIDTH_M} m")

    sec(10, "Ground Control Points (GCPs) — Unified Coordinate System")
    for col, h in enumerate(["BM ID", "Pixel X", "Pixel Y", "World X (m)", "World Y (m)", "Description"], 1):
        style_header(ws.cell(11, col, h))
    bm_desc = [
        "UP-lane outer edge  at LINE_UP",
        "UP-lane / median edge at LINE_UP",
        "DOWN-lane / median edge at LINE_DOWN (baseline Y=0)",
        "DOWN-lane outer edge at LINE_DOWN (baseline Y=0)",
    ]
    for i, (px, wy) in enumerate(zip(GCP_PIXEL_POINTS, GCP_WORLD_POINTS)):
        r = 12 + i
        for col, val in enumerate(
                [f"BM-{i+1}", int(px[0]), int(px[1]),
                 float(wy[0]), float(wy[1]), bm_desc[i]], 1):
            style_data(ws.cell(r, col, val), i % 2 == 0)

    sec(17, "Speed Trap Configuration")
    row2(18, "Method",            "Two-line speed trap per lane")
    row2(19, "UP TRAP-A line",    f"y={SPEED_TRAP_UP_A[1]}  (upstream,  x: {SPEED_TRAP_UP_A[0]}→{SPEED_TRAP_UP_A[2]})")
    row2(20, "UP TRAP-B line",    f"y={SPEED_TRAP_UP_B[1]}  (downstream, x: {SPEED_TRAP_UP_B[0]}→{SPEED_TRAP_UP_B[2]})")
    row2(21, "DOWN TRAP-A line",  f"y={SPEED_TRAP_DOWN_A[1]}  (upstream,  x: {SPEED_TRAP_DOWN_A[0]}→{SPEED_TRAP_DOWN_A[2]})")
    row2(22, "DOWN TRAP-B line",  f"y={SPEED_TRAP_DOWN_B[1]}  (downstream, x: {SPEED_TRAP_DOWN_B[0]}→{SPEED_TRAP_DOWN_B[2]})")
    row2(23, "Trap distance (m)", f"{SPEED_TRAP_DIST_M} m  (measure on-site and update)")
    row2(24, "Detection zone (m)",f"{DETECTION_ZONE_M} m  (LINE_UP to LINE_DOWN)")
    row2(25, "Speed formula",     "speed (km/h) = (distance_m / elapsed_s) × 3.6")

    auto_width(ws)
    return ws


def build_speed_class_chart_sheet(wb, speed_log):
    """
    Extra analysis: mean speed per vehicle class per lane — bar chart.
    Helps show which vehicle types are fastest/slowest.
    """
    ws = wb.create_sheet("Speed by Class")
    cls_list = list(VEHICLE_CLASSES.values())

    ws["A1"] = "Mean Speed by Vehicle Class and Lane (km/h)"
    ws["A1"].font = Font(bold=True, size=12, name="Arial", color="1F4E79")
    ws.merge_cells("A1:D1")
    ws["A1"].alignment = CENTER

    for col, h in enumerate(["Vehicle Class", "UP Lane Mean (km/h)",
                              "DOWN Lane Mean (km/h)", "Combined Mean (km/h)"], 1):
        style_header(ws.cell(2, col, h))

    data_start = 3
    for i, cls in enumerate(cls_list):
        up_s   = [s for ln, c, s in speed_log if ln == "up"   and c == cls]
        dn_s   = [s for ln, c, s in speed_log if ln == "down" and c == cls]
        all_s  = up_s + dn_s
        lbl    = CLASS_REPORT_LABEL.get(cls, cls)
        up_m   = round(statistics.mean(up_s),  2) if up_s  else ""
        dn_m   = round(statistics.mean(dn_s),  2) if dn_s  else ""
        all_m  = round(statistics.mean(all_s), 2) if all_s else ""
        r = data_start + i
        alt = (i % 2 == 0)
        for col, val in enumerate([lbl, up_m, dn_m, all_m], 1):
            style_data(ws.cell(r, col, val), alt)

    # Bar chart
    chart = BarChart()
    chart.type   = "col"
    chart.title  = "Mean Speed by Vehicle Class"
    chart.y_axis.title = "Mean Speed (km/h)"
    chart.x_axis.title = "Vehicle Class"
    chart.style  = 10
    chart.width  = 18
    chart.height = 13

    end_row = data_start + len(cls_list) - 1
    cats = Reference(ws, min_col=1, min_row=data_start, max_row=end_row)
    for col_idx, lbl in [(2, "UP Lane"), (3, "DOWN Lane")]:
        data = Reference(ws, min_col=col_idx, min_row=2, max_row=end_row)
        chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    ws.add_chart(chart, "F2")

    auto_width(ws)
    return ws


def export_excel(crossing_log, per_class, speed_log=None):
    if speed_log is None:
        speed_log = []
    wb = Workbook()
    build_summary_sheet(wb, per_class, crossing_log)
    build_camera_info_sheet(wb)
    build_composition_sheet(wb, per_class)
    build_speed_sheet(wb, speed_log)
    build_speed_class_chart_sheet(wb, speed_log)

    for lane in ("up", "down"):
        for label, duration in INTERVALS.items():
            sheet_name = f"{lane.capitalize()}_{label}"
            build_timeseries_sheet(wb, sheet_name, crossing_log, lane, duration, label)

    out = "traffic_report.xlsx"
    wb.save(out)
    print(f"[EXCEL] Saved → {out}")
    print(f"  Sheets: {[s.title for s in wb.worksheets]}")



# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
