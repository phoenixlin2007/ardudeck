#!/usr/bin/env python3
# process_mission.py (v2 — 相對路徑自適應，支援 --dir 參數)
import argparse
import json
import math
import os
import sys
import cv2
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from utils import FeatureExtractor, sky_ratio, gray_veil_ratio, contrast_of, edge_density

# ── 過濾閾值 ──
SKY_THR      = 0.35    # 天空主導(淺俯角) → 丟 (視角混疊源)
VEIL_THR     = 0.60    # 極端霧 且 低對比 → 丟 (edge+clahe 也救不回)
CONTRAST_MIN = 15.0
EDGE_MIN     = 0.015   # 邊緣稀少(水面/草地) → 丟
MAX_SPEED    = 25.0    # m/s 相鄰幀速度上限 (擋跳變)
# ── 評估門檻 ──
PASS_SCORE   = 70.0
PASS_LE10    = 0.80    # ≤10m 比例門檻
PASS_KEEP    = 0.60    # 保留幀比例門檻
MED_GOOD, MED_BAD = 5.0, 20.0


def parse_args():
    p = argparse.ArgumentParser(description="分析錄製路線影像品質、提取特徵並進行開環精度評估")
    p.add_argument("--dir", default=os.path.join(_THIS, "dataset_auto"),
                   help="資料集目錄 (預設: scripts/vision_nav/dataset_auto)")
    return p.parse_args()


def dist_m(a, b):
    return math.hypot((a[0] - b[0]) * 111000, (a[1] - b[1]) * 111000 * math.cos(math.radians(a[0])))


def enrich(ds, meta):
    """若 metadata 缺品質欄位 (舊版錄製), 從影像即時計算回填"""
    keys = ("sky", "veil", "contrast", "edge", "t")
    if all(all(k in m for k in keys) for m in meta):
        return meta, False
    for m in meta:
        img = cv2.imread(os.path.join(ds, "images", f"{m['index']:06d}.jpg"))
        if img is None:
            continue
        m.setdefault("sky", round(sky_ratio(img), 3))
        m.setdefault("veil", round(gray_veil_ratio(img), 3))
        m.setdefault("contrast", round(contrast_of(img), 2))
        m.setdefault("edge", round(edge_density(img), 4))
        m.setdefault("t", float(m["index"]) * 1.0)
    return meta, True


def main():
    args = parse_args()
    ds = args.dir
    meta_path = os.path.join(ds, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    meta, enriched = enrich(ds, meta)
    if enriched:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print("[process] 已回填品質欄位到 metadata.json (舊版錄製相容)")
    print(f"[process] 讀取 {len(meta)} 幀 ({ds}), 開始品質過濾...")

    keep, drop = [], {"sky": 0, "fog": 0, "edge": 0, "speed": 0}
    prev = None
    for m in meta:
        if prev is not None:
            dt = abs(m["t"] - prev["t"]) or 1.0
            if dist_m((m["lat"], m["lon"]), (prev["lat"], prev["lon"])) / dt > MAX_SPEED:
                drop["speed"] += 1
                continue
        if m["sky"] > SKY_THR:
            drop["sky"] += 1
            continue
        if m["veil"] > VEIL_THR and m["contrast"] < CONTRAST_MIN:
            drop["fog"] += 1
            continue
        if m["edge"] < EDGE_MIN:
            drop["edge"] += 1
            continue
        keep.append(m)
        prev = m

    keep_ratio = len(keep) / len(meta) if meta else 0
    print(f"  丟棄: sky={drop['sky']} fog={drop['fog']} edge={drop['edge']} speed={drop['speed']}"
          f" → 保留 {len(keep)}/{len(meta)} ({keep_ratio*100:.1f}%)")

    report = {"total": len(meta), "kept": len(keep), "drop": drop,
              "keep_ratio": round(keep_ratio, 3)}
    if len(keep) < 20:
        report.update({"score": 0, "pass": False,
                       "advice": "保留幀過少(<20), 建議重新錄製(避開起降段/水面/極端霧)"})
        with open(os.path.join(ds, "process_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print("❌ " + report["advice"])
        return

    # ── 建特徵 + 快取 ──
    feat_dir = os.path.join(ds, "features")
    os.makedirs(feat_dir, exist_ok=True)
    ex = FeatureExtractor()
    feats = []
    for m in keep:
        p = os.path.join(feat_dir, f"{m['index']:06d}.npy")
        if os.path.exists(p):
            f_vec = np.load(p)
        else:
            img = cv2.imread(os.path.join(ds, "images", f"{m['index']:06d}.jpg"))
            f_vec = ex.extract(img)
            np.save(p, f_vec)
        feats.append(f_vec)
    feats = np.array(feats)
    truth = [(m["lat"], m["lon"]) for m in keep]

    # ── 交錯保留法開環評估 (查詢幀不在 DB) ──
    db_i = list(range(0, len(feats), 2))
    q_i = list(range(1, len(feats), 2))
    db, dbt = feats[db_i], [truth[i] for i in db_i]
    errs = np.array([dist_m(dbt[int(np.argmax(db @ feats[q]))], truth[q]) for q in q_i])
    le10, le5 = float((errs <= 10).mean()), float((errs <= 5).mean())
    med, mean = float(np.median(errs)), float(errs.mean())
    p95, mx = float(np.percentile(errs, 95)), float(errs.max())

    # ── 評分 ──
    s_acc = le10 * 50
    s_med = 30.0 if med <= MED_GOOD else (0.0 if med >= MED_BAD
            else 30 * (MED_BAD - med) / (MED_BAD - MED_GOOD))
    s_keep = keep_ratio * 20
    total = s_acc + s_med + s_keep
    ok = (total >= PASS_SCORE) and (le10 >= PASS_LE10) and (keep_ratio >= PASS_KEEP)

    advice = ("✅ 品質合格, 可執行 repeat_mission.py" if ok else
              "❌ 品質不足, 建議重新錄製: " +
              ("; 匹配精度低(≤10m 比例不足)→ 選地標豐富路線/避開重複街區 " if le10 < PASS_LE10 else "") +
              ("; 中位誤差過大 → 檢查相機穩定性/速度 " if med > MED_GOOD else "") +
              ("; 保留幀過少 → 避開起降段/水面/極端霧區" if keep_ratio < PASS_KEEP else ""))

    report.update({"le5": round(le5, 3), "le10": round(le10, 3), "median_m": round(med, 2),
                   "mean_m": round(mean, 2), "p95_m": round(p95, 2), "max_m": round(mx, 2),
                   "score": round(total, 1), "pass": bool(ok), "advice": advice})
    with open(os.path.join(ds, "process_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(ds, "processed_metadata.json"), "w") as f:
        json.dump(keep, f, indent=2)

    print("\n" + "=" * 52)
    print("  📊 PROCESS 評估報告 (交錯保留法開環精度)")
    print("=" * 52)
    print(f"  保留幀   : {len(keep)}/{len(meta)} ({keep_ratio*100:.1f}%)")
    print(f"  平均誤差 : {mean:.2f} m | 中位: {med:.2f} m")
    print(f"  P95      : {p95:.2f} m | 最大: {mx:.2f} m")
    print(f"  ≤5m: {le5*100:.1f}% | ≤10m: {le10*100:.1f}%")
    print(f"  🏆 綜合評分: {total:.1f}/100")
    print(f"  {advice}")
    print("=" * 52)


if __name__ == "__main__":
    main()
