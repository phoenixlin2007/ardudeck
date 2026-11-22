#!/usr/bin/env python3
# record_auto.py (v4 — 直連 SITL TCP 5762 免 MAVProxy, 預設高度 250m 適用 3D 地景)
#
# 用法:
#   python3 scripts/vision_nav/record_auto.py
#   python3 scripts/vision_nav/record_auto.py --rtsp rtsp://127.0.0.1:8554/vision --alt 250
#   python3 scripts/vision_nav/record_auto.py --restore-gps   # 若先前切過無 GPS EKF，一鍵還原 GPS 參數

import argparse
import json
import math
import os
import sys
import time
import cv2
from pymavlink import mavutil

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from utils import (
    DEFAULT_MAVLINK_CONN,
    DEFAULT_RTSP_URL,
    RTSPCamera,
    connect_mavlink,
    restore_gps_ekf_params,
    sky_ratio,
    gray_veil_ratio,
    contrast_of,
    edge_density,
)

# ── 航線: 松山機場跑道(=home=起飛點) → 台北101 ──
HOME_TSA     = (25.070050, 121.539620)   # 松山跑道 = SITL home = 起飛點
TAIPEI101    = (25.0339, 121.5645)       # 台北101
ROUTE_STEP_M = 300                       # 航點間隔

TARGET_ALT       = 250.0                 # 起飛/巡航高度 (relative, 250m 避開台北 3D 建築)
WAYPOINT_RADIUS  = 15.0
CAPTURE_INTERVAL = 0.4
CTRL_LOOP_HZ     = 10.0
CLIMB_TIMEOUT_S  = 180.0


def parse_args():
    p = argparse.ArgumentParser(description="3D 地景視覺導航路線自動錄製 (直連 SITL，免 MAVProxy)")
    p.add_argument("--mavlink", default=DEFAULT_MAVLINK_CONN,
                   help=f"MAVLink 連線字串 (預設: {DEFAULT_MAVLINK_CONN})")
    p.add_argument("--rtsp", default=DEFAULT_RTSP_URL,
                   help=f"ArduDeck RTSP 串流位址 (預設: {DEFAULT_RTSP_URL})")
    p.add_argument("--alt", type=float, default=TARGET_ALT,
                   help=f"起飛與巡航相對高度 m (預設: {TARGET_ALT})")
    p.add_argument("--out", default=os.path.join(_THIS, "dataset_auto"),
                   help="資料集儲存目錄 (預設: scripts/vision_nav/dataset_auto)")
    p.add_argument("--restore-gps", action="store_true",
                   help="僅還原 EKF3 為正常 GPS 導航參數並結束 (需重啟 SITL 生效)")
    return p.parse_args()


def haversine(la1, lo1, la2, lo2):
    R = 6371000
    dla = math.radians(la2 - la1)
    dlo = math.radians(lo2 - lo1)
    a = math.sin(dla / 2) ** 2 + math.cos(math.radians(la1)) * math.cos(math.radians(la2)) * math.sin(dlo / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_rad(la1, lo1, la2, lo2):
    """計算從 (la1, lo1) 指向 (la2, lo2) 的方位角 (弧度, 正北為 0 順時針)"""
    p1, p2 = math.radians(la1), math.radians(la2)
    dl = math.radians(lo2 - lo1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.atan2(y, x)


def log(tag, msg):
    print(f"  [{tag}] {msg}", flush=True)


def build_route(step_m=ROUTE_STEP_M):
    la0, lo0 = HOME_TSA
    la1, lo1 = TAIPEI101
    total = haversine(la0, lo0, la1, lo1)
    n = max(2, int(total // step_m) + 1)
    return [(la0 + (la1 - la0) * i / n, lo0 + (lo1 - lo0) * i / n) for i in range(n + 1)]


def check_gps_health(mav, samples=5):
    """起飛前 GPS 健檢: fix/衛星數/HDOP/EKF-GPS 一致性"""
    fixes, sats, hdops = [], [], []
    for _ in range(samples):
        g = mav.recv_match(type='GPS_RAW_INT', blocking=True, timeout=2)
        if g:
            fixes.append(g.fix_type)
            sats.append(g.satellites_visible)
            hdops.append(g.eph if g.eph != 65535 else 0)
        time.sleep(0.4)
    if not fixes:
        return False, "完全收不到 GPS_RAW_INT (GPS 驅動未輸出)"
    fix, sat, hdop = max(fixes), min(sats), max(hdops)
    problems = []
    if fix < 3:
        problems.append(f"fix_type={fix} (<3, 無 3D 定位)")
    if sat < 6:
        problems.append(f"衛星數={sat} (<6)")
    if hdop > 200:
        problems.append(f"HDOP={hdop/100:.1f} (>2.0)")
    p = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
    g2 = mav.recv_match(type='GPS_RAW_INT', blocking=True, timeout=2)
    if p and g2:
        d = haversine(p.lat / 1e7, p.lon / 1e7, g2.lat / 1e7, g2.lon / 1e7)
        if d > 50:
            problems.append(f"EKF 與 GPS 位置差 {d:.0f}m (>50m, EKF 可能未用 GPS)")
    if problems:
        return False, "; ".join(problems)
    return True, f"fix={fix} sats={sat} hdop={hdop/100:.2f}"


def boost_climb_speed(c):
    """提高多軸爬升速度 (6 m/s) 與水平巡航速度 (10 m/s)"""
    params = (
        (b'WPNAV_SPEED_UP', 600.0),
        (b'WP_SPD_UP', 6.0),
        (b'WPNAV_SPEED', 1000.0),
        (b'WP_SPD', 10.0),
    )
    for name, val in params:
        c.mav.param_set_send(
            c.target_system, c.target_component,
            name, float(val), mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )
        time.sleep(0.1)


def set_guided(c):
    c.mav.set_mode_send(c.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)
    time.sleep(0.5)


def wait_mode(c, mode, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hb = c.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb and hb.custom_mode == mode:
            return True
    return False


def wait_armed(c, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hb = c.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb and (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            return True
    return False


def send_target(c, lat, lon, alt_rel, yaw_rad=None):
    type_mask = 0b0000101111111000 if yaw_rad is not None else 0b0000111111111000
    c.mav.set_position_target_global_int_send(
        0, c.target_system, c.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        type_mask,
        int(lat * 1e7), int(lon * 1e7), float(alt_rel),
        0, 0, 0, 0, 0, 0,
        float(yaw_rad) if yaw_rad is not None else 0.0, 0,
    )


def drain_telemetry(mav, last_pos, last_att):
    """排空 MAVLink 接收緩衝區，保留最新的 GLOBAL_POSITION_INT 與 ATTITUDE"""
    while True:
        m = mav.recv_match(blocking=False)
        if m is None:
            break
        t = m.get_type()
        if t == 'GLOBAL_POSITION_INT':
            last_pos = m
        elif t == 'ATTITUDE':
            last_att = m
    return last_pos, last_att


def land(c):
    c.mav.command_long_send(
        c.target_system, c.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0,
    )


def main():
    args = parse_args()
    target_alt = args.alt
    save_dir = args.out

    log("CONN", f"連接 MAVLink ({args.mavlink}) ...")
    mav = connect_mavlink(args.mavlink)

    if args.restore_gps:
        restore_gps_ekf_params(mav)
        return

    route = build_route()
    log("DB", f"航線: 松山跑道{HOME_TSA}(=home) → 台北101{TAIPEI101}, "
              f"{len(route)} 航點, 總長 {haversine(*HOME_TSA, *TAIPEI101)/1000:.2f} km, 高度 {target_alt:.0f} m")
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)

    log("CONN", f"連接 RTSP ({args.rtsp}) ...")
    camera = RTSPCamera(args.rtsp)

    # ── 起飛前 GPS 健檢 ──
    log("GPS", "起飛前 GPS 健康檢查 (約 3 秒)...")
    ok, detail = check_gps_health(mav)
    if not ok:
        log("GPS", f"❌ GPS 異常: {detail}")
        log("GPS", "⚠️ 若先前跑過無 GPS 模式，請先執行 --restore-gps 並重啟 SITL，已中止錄製")
        camera.release()
        return
    log("GPS", f"✅ GPS 健康: {detail}")

    # ── 自動解鎖 + 原地起飛 250m ──
    boost_climb_speed(mav)
    set_guided(mav)
    if not wait_mode(mav, 4, 5):
        log("MODE", "❌ 切 GUIDED 失敗, 中止")
        camera.release()
        return
    log("MODE", "✅ GUIDED")

    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0,
    )
    if not wait_armed(mav, 8):
        log("ARM", "❌ 解鎖失敗, 中止")
        camera.release()
        return
    log("ARM", "✅ ARMED")

    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, target_alt,
    )
    log("TAKEOFF", f"松山跑道原地起飛至 {target_alt:.0f} m ...")
    t0, last_print = time.time(), 0.0
    reached = False
    while time.time() - t0 < CLIMB_TIMEOUT_S:
        p = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=0.5)
        if p:
            alt = p.relative_alt / 1000.0
            if time.time() - last_print > 5.0:
                log("CLIMB", f"alt={alt:.1f} / {target_alt:.0f} m")
                last_print = time.time()
            if abs(alt - target_alt) < 5.0:
                reached = True
                break
    if not reached:
        log("TAKEOFF", f"⚠️ 爬升逾時 ({CLIMB_TIMEOUT_S:.0f}s)，將以目前高度繼續錄製")
    else:
        log("TAKEOFF", f"✅ 到達 {target_alt:.0f} m, 開始錄製 (爬升段不錄)")

    # ── 沿航線飛並錄製 (route[0]=home 已在正下方, 從 route[1] 開始導引) ──
    metadata, idx, last, wp_i = [], 0, 0.0, 1
    last_pos, last_att = None, None
    try:
        while wp_i < len(route):
            now = time.time()
            wp = route[wp_i]
            seg_yaw = bearing_rad(*route[wp_i - 1], *wp)
            last_pos, last_att = drain_telemetry(mav, last_pos, last_att)

            if now - last >= CAPTURE_INTERVAL:
                frame = camera.read()
                pos = last_pos or mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
                att = last_att or mav.recv_match(type='ATTITUDE', blocking=True, timeout=1)
                if pos:
                    last_pos = pos
                if att:
                    last_att = att
                if frame is not None and pos:
                    item = {
                        "index": idx,
                        "lat": pos.lat / 1e7,
                        "lon": pos.lon / 1e7,
                        "alt": pos.relative_alt / 1000.0,
                        "roll": math.degrees(att.roll) if att else 0.0,
                        "pitch": math.degrees(att.pitch) if att else 0.0,
                        "yaw": round(math.degrees(att.yaw) % 360.0, 2) if att else 0.0,
                        "sky": round(sky_ratio(frame), 3),
                        "veil": round(gray_veil_ratio(frame), 3),
                        "contrast": round(contrast_of(frame), 2),
                        "edge": round(edge_density(frame), 4),
                        "t": round(now, 2),
                    }
                    cv2.imwrite(os.path.join(save_dir, "images", f"{idx:06d}.jpg"), frame)
                    metadata.append(item)
                    idx += 1
                    last = now
                    log("REC", f"[{idx:04d}] alt={item['alt']:.1f}m sky={item['sky']:.2f} "
                               f"edge={item['edge']:.3f} lat={item['lat']:.6f} lon={item['lon']:.6f}")
            send_target(mav, wp[0], wp[1], target_alt, yaw_rad=seg_yaw)
            if last_pos and haversine(last_pos.lat / 1e7, last_pos.lon / 1e7, wp[0], wp[1]) < WAYPOINT_RADIUS:
                wp_i += 1
                log("NAV", f"✅ 航點 {wp_i}/{len(route)}")
            time.sleep(1.0 / CTRL_LOOP_HZ)
        log("NAV", "🏁 到達台北101, 降落")
        land(mav)
    except KeyboardInterrupt:
        print("\n⚠️ 中斷, 降落")
        land(mav)
    finally:
        meta_path = os.path.join(save_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        log("REC", f"✅ 共 {len(metadata)} 幀 → {meta_path}")
        camera.release()


if __name__ == "__main__":
    main()
