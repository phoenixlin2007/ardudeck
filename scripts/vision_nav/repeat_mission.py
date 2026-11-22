#!/usr/bin/env python3
# repeat_mission.py (ported from repeat_mission_claude.py — 直連 SITL TCP 5762 免 MAVProxy, 預設高度 250m)
#
# 目的: 拿掉「真值直接當定位」這個取巧, 同時不需要關掉 SITL 的 GPS (關掉會讓 ArduDeck 無法渲染)。
#
# 做法:
#   EKF 的外部定位輸入 (EV) 不再是真值, 而是一個「會漂移的里程計」:
#       里程計增量 = M · 真值位移增量 + 偏差 · dt + 雜訊
#       M = (1 + 尺度誤差) · 旋轉(航向誤差),  偏差為隨機漫步
#   這個里程計模擬真實 VIO / 光流會有的累積漂移。
#   視覺匹配的結果 (特徵庫最相似那一幀的座標) 以低增益方式修正里程計的積分狀態。
#
# 三種模式 (做消融對照用, 每種各飛一次):
#   --mode truth   對照組: EV = 真值
#   --mode odom    只有會漂移的里程計, 視覺只記錄不修正
#   --mode vision  會漂移的里程計 + 視覺位置修正
#
# 參數設定輔助 (免 MAVProxy):
#   python3 scripts/vision_nav/repeat_mission.py --setup-ekf      # 寫入無 GPS (ExternalNav) EKF3 參數，寫完請重啟 SITL
#   python3 scripts/vision_nav/repeat_mission.py --restore-gps    # 還原正常 GPS EKF3 參數，寫完請重啟 SITL
#
# 輸出: scripts/vision_nav/runs/<模式>_<設定>_<時間>/ 下的 run.csv、matches.csv、summary.txt

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
import cv2
import numpy as np
from pymavlink import mavutil

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from utils import (
    DEFAULT_MAVLINK_CONN,
    DEFAULT_RTSP_URL,
    RTSPCamera,
    connect_mavlink,
    FeatureExtractor,
    send_vision_position,
    set_no_gps_ekf_params,
    restore_gps_ekf_params,
    EVTimeSync,
    latlon_to_ned,
    sky_ratio,
    edge_density,
)

# ═══════════════════ 設定 ═══════════════════
_ROOT = _THIS
DEFAULT_DS = os.path.join(_ROOT, "dataset_auto")

TARGET_ALT      = 250.0    # 250m AGL 避開台北 3D 建築
CLIMB_TIMEOUT_S = 180.0
MATCH_INTERVAL  = 0.5
WAYPOINT_EVERY  = 5
EV_HZ           = 20.0
CTRL_LOOP_HZ    = 10.0
SKY_QUERY_THR   = 0.90
EDGE_QUERY_MIN  = 0.005
GATE_MAX        = 150.0
XT_STALE_S      = 8.0      # 橫向量測超過這麼久沒更新就視為失效
XT_MIN_GOOD     = 12       # ORB 比對通過比值測試的最少點數

CRUISE_SPEED    = 9.5
SLOW_DIST       = 15.0
ARRIVE_DIST     = 5.0
MIN_APPROACH_V  = 2.5
INITIAL_YAW     = 148.0

YAW_EMA_ALPHA     = 0.20
YAW_RESEND_THR    = 5.0
YAW_RESEND_PERIOD = 1.0

# 影響 EKF 正確性的必選參數 (不符則中止; VISO_TYPE 與 EK3_SRC 需重啟 SITL 才生效)
REQUIRED_PARAMS = {
    "VISO_TYPE":      1.0,
    "EK3_SRC1_POSXY": 6.0,
    "EK3_SRC1_VELXY": 6.0,
    "EK3_SRC1_VELZ":  6.0,
    "EK3_SRC1_POSZ":  1.0,
    "EK3_SRC1_YAW":   1.0,
}
# ══════════════════════════════════════════════


def log(tag, msg):
    print(f"  [{tag}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="視覺重飛 (會漂移的里程計 + 視覺位置修正, 直連 SITL 免 MAVProxy)")
    p.add_argument("--mavlink", default=DEFAULT_MAVLINK_CONN,
                   help=f"MAVLink 連線字串 (預設: {DEFAULT_MAVLINK_CONN})")
    p.add_argument("--rtsp", default=DEFAULT_RTSP_URL,
                   help=f"ArduDeck RTSP 串流位址 (預設: {DEFAULT_RTSP_URL})")
    p.add_argument("--alt", type=float, default=TARGET_ALT,
                   help=f"起飛與巡航相對高度 m (預設: {TARGET_ALT})")
    p.add_argument("--dir", default=DEFAULT_DS,
                   help="資料集目錄 (預設: scripts/vision_nav/dataset_auto)")
    p.add_argument("--setup-ekf", action="store_true",
                   help="寫入無 GPS (ExternalNav) EKF3 參數並結束 (需重啟 SITL 生效)")
    p.add_argument("--restore-gps", action="store_true",
                   help="還原正常 GPS EKF3 參數並結束 (需重啟 SITL 生效)")
    p.add_argument("--mode", choices=["truth", "odom", "vision"], default="vision")
    p.add_argument("--seed", type=int, default=1, help="漂移亂數種子")
    p.add_argument("--scale-err", type=float, default=0.015, help="里程計尺度誤差 (0.015 = 1.5%%)")
    p.add_argument("--hdg-err-deg", type=float, default=0.2, help="里程計航向誤差 (度)")
    p.add_argument("--bias-rw", type=float, default=0.002, help="速度偏差隨機漫步 (m/s/√s)")
    p.add_argument("--pos-noise", type=float, default=0.02, help="位移雜訊 (m/√s)")
    p.add_argument("--fix-gain", type=float, default=0.12, help="每次視覺修正的增益 (0~1)")
    p.add_argument("--gate", type=float, default=35.0, help="修正閘門基準 (m)")
    p.add_argument("--gate-grow", type=float, default=5.0, help="閘門隨未修正時間放寬 (m/s)")
    p.add_argument("--score-min", type=float, default=0.85, help="相似度下限")
    p.add_argument("--match-r", type=float, default=300.0, help="候選幀半徑, 以 EKF 位置為中心 (m)")
    p.add_argument("--viso-nse", type=float, default=1.0, help="VISO_POS_M_NSE (m)")
    p.add_argument("--gps-glitch-deg", type=float, default=0.0,
                   help="飛行中注入 GPS 偏移 (度, 0=不做); 用來證明 EKF 不受 GPS 影響, 畫面可能會跳")
    p.add_argument("--glitch-at-wp", type=int, default=20, help="第幾個航點時注入 GPS 偏移")
    p.add_argument("--msg-hz", type=float, default=20.0,
                   help="把 GLOBAL_POSITION_INT / ATTITUDE / SIMSTATE 拉到這個頻率 (0 = 不設定, 沿用舊行為約 4 Hz)")
    p.add_argument("--fix-slew", type=float, default=0.0,
                   help="視覺修正滑入速率上限 (m/s); 0 = 一次跳到位 (舊行為)")
    p.add_argument("--xtrack", choices=["off", "log", "on"], default="off",
                   help="橫向量測: off 不做 | log 只記錄並校準 | on 使用 (需先校準)")
    p.add_argument("--cam-hfov-deg", type=float, default=60.0, help="相機水平視角 (度), 只用於 a 的粗估")
    p.add_argument("--scene-depth", type=float, default=350.0, help="場景典型深度 (m), 只用於粗估")
    p.add_argument("--xt-a", type=float, default=None, help="c = a·dx + b·dψ + c0 的 a (m/px)")
    p.add_argument("--xt-b", type=float, default=None, help="b (m/rad)")
    p.add_argument("--xt-c0", type=float, default=0.0, help="c0 (m)")
    p.add_argument("--xt-alpha", type=float, default=0.4, help="橫向量測的平滑係數")
    p.add_argument("--xt-min-inl", type=int, default=20, help="採用橫向量測的最少內點數")
    p.add_argument("--xt-max-mad", type=float, default=12.0, help="採用橫向量測的位移離散度上限 (px)")
    p.add_argument("--xt-max", type=float, default=60.0, help="橫向修正量上限 (m)")
    p.add_argument("--out", default="runs")
    return p.parse_args()


class MavReader:
    def __init__(self, conn):
        self.conn = conn
        self.lock = threading.Lock()
        self.latest = {}
        self.params = {}
        self.statustexts = []
        self.counts = {}
        self.running = True
        self.count = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            msg = self.conn.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue
            t = msg.get_type()
            if t == 'BAD_DATA':
                continue
            with self.lock:
                if t == 'PARAM_VALUE':
                    pid = msg.param_id
                    if isinstance(pid, bytes):
                        pid = pid.decode('utf-8', errors='ignore')
                    self.params[pid.strip('\x00').strip()] = msg.param_value
                    continue
                self.latest[t] = msg
                self.count += 1
                self.counts[t] = self.counts.get(t, 0) + 1
                if t == 'STATUSTEXT':
                    self.statustexts.append(msg)
                    self.statustexts = self.statustexts[-200:]

    def get(self, mtype):
        with self.lock:
            return self.latest.get(mtype)

    def counts_snapshot(self):
        with self.lock:
            return dict(self.counts)

    def get_param(self, name):
        with self.lock:
            return self.params.get(name)

    def get_statustexts(self):
        with self.lock:
            msgs = self.statustexts[:]
            self.statustexts.clear()
        return msgs

    def stop(self):
        self.running = False


def truth_ned(reader, ref):
    """SIMSTATE 真值 → 以 ref 為原點的 (north, east); 沒資料回傳 None"""
    ss = reader.get('SIMSTATE')
    if ss is None:
        return None
    lat = ss.lat / 1e7 if abs(ss.lat) > 1000 else ss.lat
    lon = ss.lng / 1e7 if abs(ss.lng) > 1000 else ss.lng
    x, y, _ = latlon_to_ned(lat, lon, ref[2], ref)
    return x, y


class LateralEstimator:
    """即時影像與匹配到的資料庫幀做 ORB 匹配, 回傳內點的水平位移 dx (像素, 以 640 寬為準)。
       dx > 0 代表場景在即時影像中比在資料庫幀中偏右。"""

    WIDTH = 640

    def __init__(self, ds, meta, nfeat=800):
        self.orb = cv2.ORB_create(nfeatures=nfeat)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.db = []
        usable = 0
        for i, m in enumerate(meta):
            img = cv2.imread(os.path.join(ds, "images", f"{m['index']:06d}.jpg"))
            d = self.describe(img) if img is not None else (None, None)
            usable += int(d[1] is not None)
            self.db.append(d)
            if (i + 1) % 100 == 0:
                log("XT", f"資料庫 ORB 特徵 {i + 1}/{len(meta)}")
        log("XT", f"資料庫可用幀 {usable}/{len(meta)}")

    def _prep(self, img):
        h, w = img.shape[:2]
        s = self.WIDTH / w
        g = cv2.cvtColor(cv2.resize(img, (self.WIDTH, max(1, int(h * s)))), cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(g)

    def describe(self, img):
        kp, des = self.orb.detectAndCompute(self._prep(img), None)
        if des is None or len(kp) < 8:
            return None, None
        return np.float32([k.pt for k in kp]), des

    def measure(self, live_img, k):
        """回傳 (dx_px, 內點數, 位移離散度 MAD px); 失敗回傳 None"""
        pts_l, des_l = self.describe(live_img)
        pts_k, des_k = self.db[k]
        if des_l is None or des_k is None:
            return None
        knn = self.bf.knnMatch(des_k, des_l, k=2)
        good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < XT_MIN_GOOD:
            return None
        a = np.float32([pts_k[m.queryIdx] for m in good])
        b = np.float32([pts_l[m.trainIdx] for m in good])
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=6.0)
        if M is None or inl is None:
            return None
        inl = inl.ravel().astype(bool)
        if inl.sum() < 8:
            return None
        dxs = b[inl, 0] - a[inl, 0]
        dx = float(np.median(dxs))
        return dx, int(inl.sum()), float(np.median(np.abs(dxs - dx)))


class EVSource(threading.Thread):
    """以 EV_HZ 把位置餵給 EKF。
       truth 模式: 位置 = 真值
       odom/vision 模式: 位置 = 會漂移的里程計積分 (vision 模式再被視覺修正拉回)"""

    def __init__(self, conn, sync, reader, ref, args):
        super().__init__(daemon=True)
        self.conn, self.sync, self.reader, self.ref = conn, sync, reader, ref
        self.mode = args.mode
        th = math.radians(args.hdg_err_deg)
        self.M = (1.0 + args.scale_err) * np.array(
            [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        self.bias_rw = args.bias_rw
        self.noise = args.pos_noise
        self.rng = np.random.default_rng(args.seed)
        self.lock = threading.Lock()
        self.p = np.zeros(2)
        self.bias = np.zeros(2)
        self.prev = None
        self.prev_t = 0.0
        self.sent = 0
        self.running = True
        self.slew = args.fix_slew
        self.pending = np.zeros(2)
        self.last_slew_t = time.time()

    def get_p(self):
        with self.lock:
            return self.p.copy()

    def get_target(self):
        with self.lock:
            return self.p + self.pending

    def apply_fix(self, fix, gain):
        with self.lock:
            if self.slew <= 0:
                self.p = self.p + gain * (fix - self.p)
            else:
                self.pending = self.pending + gain * (fix - (self.p + self.pending))

    def _slide_pending(self, now):
        with self.lock:
            dt = now - self.last_slew_t
            self.last_slew_t = now
            n = float(np.hypot(*self.pending))
            if self.slew > 0 and n > 1e-6:
                move = self.pending * min(1.0, self.slew * dt / n)
                self.p = self.p + move
                self.pending = self.pending - move

    def _step(self, now):
        self._slide_pending(now)
        tr = truth_ned(self.reader, self.ref)
        if tr is None:
            return
        cur = np.array(tr, dtype=float)
        with self.lock:
            if self.mode == 'truth' or self.prev is None:
                self.p = cur.copy()
            else:
                dt = max(1e-4, now - self.prev_t)
                self.bias = self.bias + self.rng.normal(0.0, self.bias_rw * math.sqrt(dt), 2)
                self.p = (self.p + self.M @ (cur - self.prev) + self.bias * dt
                          + self.rng.normal(0.0, self.noise * math.sqrt(dt), 2))
            self.prev, self.prev_t = cur, now

    def run(self):
        while self.running:
            t0 = time.time()
            self._step(t0)
            gp = self.reader.get('GLOBAL_POSITION_INT')
            z = -(gp.relative_alt / 1000.0) if gp else 0.0
            x, y = self.get_p()
            send_vision_position(self.conn, float(x), float(y), z, self.sync)
            self.sent += 1
            time.sleep(max(0.0, 1.0 / EV_HZ - (time.time() - t0)))

    def stop(self):
        self.running = False


# ──────────────────────────────────────────────
#  MAVLink 輔助
# ──────────────────────────────────────────────
def set_param(conn, name, value):
    conn.mav.param_set_send(conn.target_system, conn.target_component,
                            name.encode(), float(value),
                            mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    time.sleep(0.15)


def read_param(reader, conn, name, timeout=2.0):
    v = reader.get_param(name)
    if v is not None:
        return v
    conn.mav.param_request_read_send(conn.target_system, conn.target_component,
                                     name.encode(), -1)
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = reader.get_param(name)
        if v is not None:
            return v
        time.sleep(0.05)
    return None


def wait_for_param_sync(reader, timeout=20.0, stable_sec=1.5):
    t0 = time.time()
    last_count, last_change = 0, time.time()
    while time.time() - t0 < timeout:
        cur = len(reader.params)
        if cur != last_count:
            last_count, last_change = cur, time.time()
        elif time.time() - last_change > stable_sec:
            return True, cur
        time.sleep(0.2)
    return False, len(reader.params)


def preflight_check(reader, conn):
    print("=" * 60, flush=True)
    stable, cnt = wait_for_param_sync(reader)
    log("PARAM", f"參數 {cnt} 條" + ("" if stable else " (廣播未穩定, 繼續驗證)"))

    gps_en = read_param(reader, conn, "SIM_GPS1_ENABLE")
    if gps_en is not None and gps_en > 0.5:
        log("PARAM", "SIM_GPS1_ENABLE=1: GPS 仍在發訊 (渲染需要), EKF 因來源設定而不採用")

    ok = True
    for name, want in REQUIRED_PARAMS.items():
        actual = read_param(reader, conn, name)
        if actual is None or abs(actual - want) > 1e-3:
            ok = False
            log("PARAM", f"❌ {name}: 期望 {want:g}, 實際 {'讀取逾時' if actual is None else f'{actual:g}'}")
        else:
            log("PARAM", f"✓ {name} = {actual:g}")
    if not ok:
        log("PARAM", "必選參數不符, 請先執行: python3 scripts/vision_nav/repeat_mission.py --setup-ekf 並重啟 SITL")
    print("=" * 60, flush=True)
    return ok


def set_message_intervals(conn, hz):
    """MAV_CMD_SET_MESSAGE_INTERVAL (511): 位置/姿態/真值預設只有約 4 Hz, 控制迴路會吃到舊資料"""
    for msg_id in (33, 30, 164):   # GLOBAL_POSITION_INT, ATTITUDE, SIMSTATE
        conn.mav.command_long_send(conn.target_system, conn.target_component,
                                   511, 0, msg_id, 1e6 / hz, 0, 0, 0, 0, 0)
        time.sleep(0.05)


def measure_rates(reader, names, secs=2.0):
    c0 = reader.counts_snapshot()
    time.sleep(secs)
    c1 = reader.counts_snapshot()
    return {n: (c1.get(n, 0) - c0.get(n, 0)) / secs for n in names}


def set_ekf_origin(conn, lat, lon, alt_m):
    conn.mav.set_gps_global_origin_send(
        conn.target_system, int(lat * 1e7), int(lon * 1e7), int(alt_m * 1000))
    time.sleep(0.5)


def set_guided(conn):
    conn.mav.set_mode_send(
        conn.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)
    time.sleep(0.5)


def wait_mode(reader, mode, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hb = reader.get('HEARTBEAT')
        if hb and hb.custom_mode == mode:
            return True
        time.sleep(0.1)
    return False


def wait_armed(reader, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hb = reader.get('HEARTBEAT')
        if hb and (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            return True
        time.sleep(0.1)
    return False


def drain_statustext(reader):
    for st in reader.get_statustexts():
        txt = st.text.decode() if isinstance(st.text, bytes) else st.text
        log("AP", txt)


def send_condition_yaw(conn, yaw_deg, speed=30.0):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0, float(yaw_deg), speed, 1, 0, 0, 0, 0)


def send_velocity_ned(conn, vx, vy, vz):
    type_mask = 0b0000111111000111
    conn.mav.set_position_target_local_ned_send(
        0, conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, type_mask,
        0.0, 0.0, 0.0, float(vx), float(vy), float(vz),
        0.0, 0.0, 0.0, 0.0, 0.0)


def land(conn):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)


def wait_for_simstate(reader, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        ss = reader.get('SIMSTATE')
        if ss is not None:
            lat = ss.lat / 1e7 if abs(ss.lat) > 1000 else ss.lat
            lon = ss.lng / 1e7 if abs(ss.lng) > 1000 else ss.lng
            return (lat, lon, 0.0)
        time.sleep(0.1)
    return None


def angle_diff(a, b):
    return (a - b + 180) % 360 - 180


def auto_arm_takeoff(mav, reader, target_alt, tries=3):
    # 提高爬升速度以加快爬升至 250m
    for name, val in (("WPNAV_SPEED_UP", 600.0), ("WP_SPD_UP", 6.0)):
        set_param(mav, name, val)

    armed = False
    for attempt in range(1, tries + 1):
        set_guided(mav)
        if not wait_mode(reader, 4, timeout=5):
            log("MODE", f"Guided 切換失敗 ({attempt}/{tries})")
            time.sleep(2)
            continue
        mav.mav.command_long_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
        if wait_armed(reader, timeout=8):
            log("ARM", "✅ ARMED")
            armed = True
            break
        log("ARM", f"解鎖失敗 ({attempt}/{tries})")
        drain_statustext(reader)
        time.sleep(2)
    if not armed:
        return False

    log("TAKEOFF", f"起飛至 {target_alt:.0f}m...")
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, target_alt)

    t0, last_print = time.time(), 0.0
    while time.time() - t0 < CLIMB_TIMEOUT_S:
        drain_statustext(reader)
        gp = reader.get('GLOBAL_POSITION_INT')
        if gp:
            alt = gp.relative_alt / 1000.0
            if time.time() - last_print > 5:
                log("CLIMB", f"alt={alt:.1f} / {target_alt:.0f}m")
                last_print = time.time()
            if abs(alt - target_alt) < 5:
                log("CLIMB", f"✅ 到達 {target_alt:.0f}m")
                send_condition_yaw(mav, INITIAL_YAW)
                time.sleep(2)
                return True
        time.sleep(0.2)
    return False


def stat_line(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return "(無資料)"
    return (f"平均 {a.mean():.2f} | 中位 {np.median(a):.2f} | "
            f"95% {np.percentile(a, 95):.2f} | 最大 {a.max():.2f}")


def fnum(v, nd=1):
    return "" if v != v else f"{v:.{nd}f}"


def xt_report(samples, args, a, b, c0):
    """samples: (dx, dψ_rad, c_true, n_inl, mad, c_est)"""
    lines = [f"  橫向量測 (xtrack={args.xtrack}): 使用 a={a:.4f} m/px, b={b:.1f} m/rad, c0={c0:.2f} m"]
    S = [x for x in samples if x[3] >= args.xt_min_inl and x[4] <= args.xt_max_mad]
    lines.append(f"  橫向量測樣本: 有真值 {len(samples)} 筆, 品質合格 {len(S)} 筆")
    if len(S) < 30:
        lines.append("  樣本太少 (< 30), 無法擬合; 檢查 ORB 匹配是否成功 (內點數 / 位移離散度)")
        return lines
    arr = np.array(S, dtype=float)
    y = arr[:, 2]
    est_err = np.abs(arr[:, 5] - y)
    lines.append(f"  目前參數的橫向估計誤差 |c_est − c_true| (m): {stat_line(est_err)}")
    lines.append(f"  真實橫向偏離 c_true 的範圍: 標準差 {y.std():.1f} m, 最小 {y.min():.1f}, 最大 {y.max():.1f}")
    X = np.column_stack([arr[:, 0], arr[:, 1], np.ones(len(arr))])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    r2 = 1.0 - float(res.var() / (y.var() + 1e-9))
    lines.append(f"  擬合 c_true ≈ a·dx + b·dψ + c0:  a={coef[0]:.4f}  b={coef[1]:.1f}  c0={coef[2]:.2f}  "
                 f"R²={r2:.2f}  殘差標準差 {res.std():.1f} m")
    if r2 >= 0.3 and y.std() > 5.0:
        lines.append(f"  建議參數: --xt-a {coef[0]:.4f} --xt-b {coef[1]:.1f} --xt-c0 {coef[2]:.2f}")
    elif y.std() <= 5.0:
        lines.append("  真實橫向偏離太小 (標準差 ≤ 5 m), 無法校準; 請用較大的 --hdg-err-deg (例如 1.0) 飛 odom 模式")
    else:
        lines.append("  擬合不佳 (R² < 0.3): 橫向量測目前不可靠, 不建議開 --xtrack on")
    return lines


RUN_COLS = ["t", "wp", "truth_n", "truth_e", "ekf_n", "ekf_e", "odom_n", "odom_e",
            "ekf_err_m", "truth_route_dist_m", "truth_idx", "ekf_idx",
            "cmd_speed", "alt_m", "yaw_deg", "gps_lat", "gps_lon", "fix_applied", "fix_rejected"]
MATCH_COLS = ["t", "k", "score", "cand", "fix_n", "fix_e", "est_n", "est_e",
              "innov_m", "allow_m", "ok", "applied", "err_vs_truth_m", "err_vs_ekf_m",
              "dx_px", "n_inl", "mad_px", "dyaw_deg", "c_est_m", "c_true_m", "xt_used", "key_yaw_meta"]


def main():
    args = parse_args()

    if args.setup_ekf:
        mav = connect_mavlink(args.mavlink)
        set_no_gps_ekf_params(mav)
        return
    if args.restore_gps:
        mav = connect_mavlink(args.mavlink)
        restore_gps_ekf_params(mav)
        return

    ds = args.dir
    target_alt = args.alt

    tag = args.mode
    tag += f"_alt{target_alt:g}"
    tag += f"_hz{args.msg_hz:g}" if args.msg_hz > 0 else "_hzdefault"
    if args.fix_slew > 0:
        tag += f"_slew{args.fix_slew:g}"
    if args.mode != 'truth':
        tag += f"_hdg{args.hdg_err_deg:g}"
    if args.xtrack != 'off':
        tag += f"_xt{args.xtrack}"
    out_dir = os.path.join(_ROOT, args.out, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    log("RUN", f"模式={args.mode}  高度={target_alt:.0f}m  輸出={out_dir}")
    if args.xtrack == 'on' and (args.xt_a is None or args.xt_b is None):
        log("XT", "❌ --xtrack on 需要明確給 --xt-a 與 --xt-b; 請先用 --xtrack log 校準")
        return
    if args.mode != 'truth':
        log("RUN", f"漂移: 尺度 {args.scale_err*100:.1f}% | 航向 {args.hdg_err_deg:.2f}° | "
                   f"偏差漫步 {args.bias_rw} | 雜訊 {args.pos_noise} | seed {args.seed}")

    pm = os.path.join(ds, "processed_metadata.json")
    if not os.path.exists(pm):
        pm = os.path.join(ds, "metadata.json")
    with open(pm) as f:
        meta = json.load(f)
    log("DB", f"載入 {len(meta)} 幀 ({pm})")

    ex = FeatureExtractor()
    feats = []
    for m in meta:
        fp = os.path.join(ds, "features", f"{m['index']:06d}.npy")
        if os.path.exists(fp):
            feats.append(np.load(fp))
        else:
            feats.append(ex.extract(cv2.imread(os.path.join(ds, "images", f"{m['index']:06d}.jpg"))))
    db_features = np.array(feats)
    log("DB", f"特徵庫 {db_features.shape}")

    camera = RTSPCamera(args.rtsp)
    mav = connect_mavlink(args.mavlink)
    sync = EVTimeSync(mav)
    reader = MavReader(mav)
    mav.mav.request_data_stream_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)

    t0 = time.time()
    while reader.get('HEARTBEAT') is None and time.time() - t0 < 5:
        time.sleep(0.2)
    log("CONN", f"Heartbeat 正常, 已收 {reader.count} 條訊息")

    if args.msg_hz > 0:
        set_message_intervals(mav, args.msg_hz)
    rates = measure_rates(reader, ["GLOBAL_POSITION_INT", "ATTITUDE", "SIMSTATE"])
    log("CONN", "訊息更新率: " + " | ".join(f"{k} {v:.1f} Hz" for k, v in rates.items()))
    if args.msg_hz > 0 and rates["GLOBAL_POSITION_INT"] < 0.7 * args.msg_hz:
        log("CONN", "⚠️ GLOBAL_POSITION_INT 沒有達到設定頻率, 結果仍會有位置延遲")

    mav.mav.param_request_list_send(mav.target_system, mav.target_component)
    time.sleep(1.0)
    if not preflight_check(reader, mav):
        reader.stop()
        camera.release()
        return

    set_param(mav, "VISO_POS_M_NSE", args.viso_nse)
    nse = read_param(reader, mav, "VISO_POS_M_NSE")
    log("PARAM", f"VISO_POS_M_NSE = {nse}" if nse is not None else "⚠️ VISO_POS_M_NSE 讀不到, 沿用現值")

    ph = wait_for_simstate(reader, timeout=10)
    if ph is None:
        log("REF", "❌ 收不到 SIMSTATE (里程計模擬與評估都需要), 中止")
        reader.stop()
        camera.release()
        return
    ref = ph
    log("REF", f"物理 home = ({ref[0]:.7f}, {ref[1]:.7f})")

    db_ned = np.array([latlon_to_ned(m["lat"], m["lon"], ref[2], ref)[:2] for m in meta])
    spacing = float(np.hypot(*np.diff(db_ned, axis=0).T).mean())
    log("DB", f"航線約 {spacing * (len(meta) - 1):.0f} m, 幀距約 {spacing:.1f} m")

    tang = np.zeros_like(db_ned)
    for i in range(len(db_ned)):
        lo, hi = max(0, i - 3), min(len(db_ned) - 1, i + 3)
        v = db_ned[hi] - db_ned[lo]
        tang[i] = v / (float(np.hypot(*v)) + 1e-9)
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)      # 航線右側法向量 (north, east)
    bearing = np.degrees(np.arctan2(tang[:, 1], tang[:, 0])) % 360.0

    lat_est = None
    xt_a, xt_b, xt_c0 = args.xt_a, args.xt_b, args.xt_c0
    if args.xtrack != 'off':
        f_px = (LateralEstimator.WIDTH / 2) / math.tan(math.radians(args.cam_hfov_deg) / 2)
        if xt_a is None:
            xt_a = -args.scene_depth / f_px
        if xt_b is None:
            xt_b = -args.scene_depth
        log("XT", f"橫向量測 {args.xtrack}: a={xt_a:.4f} m/px, b={xt_b:.1f} m/rad, c0={xt_c0:.2f} m")
        lat_est = LateralEstimator(ds, meta)

    wps_idx = list(range(0, len(meta), WAYPOINT_EVERY))
    if wps_idx[-1] != len(meta) - 1:
        wps_idx.append(len(meta) - 1)
    log("DB", f"航點數 {len(wps_idx)}")

    gp = reader.get('GLOBAL_POSITION_INT')
    origin_amsl = (gp.alt - gp.relative_alt) / 1000.0 if gp else 0.0
    set_ekf_origin(mav, ref[0], ref[1], origin_amsl)
    log("REF", f"EKF 原點 = 物理 home (AMSL {origin_amsl:.1f}m)")

    ev = EVSource(mav, sync, reader, ref, args)
    ev.start()
    log("EV", f"EV 來源啟動 (模式 {args.mode}), 預熱 3 秒")
    time.sleep(3)

    run_f = open(os.path.join(out_dir, "run.csv"), "w", newline="")
    match_f = open(os.path.join(out_dir, "matches.csv"), "w", newline="")
    run_w, match_w = csv.writer(run_f), csv.writer(match_f)
    run_w.writerow(RUN_COLS)
    match_w.writerow(MATCH_COLS)

    ekf_errs, route_devs, idx_lags = [], [], []
    xt_samples = []
    fix_err_all, fix_err_applied = [], []
    n_match = n_ok = n_applied = n_rej = 0
    wp_i = 0
    started = False
    glitched = False
    t_start = time.time()

    try:
        if not auto_arm_takeoff(mav, reader, target_alt):
            log("TAKEOFF", "❌ 起飛失敗")
            return

        started = True
        t_start = time.time()
        print("=" * 60, flush=True)
        print(f"  REPEAT  模式={args.mode}  高度 {target_alt:.0f}m  巡航 {CRUISE_SPEED:.1f} m/s", flush=True)
        print("=" * 60, flush=True)

        last_match = 0.0
        last_prog = 0.0
        last_accept_t = time.time()
        xt_cf, xt_ct = 0.0, -1e9
        yaw_smooth, yaw_last_sent, yaw_last_t, yaw_first = INITIAL_YAW, INITIAL_YAW, 0.0, True
        speed = 0.0

        while wp_i < len(wps_idx):
            now = time.time()
            wp_x, wp_y = db_ned[wps_idx[wp_i]]
            drain_statustext(reader)

            gp = reader.get('GLOBAL_POSITION_INT')
            if gp is None:
                time.sleep(1.0 / CTRL_LOOP_HZ)
                continue

            ekf_x, ekf_y, _ = latlon_to_ned(gp.lat / 1e7, gp.lon / 1e7, ref[2], ref)
            cur_alt = gp.relative_alt / 1000.0
            att = reader.get('ATTITUDE')
            cur_yaw = math.degrees(att.yaw) % 360.0 if att else 0.0

            # ── 控制: 只用 EKF 位置 ──
            dx, dy = wp_x - ekf_x, wp_y - ekf_y
            dist = math.hypot(dx, dy)
            ux, uy = (dx / dist, dy / dist) if dist > 1.0 else (0.0, 0.0)
            if dist > SLOW_DIST:
                speed = CRUISE_SPEED
            elif dist > ARRIVE_DIST:
                frac = (dist - ARRIVE_DIST) / (SLOW_DIST - ARRIVE_DIST)
                speed = MIN_APPROACH_V + (CRUISE_SPEED - MIN_APPROACH_V) * frac
            else:
                speed = 0.0
            vz = max(-1.5, min(1.5, -0.6 * (target_alt - cur_alt)))
            send_velocity_ned(mav, ux * speed, uy * speed, vz)

            yaw_raw = math.degrees(math.atan2(dy, dx)) % 360.0
            if yaw_first:
                yaw_smooth, yaw_first = yaw_raw, False
            else:
                yaw_smooth = (yaw_smooth + YAW_EMA_ALPHA * angle_diff(yaw_raw, yaw_smooth)) % 360.0
            if (abs(angle_diff(yaw_smooth, yaw_last_sent)) > YAW_RESEND_THR
                    or now - yaw_last_t > YAW_RESEND_PERIOD):
                send_condition_yaw(mav, yaw_smooth)
                yaw_last_sent, yaw_last_t = yaw_smooth, now

            # ── GPS 注入偏移測試 (可選) ──
            if args.gps_glitch_deg and not glitched and wp_i >= args.glitch_at_wp:
                set_param(mav, "SIM_GPS1_GLTCH_X", args.gps_glitch_deg)
                set_param(mav, "SIM_GPS1_GLTCH_Y", args.gps_glitch_deg)
                glitched = True
                log("GPS", f"已注入 GPS 偏移 {args.gps_glitch_deg}° (航點 {wp_i})")

            # ── 評估用: 真值只寫進紀錄, 不回饋控制 ──
            tr = truth_ned(reader, ref)
            ekf_idx = int(np.argmin(np.hypot(db_ned[:, 0] - ekf_x, db_ned[:, 1] - ekf_y)))
            ekf_err = route_dev = truth_idx = float('nan')
            if tr is not None:
                d_truth = np.hypot(db_ned[:, 0] - tr[0], db_ned[:, 1] - tr[1])
                truth_idx = int(np.argmin(d_truth))
                route_dev = float(d_truth[truth_idx])
                ekf_err = math.hypot(ekf_x - tr[0], ekf_y - tr[1])
                ekf_errs.append(ekf_err)
                route_devs.append(route_dev)
                idx_lags.append(truth_idx - ekf_idx)
            p = ev.get_p()
            gps = reader.get('GPS_RAW_INT')
            run_w.writerow([f"{now - t_start:.2f}", wp_i,
                            f"{tr[0]:.2f}" if tr else "", f"{tr[1]:.2f}" if tr else "",
                            f"{ekf_x:.2f}", f"{ekf_y:.2f}", f"{p[0]:.2f}", f"{p[1]:.2f}",
                            f"{ekf_err:.2f}", f"{route_dev:.2f}", truth_idx, ekf_idx,
                            f"{speed:.2f}", f"{cur_alt:.1f}", f"{cur_yaw:.1f}",
                            f"{gps.lat / 1e7:.7f}" if gps else "", f"{gps.lon / 1e7:.7f}" if gps else "",
                            n_applied, n_rej])

            # ── 視覺匹配: 候選以 EKF 位置為中心 ──
            if now - last_match >= MATCH_INTERVAL:
                last_match = now
                frame = camera.read()
                if frame is not None:
                    sr, ed = sky_ratio(frame), edge_density(frame)
                    if sr < SKY_QUERY_THR and ed >= EDGE_QUERY_MIN:
                        feat = ex.extract(frame)
                        cand = np.where(np.hypot(db_ned[:, 0] - ekf_x,
                                                 db_ned[:, 1] - ekf_y) < args.match_r)[0]
                        if len(cand) == 0:
                            cand = np.arange(len(meta))
                        sub = db_features[cand] @ feat
                        bi = int(np.argmax(sub))
                        k, score = int(cand[bi]), float(sub[bi])
                        fix = db_ned[k]
                        est = ev.get_target()

                        xt_dx = xt_n = xt_mad = xt_c = xt_ctrue = xt_dyaw = float('nan')
                        xt_used = 0
                        if lat_est is not None:
                            meas = lat_est.measure(frame, k)
                            xt_dyaw = angle_diff(cur_yaw, float(bearing[k]))
                            if meas is not None:
                                xt_dx, xt_n, xt_mad = meas
                                xt_c = xt_a * xt_dx + xt_b * math.radians(xt_dyaw) + xt_c0
                                if tr is not None:
                                    xt_ctrue = float((np.array(tr) - db_ned[k]) @ normal[k])
                                    xt_samples.append((xt_dx, math.radians(xt_dyaw), xt_ctrue, xt_n, xt_mad, xt_c))
                                if (xt_n >= args.xt_min_inl and xt_mad <= args.xt_max_mad
                                        and abs(xt_c) <= args.xt_max):
                                    if now - xt_ct > XT_STALE_S:
                                        xt_cf = xt_c
                                    else:
                                        xt_cf = (1 - args.xt_alpha) * xt_cf + args.xt_alpha * xt_c
                                    xt_ct = now

                        fix_eff = fix
                        if args.xtrack == 'on' and args.mode == 'vision':
                            if now - xt_ct < XT_STALE_S:
                                fix_eff = fix + xt_cf * normal[k]
                            else:
                                fix_eff = est + float((fix - est) @ tang[k]) * tang[k]
                            xt_used = 1
                        innov = float(np.hypot(*(fix_eff - est)))
                        allow = min(args.gate + args.gate_grow * (now - last_accept_t), GATE_MAX)
                        ok = score >= args.score_min and innov <= allow
                        applied = ok and args.mode == 'vision'
                        if applied:
                            ev.apply_fix(fix_eff, args.fix_gain)
                            last_accept_t = now
                            n_applied += 1
                        elif args.mode == 'vision':
                            n_rej += 1
                        n_match += 1
                        n_ok += int(ok)
                        err_truth = math.hypot(fix_eff[0] - tr[0], fix_eff[1] - tr[1]) if tr else float('nan')
                        err_ekf = math.hypot(fix_eff[0] - ekf_x, fix_eff[1] - ekf_y)
                        if tr:
                            fix_err_all.append(err_truth)
                            if applied:
                                fix_err_applied.append(err_truth)
                        match_w.writerow([f"{now - t_start:.2f}", k, f"{score:.3f}", len(cand),
                                          f"{fix_eff[0]:.2f}", f"{fix_eff[1]:.2f}", f"{est[0]:.2f}", f"{est[1]:.2f}",
                                          f"{innov:.1f}", f"{allow:.1f}", int(ok), int(applied),
                                          f"{err_truth:.1f}", f"{err_ekf:.1f}",
                                          fnum(xt_dx), fnum(xt_n, 0), fnum(xt_mad), fnum(xt_dyaw), fnum(xt_c),
                                          fnum(xt_ctrue), xt_used, meta[k].get("yaw", "")])
                    else:
                        log("MATCH", f"⛔ 天空={sr:.2f} 邊緣={ed:.3f}")

            if dist < ARRIVE_DIST + 2.0:
                wp_i += 1
                log("NAV", f"✅ 航點 {wp_i}/{len(wps_idx)}")

            if now - last_prog > 5:
                last_prog = now
                recent_err = np.mean(ekf_errs[-50:]) if ekf_errs else float('nan')
                recent_dev = np.mean(route_devs[-50:]) if route_devs else float('nan')
                log("PROG", f"航點 {wp_i}/{len(wps_idx)} | EKF 對真值 {recent_err:.1f}m | "
                            f"真值偏離航線 {recent_dev:.1f}m | 修正 {n_applied} 次 拒絕 {n_rej} 次")

            time.sleep(1.0 / CTRL_LOOP_HZ)

        log("NAV", "🏁 完成, 降落")
        land(mav)

    except KeyboardInterrupt:
        log("NAV", "中斷, 降落")
        land(mav)

    finally:
        if glitched:
            set_param(mav, "SIM_GPS1_GLTCH_X", 0)
            set_param(mav, "SIM_GPS1_GLTCH_Y", 0)
        ev.stop()
        reader.stop()
        camera.release()
        run_f.close()
        match_f.close()

        if started:
            lines = ["=" * 60,
                     f"  REPEAT 評估報告  模式={args.mode}  高度={target_alt:.0f}m",
                     "=" * 60,
                     f"  設定: msg-hz={args.msg_hz:g} | fix-slew={args.fix_slew:g} | "
                     f"漂移 尺度{args.scale_err * 100:.1f}% 航向{args.hdg_err_deg:g}° | xtrack={args.xtrack} | "
                     f"實測更新率 " + " / ".join(f"{v:.0f}" for v in rates.values()) + " Hz (位置/姿態/真值)",
                     f"  完成航點: {wp_i}/{len(wps_idx)} | 歷時約 {time.time() - t_start:.0f} s",
                     f"  EKF 位置對真值誤差 (m): {stat_line(ekf_errs)}",
                     f"  真值偏離航線 (m):       {stat_line(route_devs)}  (含幀距量化下限約 {spacing / 4:.1f} m)"]
            if idx_lags:
                lag_m = np.abs(np.array(idx_lags)) * spacing
                lines.append(f"  沿航線誤差 (|真值幀-EKF幀|×幀距, m): {stat_line(lag_m)}")
            lines.append(f"  匹配嘗試 {n_match} | 通過閘門 {n_ok} | 實際修正 {n_applied}")
            lines.append(f"  匹配幀座標對真值誤差 (m), 全部: {stat_line(fix_err_all)}")
            if args.mode == 'vision':
                lines.append(f"  匹配幀座標對真值誤差 (m), 已採用: {stat_line(fix_err_applied)}")
            if lat_est is not None:
                lines += xt_report(xt_samples, args, xt_a, xt_b, xt_c0)
            lines.append("=" * 60)
            report = "\n".join(lines)
            print("\n" + report)
            with open(os.path.join(out_dir, "summary.txt"), "w") as f:
                f.write(report + "\n")
        log("RUN", f"紀錄已存到 {out_dir}")


if __name__ == "__main__":
    main()
