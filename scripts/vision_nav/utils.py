# utils.py (v23 — 直連 ArduPilot SITL TCP 5762，免 MAVProxy，支援 macOS MPS / Windows CUDA / CPU)
#
# 改動說明:
#   1. connect_mavlink 預設直連 tcp:127.0.0.1:5762 (SITL SERIAL1)，免開 MAVProxy
#      連線後主動發送 GCS HEARTBEAT + REQUEST_DATA_STREAM，確保 SITL 立即回傳遙測
#   2. FeatureExtractor 自動選擇 cuda / mps / cpu (相容 Mac M1 與 Windows NVIDIA GPU)
#   3. 保留 v22 EVTimeSync (monotonic + 週期 resync)、非零協方差 EV 發送與影像品質指標

import cv2
import time
import threading
import queue
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from pymavlink import mavutil
import math


DEFAULT_MAVLINK_CONN = "tcp:127.0.0.1:5762"
DEFAULT_RTSP_URL = "rtsp://127.0.0.1:8554/vision"


# ═══════════════════════════════════════════════
#  1. 零延遲 RTSP 攝影機
# ═══════════════════════════════════════════════
class RTSPCamera:
    def __init__(self, rtsp_url=DEFAULT_RTSP_URL):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = queue.Queue(maxsize=2)
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            self.q.put(frame)
        self.cap.release()

    def read(self):
        try:
            return self.q.get(timeout=2.0)
        except queue.Empty:
            return None

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=3.0)


# ═══════════════════════════════════════════════
#  2. 影像預處理與特徵提取 (edge+clahe)
# ═══════════════════════════════════════════════
class FeatureExtractor:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"[FeatureExtractor] 使用裝置: {self.device} | 管線: edge+clahe", flush=True)

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def preprocess(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.clip(np.sqrt(gx ** 2 + gy ** 2), 0, None)
        mag = (mag / (mag.max() + 1e-6) * 255).astype(np.uint8)
        mag_eq = self.clahe.apply(mag)
        return cv2.cvtColor(mag_eq, cv2.COLOR_GRAY2BGR)

    @torch.no_grad()
    def extract(self, frame_bgr):
        enhanced = self.preprocess(frame_bgr)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        feat = self.model(tensor).squeeze()
        feat = feat / (feat.norm() + 1e-8)
        return feat.cpu().numpy()


# ═══════════════════════════════════════════════
#  3. MAVLink 直連 (免 MAVProxy)
# ═══════════════════════════════════════════════
def _try_connect(conn_string, timeout):
    conn = mavutil.mavlink_connection(conn_string, source_system=254, source_component=190)
    # 主動送出一筆 GCS HEARTBEAT，喚醒 SITL TCP SERIAL1/SERIAL2 輸出串流
    try:
        conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0,
        )
    except Exception:
        pass
    hb = conn.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
    if hb is None:
        try:
            conn.close()
        except Exception:
            pass
        return None
    # 主動請求資料串流（免 MAVProxy 時 SITL 副埠預設不會主動推播全量遙測）
    conn.mav.request_data_stream_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1,
    )
    return conn


def connect_mavlink(conn_string=DEFAULT_MAVLINK_CONN, timeout=10):
    candidates = [conn_string]
    if conn_string == DEFAULT_MAVLINK_CONN:
        candidates.append("tcp:127.0.0.1:5763")

    last_err = None
    for cs in candidates:
        try:
            conn = _try_connect(cs, timeout=timeout if cs == conn_string else 4)
            if conn is not None:
                print(f"[MAVLink] 直連成功: {cs} (sysid={conn.target_system}, 免 MAVProxy)", flush=True)
                return conn
        except Exception as e:
            last_err = e

    raise SystemExit(
        f"[錯誤] {timeout} 秒內無法連線或收不到 heartbeat（嘗試過: {', '.join(candidates)}）。\n"
        f"  最後錯誤: {last_err}\n"
        f"  請確認啟動順序：ArduDeck（連 SITL 5760）→ SITL 已啟動 → 開啟 RTSP → 執行本腳本。\n"
        f"  若 SITL 使用其他埠，請加上參數 --mavlink tcp:127.0.0.1:5762"
    )


def send_gps_input(conn, lat, lon, alt, horiz_acc=5.0):
    time_usec = int(time.time() * 1e6)
    conn.mav.gps_input_send(
        time_usec, 0, 1 | 16, 0, 0, 3,
        int(lat * 1e7), int(lon * 1e7), alt,
        horiz_acc, 5.0, 0.0, 0.0, 0.0, 1.0, horiz_acc, 5.0, 0
    )


def set_no_gps_ekf_params(conn):
    """設定 EKF3 切換為外部視覺 (ExternalNav, 不參考 GPS)，設定完需重啟 SITL 生效"""
    print("[MAVLink] 正在寫入無 GPS (ExternalNav) EKF3 參數...", flush=True)
    params = {
        b'VISO_TYPE': 1.0,
        b'EK3_SRC1_POSXY': 6.0,
        b'EK3_SRC1_VELXY': 6.0,
        b'EK3_SRC1_VELZ': 6.0,
        b'EK3_SRC1_POSZ': 1.0,
        b'EK3_SRC1_YAW': 1.0,
    }
    for param, value in params.items():
        conn.mav.param_set_send(
            conn.target_system, conn.target_component,
            param, float(value), mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )
        time.sleep(0.2)
    print("[MAVLink] ✅ 已寫入無 GPS 參數，請重啟 SITL 後再執行 repeat_mission.py", flush=True)


def restore_gps_ekf_params(conn):
    """還原 EKF3 為正常 GPS 導航 (給 record_auto.py 錄製前使用)，設定完需重啟 SITL 生效"""
    print("[MAVLink] 正在還原正常 GPS EKF3 參數...", flush=True)
    params = {
        b'VISO_TYPE': 0.0,
        b'EK3_SRC1_POSXY': 3.0,
        b'EK3_SRC1_VELXY': 3.0,
        b'EK3_SRC1_VELZ': 3.0,
        b'EK3_SRC1_POSZ': 1.0,
        b'EK3_SRC1_YAW': 1.0,
        b'EK3_SRC2_POSXY': 0.0,
        b'EK3_SRC2_VELXY': 0.0,
        b'EK3_SRC2_VELZ': 0.0,
        b'EK3_SRC2_POSZ': 1.0,
        b'EK3_SRC2_YAW': 0.0,
    }
    for param, value in params.items():
        conn.mav.param_set_send(
            conn.target_system, conn.target_component,
            param, float(value), mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )
        time.sleep(0.2)
    print("[MAVLink] ✅ 已還原 GPS 參數，請重啟 SITL 後再執行 record_auto.py", flush=True)


def latlon_to_ned(lat, lon, alt, ref):
    """經緯度 → 相對 ref 原點的 NED 公尺 (x=north, y=east, z=down)"""
    dn = (lat - ref[0]) * 111000.0
    de = (lon - ref[1]) * 111000.0 * math.cos(math.radians(ref[0]))
    dd = -(alt - ref[2])
    return dn, de, dd


# ═══════════════════════════════════════════════
#  4. EV 時間同步 (v22: monotonic + 週期 resync)
# ═══════════════════════════════════════════════
class EVTimeSync:
    """把本機時鐘對齊到飛控 uptime 基準 (EKF3 融合 EV 需要)"""
    RESYNC_PERIOD_S = 30.0

    def __init__(self, conn):
        self.conn = conn
        self.lock = threading.Lock()
        self.ap_boot_us = 0
        self.base_mono_us = int(time.monotonic() * 1e6)
        self.last_resync_mono_us = self.base_mono_us
        self._resync(verbose=True)

    def _resync(self, verbose=False):
        """主動請求 SYSTEM_TIME 並重新對齊"""
        try:
            while self.conn.recv_match(type='SYSTEM_TIME', blocking=False):
                pass
            m = self.conn.recv_match(type='SYSTEM_TIME', blocking=True, timeout=1.0)
        except Exception:
            m = None
        with self.lock:
            if m is not None:
                self.ap_boot_us = int(m.time_boot_ms) * 1000
            self.base_mono_us = int(time.monotonic() * 1e6)
            self.last_resync_mono_us = self.base_mono_us
        if verbose:
            print(f"[EVTimeSync] 飛控 uptime 基準: {self.ap_boot_us} us (monotonic 對齊)", flush=True)

    def now_us(self):
        with self.lock:
            now_mono = int(time.monotonic() * 1e6)
            if (now_mono - self.last_resync_mono_us) <= int(self.RESYNC_PERIOD_S * 1e6):
                return self.ap_boot_us + (now_mono - self.base_mono_us)
        self._resync(verbose=False)
        with self.lock:
            now_mono = int(time.monotonic() * 1e6)
            return self.ap_boot_us + (now_mono - self.base_mono_us)


# ═══════════════════════════════════════════════
#  5. EV 訊息發送 (v22: 協方差非零)
# ═══════════════════════════════════════════════
def send_vision_position(conn, x, y, z, sync=None, yaw_rad=0.0):
    """送 VISION_POSITION_ESTIMATE (#102, local NED)
       z 協方差極大 → EKF 完全忽略 EV 的 z, 高度只從 Baro 拿"""
    usec = sync.now_us() if sync else 0

    cov = [0.0] * 21
    cov[0]  = 0.04     # x 方差 (std 0.2m, 高信任)
    cov[6]  = 0.04     # y 方差
    cov[11] = 100.0    # z 方差極大 (std 10m) → EKF 忽略 z
    cov[15] = 0.01     # roll
    cov[18] = 0.01     # pitch
    cov[20] = 0.01     # yaw

    conn.mav.vision_position_estimate_send(
        usec,
        float(x), float(y), float(z),
        0.0, 0.0, float(yaw_rad),
        cov,
        0
    )


def send_vision_speed(conn, vx, vy, vz, sync=None):
    """送 VISION_SPEED_ESTIMATE (#103, local NED)
       vz 協方差極大 → EKF 忽略 EV 的 vz"""
    usec = sync.now_us() if sync else 0

    cov = [0.0] * 9
    cov[0] = 0.25      # vx 方差
    cov[4] = 0.25      # vy 方差
    cov[8] = 100.0     # vz 方差極大 → EKF 忽略 vz

    conn.mav.vision_speed_estimate_send(
        usec,
        float(vx), float(vy), float(vz),
        cov,
        0
    )


def disable_sim_gps(conn):
    """SITL: 立即關閉模擬 GPS"""
    for name in [b'SIM_GPS_DISABLE', b'SIM_GPS1_DISABLE']:
        conn.mav.param_set_send(conn.target_system, conn.target_component,
                                name, 1, mavutil.mavlink.MAV_PARAM_TYPE_INT8)
        time.sleep(0.2)


def send_global_vision_position(conn, lat, lon, alt, sync=None, yaw_rad=0.0):
    """送 GLOBAL_VISION_POSITION_ESTIMATE (#101, 全球座標) 給 EKF3 ExternalNav"""
    usec = sync.now_us() if sync else 0
    cov = [0.0] * 21
    for i in (0, 6, 11):
        cov[i] = 0.04      # x,y,z
    for i in (15, 18, 20):
        cov[i] = 0.01      # roll,pitch,yaw
    conn.mav.global_vision_position_estimate_send(
        usec,
        lat, lon, alt,
        0.0, 0.0, yaw_rad,
        cov,
        0
    )


# ═══════════════════════════════════════════════
#  6. 影像品質指標 (record / process / repeat 共用)
# ═══════════════════════════════════════════════
def sky_ratio(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float(((h >= 90) & (h <= 140) & (s > 60) & (v > 80)).mean())


def gray_veil_ratio(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    return float(((s < 30) & (v > 90) & (v < 225)).mean())


def contrast_of(frame):
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).std())


def edge_density(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float((cv2.Canny(g, 50, 150) > 0).mean())
