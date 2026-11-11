#!/usr/bin/env python3
"""
ArduDeck Dual-SITL UAV Interception Demo (Taipei 101 / Xinyi Road Scenario)
==========================================================================

Scenario:
  - Target (SYS 2, ArduCopter):
      Starts from Taipei West District (near Ximen / CKS Memorial Hall on Xinyi Rd),
      flies East along Xinyi Road toward Taipei 101. If not intercepted (or after
      an intercept pass), it flies back West to the Tamsui River side and repeats.
  - Interceptor (SYS 1, ArduPlane):
      Takes off from Songshan Airport (RCSS Runway 10), flies South toward
      Xinyi Road / Taipei 101, and intercepts the multirotor using 20 Hz
      `SET_ATTITUDE_TARGET` attitude/rate commands driven by a pluggable
      visual/LOS error function (`get_target_visual_error`).
  - GCS Bridge:
      Automatically forwards all MAVLink traffic from BOTH SITL instances to
      UDP 127.0.0.1:14550 so ArduDeck displays both vehicles (`SYS 1` & `SYS 2`),
      synthetic vision FPV, Taipei 101 3D GLB, and the target drone 3D GLB.

Usage:
  python3 scripts/intercept_demo.py --ardupilot-dir ~/ardupilot
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

try:
    from pymavlink import mavutil
except ImportError:
    print("ERROR: pymavlink is not installed. Run: pip install pymavlink")
    sys.exit(1)


# ============================================================================
# Geographic Waypoints (WGS84)
# ============================================================================
# Songshan Airport Runway 10 (Fixed-Wing Interceptor Launch)
SONGSHAN_HOME = (25.0694, 121.5525, 10.0, 100.0)  # lat, lon, alt_msl, heading_deg

# Xinyi Road Corridor (Multirotor Target Route)
# Starts on Xinyi Rd in West/Central Taipei so it reaches Taipei 101 right as
# the fixed-wing arrives from Songshan Airport (~4.2 km away).
XINYI_START = (25.0338, 121.5225, 10.0, 92.0)     # Xinyi Rd (near CKS Memorial Hall)
TAIPEI_101_WP = (25.0330, 121.5640)               # Xinyi Rd & Taipei 101 (Eastbound target)
WEST_RIVER_WP = (25.0342, 121.5045)               # West District / Tamsui River side (Westbound return)

TARGET_CRUISE_ALT_AGL = 250.0   # m AGL (250m clears all Taipei OSM buildings; ~half height of Taipei 101)
TARGET_SPEED_MPS = 15.0         # m/s (~54 km/h fast multirotor transit)

INTERCEPT_MIN_ALT_AGL = 120.0   # m AGL safety floor for fixed-wing attitude control
INTERCEPT_HIT_RADIUS_M = 18.0   # m closest-approach distance for intercept success

# Persistent integral trim so steady-state altitude offset between Plane & Copter is 0.0m
_alt_integral_deg: float = 0.0
_last_ctrl_wall: float = 0.0


@dataclass
class VehicleState:
    sysid: int
    lat: float = 0.0
    lon: float = 0.0
    alt_msl: float = 0.0
    alt_agl: float = 0.0
    vn: float = 0.0
    ve: float = 0.0
    vd: float = 0.0
    roll: float = 0.0       # rad
    pitch: float = 0.0      # rad
    yaw: float = 0.0        # rad
    roll_rate: float = 0.0  # rad/s
    pitch_rate: float = 0.0 # rad/s
    yaw_rate: float = 0.0   # rad/s
    airspeed: float = 0.0
    groundspeed: float = 0.0
    armed: bool = False
    custom_mode: int = -1
    last_update: float = 0.0

    @property
    def has_fix(self) -> bool:
        return self.lat != 0.0 and self.lon != 0.0


# ============================================================================
# Math & Geodesy Helpers
# ============================================================================
M_PER_DEG_LAT = 111_320.0


def m_per_deg_lon(lat_deg: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat_deg))


def ned_offset_m(lat1: float, lon1: float, alt1: float, lat2: float, lon2: float, alt2: float) -> Tuple[float, float, float]:
    """Return (north_m, east_m, down_m) from point 1 to point 2."""
    dn = (lat2 - lat1) * M_PER_DEG_LAT
    de = (lon2 - lon1) * m_per_deg_lon((lat1 + lat2) * 0.5)
    dd = -(alt2 - alt1)
    return dn, de, dd


def wrap_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> list[float]:
    """Convert Euler angles (rad) to [w, x, y, z] quaternion."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


# ============================================================================
# Pluggable Visual / Line-of-Sight Target Tracker
# ============================================================================
def get_target_visual_error(
    plane: VehicleState,
    copter: VehicleState,
    lead_factor: float = 0.90,
) -> Tuple[float, float, float, float]:
    """
    Compute the angular tracking error (yaw_err_rad, pitch_err_rad, desired_yaw_rad, dist_3d_m)
    of the target relative to the fixed-wing's velocity/nose (virtual FPV camera).

    NOTE FOR YOLO INTEGRATION:
    If you later replace this with a live YOLO bounding-box detector on the ArduDeck
    FPV video stream (width W, height H, vertical FOV = 62 deg):
        yaw_err_rad   = ((bbox_cx - W/2) / (H/2)) * math.radians(31.0)
        pitch_err_rad = ((H/2 - bbox_cy) / (H/2)) * math.radians(31.0)
    """
    dn, de, dd = ned_offset_m(
        plane.lat, plane.lon, plane.alt_agl,
        copter.lat, copter.lon, copter.alt_agl,
    )
    dist_horiz = math.hypot(dn, de)
    dist_3d = math.hypot(dist_horiz, dd)

    # Closing velocity along the 3D Line-of-Sight
    v_rel_n = copter.vn - plane.vn
    v_rel_e = copter.ve - plane.ve
    v_rel_d = copter.vd - plane.vd
    closing_speed = max(18.0, -(dn * v_rel_n + de * v_rel_e + dd * v_rel_d) / max(1.0, dist_3d))
    t_go = clamp(dist_3d / closing_speed, 0.0, 4.5) * lead_factor

    aim_n = dn + copter.vn * t_go
    aim_e = de + copter.ve * t_go
    aim_u = -dd + (-copter.vd) * t_go

    aim_horiz = max(1.0, math.hypot(aim_n, aim_e))
    desired_yaw = math.atan2(aim_e, aim_n)  # 0 = North, +pi/2 = East
    desired_pitch = math.atan2(aim_u, aim_horiz)

    # Use ground track course when moving so wind/sideslip never causes steady-state miss
    current_heading = math.atan2(plane.ve, plane.vn) if plane.groundspeed > 5.0 else plane.yaw
    yaw_err = wrap_pi(desired_yaw - current_heading)
    pitch_err = desired_pitch - plane.pitch
    return yaw_err, pitch_err, desired_yaw, dist_3d


# ============================================================================
# Fixed-Wing Attitude & Rate Controller (`SET_ATTITUDE_TARGET`)
# ============================================================================
def send_plane_attitude_target(
    conn: mavutil.mavfile,
    plane: VehicleState,
    copter: VehicleState,
    boot_ms: int,
) -> Tuple[float, float, float, float]:
    """
    Compute and transmit MAVLink `SET_ATTITUDE_TARGET` to ArduPlane.
    Combines 3D Proportional Navigation (Pro-Nav) with Visual FOV Error & Integral Altitude Trim
    so steady-state vertical/horizontal miss distance converges to < 2m.
    Returns (dist_3d_m, roll_cmd_deg, pitch_cmd_deg, yaw_err_deg).
    """
    global _alt_integral_deg, _last_ctrl_wall
    now = time.time()
    dt = clamp(now - _last_ctrl_wall, 0.01, 0.2) if _last_ctrl_wall > 0.0 else 0.05
    _last_ctrl_wall = now

    yaw_err, _pitch_err, _desired_yaw, dist_3d = get_target_visual_error(plane, copter)

    dn, de, dd = ned_offset_m(
        plane.lat, plane.lon, plane.alt_agl,
        copter.lat, copter.lon, copter.alt_agl,
    )
    dist_horiz_sq = max(25.0, dn * dn + de * de)
    v_rel_n = copter.vn - plane.vn
    v_rel_e = copter.ve - plane.ve
    # Horizontal Line-of-Sight (LOS) rotation rate (rad/s) for Proportional Navigation
    los_rate_yaw = (dn * v_rel_e - de * v_rel_n) / dist_horiz_sq
    closing_speed = max(20.0, plane.groundspeed)
    pronav_accel_lat = 3.6 * closing_speed * los_rate_yaw
    pronav_roll = math.atan2(pronav_accel_lat, 9.81) if abs(yaw_err) < math.radians(45.0) else 0.0

    # 1. Bank angle (Roll) command: Pro-Nav feedforward + visual LOS error + yaw-rate damping
    kp_roll = 2.4 if dist_3d < 400.0 else 1.6
    max_bank_rad = math.radians(45.0)
    roll_cmd = clamp(pronav_roll + kp_roll * yaw_err - 0.12 * plane.yaw_rate, -max_bank_rad, max_bank_rad)

    # 2. Pitch angle command with Integral Trim + Terminal 3D LOS Lock:
    #    Climb immediately to TARGET_CRUISE_ALT_AGL (250m) after takeoff even while the
    #    copter is still ascending, so the fixed-wing clears all Taipei OSM buildings!
    desired_alt_agl = (
        max(copter.alt_agl, TARGET_CRUISE_ALT_AGL)
        if (dist_3d > 500.0 or copter.alt_agl < TARGET_CRUISE_ALT_AGL * 0.85)
        else copter.alt_agl
    )
    alt_err_m = desired_alt_agl - plane.alt_agl
    if abs(alt_err_m) < 35.0 and plane.alt_agl > INTERCEPT_MIN_ALT_AGL:
        _alt_integral_deg = clamp(_alt_integral_deg + alt_err_m * 0.14 * dt, -6.5, 6.5)

    climb_rate_mps = -plane.vd
    target_climb_mps = -copter.vd if desired_alt_agl == copter.alt_agl else 0.0
    # In terminal phase (< 400m), blend directly into 3D LOS vertical pursuit
    los_climb_mps = target_climb_mps + clamp(alt_err_m * 0.45, -10.0, 12.0)
    climb_err_mps = los_climb_mps - climb_rate_mps

    # Lift compensation in a banked turn: 1/cos(roll) - 1
    cos_roll = max(0.5, math.cos(roll_cmd))
    bank_pitch_comp_deg = 5.0 * (1.0 / cos_roll - 1.0)
    level_trim_deg = -2.6 + _alt_integral_deg

    pitch_cmd_deg = (
        level_trim_deg
        + clamp(alt_err_m * 0.38, -12.0, 14.0)
        + clamp(climb_err_mps * 1.25, -8.0, 9.0)
        + bank_pitch_comp_deg
    )
    if plane.alt_agl < INTERCEPT_MIN_ALT_AGL and pitch_cmd_deg < 6.0:
        pitch_cmd_deg = 8.0
    pitch_cmd = math.radians(clamp(pitch_cmd_deg, -18.0, 22.0))

    # 3. Coordinated body angular rates (p, q, r)
    speed = max(16.0, plane.airspeed if plane.airspeed > 5.0 else plane.groundspeed)
    turn_rate_r = (9.81 / speed) * math.tan(roll_cmd)
    roll_rate_cmd = clamp((roll_cmd - plane.roll) * 2.8, -1.8, 1.8)
    pitch_rate_cmd = clamp((pitch_cmd - plane.pitch) * 2.2, -1.2, 1.2)

    # 4. Throttle command: controlled pursuit speed so the wing doesn't balloon in terminal phase
    throttle_cmd = 0.82 if dist_3d > 250.0 else 0.68

    # CRITICAL ARDUPLANE QUIRK (see GCS_MAVLink_Plane.cpp:1063 & Attitude.cpp:534):
    # ArduPlane interprets `q.get_euler_yaw()` in SET_ATTITUDE_TARGET as direct
    # rudder servo deflection (`commanded_rudder`), NOT compass heading!
    # Therefore `yaw` in the quaternion MUST be 0.0 so the rudder stays coordinated!
    q = euler_to_quaternion(roll_cmd, pitch_cmd, 0.0)

    conn.mav.set_attitude_target_send(
        boot_ms,
        plane.sysid,
        1,
        0,  # type_mask
        q,
        roll_rate_cmd,
        pitch_rate_cmd,
        turn_rate_r,
        throttle_cmd,
    )

    return (
        dist_3d,
        math.degrees(roll_cmd),
        math.degrees(pitch_cmd),
        math.degrees(yaw_err),
    )


# ============================================================================
# MAVLink Helpers
# ============================================================================
def request_fast_streams(conn: mavutil.mavfile, sysid: int, rate_hz: int = 20) -> None:
    """Request high-rate position and attitude telemetry from SITL."""
    conn.mav.request_data_stream_send(
        sysid,
        1,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        rate_hz,
        1,
    )
    interval_us = int(1_000_000 / rate_hz)
    for msg_id in (
        mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
        mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD,
        mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,
    ):
        conn.mav.command_long_send(
            sysid,
            1,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            msg_id,
            interval_us,
            0, 0, 0, 0, 0,
        )


def set_param(conn: mavutil.mavfile, sysid: int, name: str, value: float, ptype: int = mavutil.mavlink.MAV_PARAM_TYPE_REAL32) -> None:
    conn.mav.param_set_send(
        sysid,
        1,
        name.encode("ascii"),
        float(value),
        ptype,
    )


def set_mode_ardupilot(conn: mavutil.mavfile, sysid: int, custom_mode_num: int) -> None:
    conn.mav.set_mode_send(
        sysid,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        custom_mode_num,
    )


def arm_vehicle(conn: mavutil.mavfile, sysid: int) -> None:
    conn.mav.command_long_send(
        sysid,
        1,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,     # arm
        21196, # force-arm magic in SITL
        0, 0, 0, 0, 0,
    )


def send_copter_goto(conn: mavutil.mavfile, sysid: int, lat: float, lon: float, alt_agl: float) -> None:
    """Command ArduCopter in GUIDED mode to fly to (lat, lon, alt_agl)."""
    type_mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )
    conn.mav.set_position_target_global_int_send(
        0,
        sysid,
        1,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        type_mask,
        int(lat * 1e7),
        int(lon * 1e7),
        float(alt_agl),
        0, 0, 0,
        0, 0, 0,
        0, 0,
    )


def update_vehicle_state(msg, state: VehicleState) -> None:
    mtype = msg.get_type()
    now = time.time()
    if mtype == "HEARTBEAT":
        if msg.type in (
            mavutil.mavlink.MAV_TYPE_FIXED_WING,
            mavutil.mavlink.MAV_TYPE_QUADROTOR,
            mavutil.mavlink.MAV_TYPE_HEXAROTOR,
            mavutil.mavlink.MAV_TYPE_OCTOROTOR,
        ):
            state.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            state.custom_mode = int(msg.custom_mode)
            state.last_update = now
    elif mtype == "GLOBAL_POSITION_INT":
        state.lat = msg.lat / 1e7
        state.lon = msg.lon / 1e7
        state.alt_msl = msg.alt / 1000.0
        state.alt_agl = msg.relative_alt / 1000.0
        state.vn = msg.vx / 100.0
        state.ve = msg.vy / 100.0
        state.vd = msg.vz / 100.0
        state.last_update = now
    elif mtype == "ATTITUDE":
        state.roll = msg.roll
        state.pitch = msg.pitch
        state.yaw = msg.yaw
        state.roll_rate = msg.rollspeed
        state.pitch_rate = msg.pitchspeed
        state.yaw_rate = msg.yawspeed
        state.last_update = now
    elif mtype == "VFR_HUD":
        state.airspeed = msg.airspeed
        state.groundspeed = msg.groundspeed
        state.last_update = now
    elif mtype == "STATUSTEXT":
        text = getattr(msg, "text", "")
        if text:
            print(f"  [SYS {state.sysid} STATUS] {text}")


# ============================================================================
# Direct SITL Binary Launcher (bypasses sim_vehicle.py subshell daemonizing)
# ============================================================================
def find_ardupilot_root(user_dir: Optional[str]) -> Path:
    candidates = []
    if user_dir:
        p = Path(user_dir).expanduser().resolve()
        candidates.extend([p, p.parent.parent])
    which_sv = shutil.which("sim_vehicle.py")
    if which_sv:
        candidates.append(Path(which_sv).resolve().parent.parent.parent)
    candidates.extend([
        Path.home() / "ardupilot",
        Path("/Users/phoenix/ardupilot"),
    ])
    for c in candidates:
        if (c / "Tools" / "autotest" / "sim_vehicle.py").is_file():
            return c
    raise FileNotFoundError(
        "Could not locate ArduPilot directory. Pass --ardupilot-dir ~/ardupilot"
    )


def ensure_sitl_binary(ardupilot_root: Path, vehicle: str) -> Path:
    bin_name = "arduplane" if vehicle == "ArduPlane" else "arducopter"
    target = "bin/arduplane" if vehicle == "ArduPlane" else "bin/arducopter"
    bin_path = ardupilot_root / "build" / "sitl" / "bin" / bin_name
    if bin_path.is_file():
        return bin_path

    waf = ardupilot_root / "modules" / "waf" / "waf-light"
    print(f"[BUILD] Building {vehicle} SITL binary ({target})...")
    subprocess.run([sys.executable, str(waf), "configure", "--board", "sitl"], cwd=str(ardupilot_root), check=True)
    subprocess.run([sys.executable, str(waf), "build", "--target", target], cwd=str(ardupilot_root), check=True)
    return bin_path


def launch_sitl_binary(
    ardupilot_root: Path,
    vehicle: str,
    instance: int,
    sysid: int,
    home: Tuple[float, float, float, float],
    work_dir: Path,
) -> Tuple[subprocess.Popen, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    bin_path = ensure_sitl_binary(ardupilot_root, vehicle)
    lat, lon, alt, hdg = home
    loc_str = f"{lat:.6f},{lon:.6f},{alt:.1f},{hdg:.1f}"
    model = "plane" if vehicle == "ArduPlane" else "+"

    cmd = [
        str(bin_path),
        "-w",
        "--model", model,
        "--speedup", "1",
        "--sysid", str(sysid),
        f"-I{instance}",
        "--home", loc_str,
    ]
    log_path = work_dir / f"sitl_{vehicle}_{instance}.log"
    print(f"[LAUNCH] Starting {vehicle} (Instance {instance}, SYSID {sysid}) at {loc_str}")
    log_file = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        cwd=str(work_dir),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc, log_path


def connect_tcp_with_retry(
    port: int,
    label: str,
    proc: Optional[subprocess.Popen] = None,
    log_path: Optional[Path] = None,
    timeout_s: float = 60.0,
) -> mavutil.mavfile:
    deadline = time.time() + timeout_s
    print(f"[CONNECT] Waiting for {label} SITL on tcp:127.0.0.1:{port} ...")
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = ""
            if log_path and log_path.is_file():
                lines = log_path.read_text(errors="replace").splitlines()
                tail = "\n".join(lines[-20:])
            raise RuntimeError(
                f"{label} process exited early (code {proc.returncode}). Log tail:\n{tail}"
            )
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=1.0)
            sock.close()
            conn = mavutil.mavlink_connection(f"tcp:127.0.0.1:{port}", source_system=254)
            conn.wait_heartbeat(timeout=10)
            print(f"[CONNECT] {label} heartbeat received on port {port}!")
            return conn
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {label} on tcp:127.0.0.1:{port}")


def send_gcs_statustext(
    gcs_sock: socket.socket,
    gcs_addr: Tuple[str, int],
    sysid: int,
    text: str,
    severity: int = mavutil.mavlink.MAV_SEVERITY_ALERT,
) -> None:
    """Encode a MAVLink STATUSTEXT message from `sysid` and send directly to ArduDeck UDP port."""
    try:
        mav = mavutil.mavlink.MAVLink(None, srcSystem=sysid, srcComponent=1)
        msg = mav.statustext_encode(severity, text.encode("utf-8")[:50])
        pkt = msg.pack(mav)
        gcs_sock.sendto(bytes(pkt), gcs_addr)
    except Exception:
        pass


# ============================================================================
# Main Orchestration & Interception Loop
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="ArduDeck Dual-SITL Interception Demo")
    parser.add_argument("--ardupilot-dir", type=str, default=os.environ.get("ARDUPILOT_DIR"),
                        help="Path to ArduPilot repository root (e.g. ~/ardupilot)")
    parser.add_argument("--gcs-port", type=int, default=14550,
                        help="UDP port where ArduDeck listens (default: 14550)")
    parser.add_argument("--hit-distance", type=float, default=INTERCEPT_HIT_RADIUS_M,
                        help=f"3D distance threshold (m) for intercept success (default: {INTERCEPT_HIT_RADIUS_M}m)")
    parser.add_argument("--continuous", action="store_true",
                        help="Keep flying repeated intercept passes instead of landing target on hit")
    parser.add_argument("--no-launch", action="store_true",
                        help="Connect to already-running SITL instances on TCP 5760 & 5770")
    args = parser.parse_args()

    procs: list[subprocess.Popen] = []
    run_root = Path("/tmp/ardudeck_intercept_sitl")
    run_root.mkdir(parents=True, exist_ok=True)
    telem_log = open(run_root / "telemetry.log", "w", buffering=1)

    def cleanup(*_sig) -> None:
        print("\n[SHUTDOWN] Stopping SITL processes...")
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        subprocess.run(["killall", "-9", "arduplane", "arducopter"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    if not args.no_launch:
        # Kill any lingering SITL binaries spawned by previous runs
        subprocess.run(["killall", "-9", "arduplane", "arducopter"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)

        ardupilot_root = find_ardupilot_root(args.ardupilot_dir)
        plane_proc, plane_log = launch_sitl_binary(
            ardupilot_root, "ArduPlane", instance=0, sysid=1,
            home=SONGSHAN_HOME, work_dir=run_root / "plane_0",
        )
        procs.append(plane_proc)
        plane_conn = connect_tcp_with_retry(5760, "ArduPlane (SYS 1)", plane_proc, plane_log)

        copter_proc, copter_log = launch_sitl_binary(
            ardupilot_root, "ArduCopter", instance=1, sysid=2,
            home=XINYI_START, work_dir=run_root / "copter_1",
        )
        procs.append(copter_proc)
        copter_conn = connect_tcp_with_retry(5770, "ArduCopter (SYS 2)", copter_proc, copter_log)
    else:
        plane_conn = connect_tcp_with_retry(5760, "ArduPlane (SYS 1)")
        copter_conn = connect_tcp_with_retry(5770, "ArduCopter (SYS 2)")

    # Single UDP bridge socket to ArduDeck GCS (127.0.0.1:14550)
    gcs_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    gcs_sock.setblocking(False)
    gcs_addr = ("127.0.0.1", args.gcs_port)
    print(f"[BRIDGE] Forwarding both SYS 1 (Plane) & SYS 2 (Copter) to ArduDeck at udp://{gcs_addr[0]}:{gcs_addr[1]}")

    # Configure SITL parameters & high-rate telemetry streams
    set_param(plane_conn, 1, "SYSID_THISMAV", 1)
    set_param(plane_conn, 1, "AIRSPEED_CRUISE", 28)
    set_param(plane_conn, 1, "TRIM_ARSPD_CM", 2800)
    set_param(plane_conn, 1, "TKOFF_ALT", 80)

    set_param(copter_conn, 2, "SYSID_THISMAV", 2)
    set_param(copter_conn, 2, "WPNAV_SPEED", TARGET_SPEED_MPS * 100.0)
    set_param(copter_conn, 2, "WPNAV_SPEED_UP", 800.0)
    set_param(copter_conn, 2, "PILOT_SPEED_UP", 800.0)
    set_param(copter_conn, 2, "WP_SPD_UP", 8.0)
    set_param(copter_conn, 2, "PILOT_S_UP", 8.0)

    request_fast_streams(plane_conn, 1, rate_hz=25)
    request_fast_streams(copter_conn, 2, rate_hz=25)

    plane_state = VehicleState(sysid=1)
    copter_state = VehicleState(sysid=2)

    start_wall = time.time()
    copter_stage = "INIT"      # INIT -> TAKEOFF -> TO_101 -> TO_WEST -> INTERCEPTED
    plane_stage = "INIT"       # INIT -> TAKEOFF -> INTERCEPT -> COMPLETE
    last_control_t = 0.0
    last_log_t = 0.0
    last_goto_t = 0.0
    last_plane_arm_t = 0.0
    last_copter_arm_t = 0.0
    last_copter_tkoff_t = 0.0
    intercept_count = 0
    in_hit_zone = False
    cpa_min_m = 9999.0
    best_overall_m = 9999.0

    print("\n======================================================================")
    print("  ArduDeck Dual-SITL Interception Running!")
    print("  1. In ArduDeck, connect via UDP (Port 14550).")
    print("  2. Select SYS 1 (Plane) in the top Fleet bar.")
    print("  3. Open Vision -> Synthetic and enable the '3D GLB' button.")
    print("======================================================================\n")

    while True:
        now = time.time()
        boot_ms = int((now - start_wall) * 1000) & 0xFFFFFFFF

        # 1. Drain messages from both SITLs, update states, and forward raw bytes to ArduDeck
        for conn, state in ((plane_conn, plane_state), (copter_conn, copter_state)):
            while True:
                msg = conn.recv_match(blocking=False)
                if msg is None:
                    break
                buf = msg.get_msgbuf()
                if buf:
                    try:
                        gcs_sock.sendto(bytes(buf), gcs_addr)
                    except OSError:
                        pass
                update_vehicle_state(msg, state)

        # Forward any GCS packets from ArduDeck back to both SITLs
        while True:
            try:
                data, _ = gcs_sock.recvfrom(4096)
                plane_conn.write(data)
                copter_conn.write(data)
            except BlockingIOError:
                break
            except OSError:
                break

        # 2. Target Multirotor (SYS 2) State Machine: Xinyi Rd <-> Taipei 101 <-> West River
        if copter_state.has_fix:
            if copter_stage == "INIT":
                if copter_state.armed:
                    print(f"[SYS 2 COPTER] Armed! Ascending to {TARGET_CRUISE_ALT_AGL}m AGL...")
                    copter_stage = "TAKEOFF"
                elif now - last_copter_arm_t >= 1.5:
                    last_copter_arm_t = now
                    set_mode_ardupilot(copter_conn, 2, 4)  # GUIDED = 4
                    arm_vehicle(copter_conn, 2)

            elif copter_stage == "TAKEOFF":
                if not copter_state.armed:
                    copter_stage = "INIT"
                elif copter_state.alt_agl >= TARGET_CRUISE_ALT_AGL - 3.0:
                    print(f"[SYS 2 COPTER] Reached {TARGET_CRUISE_ALT_AGL:.0f}m cruise altitude! Flying East along Xinyi Rd -> Taipei 101")
                    send_copter_goto(copter_conn, 2, TAIPEI_101_WP[0], TAIPEI_101_WP[1], TARGET_CRUISE_ALT_AGL)
                    last_goto_t = now
                    copter_stage = "TO_101"
                elif copter_state.alt_agl < 3.0 and (now - last_copter_tkoff_t) >= 1.5:
                    last_copter_tkoff_t = now
                    set_param(copter_conn, 2, "WPNAV_SPEED_UP", 800.0)
                    set_param(copter_conn, 2, "PILOT_SPEED_UP", 800.0)
                    set_param(copter_conn, 2, "WP_SPD_UP", 8.0)
                    set_param(copter_conn, 2, "PILOT_S_UP", 8.0)
                    set_mode_ardupilot(copter_conn, 2, 4)
                    copter_conn.mav.command_long_send(
                        2, 1,
                        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                        0, 0, 0, 0, 0, 0, 0,
                        TARGET_CRUISE_ALT_AGL,
                    )

            elif copter_stage == "TO_101":
                dn, de, _ = ned_offset_m(
                    copter_state.lat, copter_state.lon, 0,
                    TAIPEI_101_WP[0], TAIPEI_101_WP[1], 0,
                )
                if math.hypot(dn, de) < 45.0:
                    print("[SYS 2 COPTER] Reached Taipei 101 (unintercepted)! Turning West toward Tamsui River...")
                    send_copter_goto(copter_conn, 2, WEST_RIVER_WP[0], WEST_RIVER_WP[1], TARGET_CRUISE_ALT_AGL)
                    last_goto_t = now
                    copter_stage = "TO_WEST"
                elif now - last_goto_t > 3.0:
                    send_copter_goto(copter_conn, 2, TAIPEI_101_WP[0], TAIPEI_101_WP[1], TARGET_CRUISE_ALT_AGL)
                    last_goto_t = now

            elif copter_stage == "TO_WEST":
                dn, de, _ = ned_offset_m(
                    copter_state.lat, copter_state.lon, 0,
                    WEST_RIVER_WP[0], WEST_RIVER_WP[1], 0,
                )
                if math.hypot(dn, de) < 45.0:
                    print("[SYS 2 COPTER] Reached West District River! Heading East along Xinyi Rd -> Taipei 101")
                    send_copter_goto(copter_conn, 2, TAIPEI_101_WP[0], TAIPEI_101_WP[1], TARGET_CRUISE_ALT_AGL)
                    last_goto_t = now
                    copter_stage = "TO_101"
                elif now - last_goto_t > 3.0:
                    send_copter_goto(copter_conn, 2, WEST_RIVER_WP[0], WEST_RIVER_WP[1], TARGET_CRUISE_ALT_AGL)
                    last_goto_t = now

            elif copter_stage == "INTERCEPTED":
                if copter_state.custom_mode != 9 and (now - last_goto_t) > 1.5:
                    last_goto_t = now
                    set_mode_ardupilot(copter_conn, 2, 9)  # LAND = 9

        # 3. Fixed-Wing Interceptor (SYS 1) State Machine: Songshan Takeoff -> Attitude-Rate Homing
        if plane_state.has_fix and copter_state.has_fix:
            if plane_stage == "INIT":
                if plane_state.armed:
                    print("[SYS 1 PLANE] Armed at Songshan Airport! Rolling for takeoff...")
                    plane_stage = "TAKEOFF"
                elif now - last_plane_arm_t >= 1.5:
                    last_plane_arm_t = now
                    set_mode_ardupilot(plane_conn, 1, 13)  # TAKEOFF = 13
                    arm_vehicle(plane_conn, 1)

            elif plane_stage == "TAKEOFF":
                if not plane_state.armed:
                    plane_stage = "INIT"
                elif plane_state.alt_agl >= 30.0:
                    set_mode_ardupilot(plane_conn, 1, 15)  # GUIDED = 15
                    print("[SYS 1 PLANE] Airborne (>30m)! Engaging GUIDED + SET_ATTITUDE_TARGET interception!")
                    plane_stage = "INTERCEPT"
                elif now - last_log_t >= 1.5:
                    last_log_t = now
                    msg_line = f"[SYS 1 PLANE] Takeoff climb: {plane_state.alt_agl:.1f}m / 30.0m AGL (Speed: {plane_state.groundspeed:.1f} m/s)"
                    print(msg_line)
                    telem_log.write(msg_line + "\n")

            elif plane_stage == "INTERCEPT" and (now - last_control_t) >= 0.05:
                last_control_t = now
                if plane_state.custom_mode != 15:
                    set_mode_ardupilot(plane_conn, 1, 15)
                dist_3d, roll_deg, pitch_deg, yaw_err_deg = send_plane_attitude_target(
                    plane_conn, plane_state, copter_state, boot_ms
                )
                best_overall_m = min(best_overall_m, dist_3d)

                # Track Closest Point of Approach (CPA) inside hit radius
                if dist_3d <= args.hit_distance:
                    cpa_min_m = min(cpa_min_m, dist_3d)
                    # Trigger immediately on direct impact (<= 8m) or right at CPA minimum inside hit radius
                    if not in_hit_zone and (dist_3d <= 8.0 or dist_3d > cpa_min_m + 0.3):
                        in_hit_zone = True
                        intercept_count += 1
                        banner = f"INTERCEPT #{intercept_count} SUCCESS! Closest dist: {cpa_min_m:.1f}m"
                        print("\n" + "=" * 72)
                        print(f"  🎯🎯🎯 {banner} 🎯🎯🎯")
                        if not args.continuous:
                            print("  -> Target Multirotor (SYS 2) neutralized! Entering LAND mode.")
                            print("  -> Fixed-Wing Interceptor (SYS 1) entering victory LOITER.")
                        print("=" * 72 + "\n")
                        telem_log.write(f"HIT: {banner}\n")

                        # Send MAVLink STATUSTEXT directly to ArduDeck UDP port so GCS HUD displays it!
                        send_gcs_statustext(gcs_sock, gcs_addr, 1, banner)
                        send_gcs_statustext(gcs_sock, gcs_addr, 2, f"SYS 2 INTERCEPTED AT {cpa_min_m:.1f}M - LANDING")

                        if not args.continuous:
                            copter_stage = "INTERCEPTED"
                            plane_stage = "COMPLETE"
                            set_mode_ardupilot(copter_conn, 2, 9)   # Copter LAND = 9
                            set_mode_ardupilot(plane_conn, 1, 12)   # Plane LOITER = 12
                elif dist_3d > args.hit_distance * 2.0:
                    in_hit_zone = False
                    cpa_min_m = 9999.0

                if plane_stage == "INTERCEPT" and (now - last_log_t) >= 1.0:
                    last_log_t = now
                    phase_tag = "TERMINAL_FOV_LOCK" if abs(yaw_err_deg) < 25.0 and dist_3d < 400.0 else "PURSUIT"
                    line = (
                        f"[{phase_tag:17s}] Dist: {dist_3d:6.1f}m (Best: {best_overall_m:5.1f}m) | "
                        f"FOV YawErr: {yaw_err_deg:+6.1f}° | "
                        f"Cmd(Roll={roll_deg:+5.1f}°, Pitch={pitch_deg:+5.1f}°) | "
                        f"Alt(Plane={plane_state.alt_agl:5.1f}m, Copter={copter_state.alt_agl:5.1f}m)"
                    )
                    print(line)
                    telem_log.write(line + "\n")

            elif plane_stage == "COMPLETE" and (now - last_log_t) >= 2.5:
                last_log_t = now
                if plane_state.custom_mode != 12:
                    set_mode_ardupilot(plane_conn, 1, 12)  # LOITER = 12
                line = (
                    f"[INTERCEPT SUCCESS] Target SYS 2 Neutralized (CPA: {cpa_min_m:.1f}m) | "
                    f"Copter Alt: {copter_state.alt_agl:5.1f}m (LANDING) | "
                    f"Plane Alt: {plane_state.alt_agl:5.1f}m (LOITER)"
                )
                print(line)
                telem_log.write(line + "\n")

        time.sleep(0.005)


if __name__ == "__main__":
    main()
