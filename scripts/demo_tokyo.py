#!/usr/bin/env python3
"""
ArduDeck Fixed-Wing Tokyo Sightseeing Demo (ArduPlane SITL)
===========================================================

Launches an ArduPlane SITL instance at Haneda Airport (RJTT), bridges MAVLink
telemetry to ArduDeck (local UDP 14550 + optional LAN TCP/UDP for mobile/iPad),
uploads a scenic Tokyo landmark mission, takes off automatically, and flies
past major Tokyo landmarks in 3D view.

Usage:
  python3 scripts/demo_tokyo.py
  python3 scripts/demo_tokyo.py --ardupilot-dir ~/ardupilot --speedup 1
  python3 scripts/demo_tokyo.py --route-only   # Print route distance without starting SITL
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
from typing import List, Optional, Tuple

try:
    from pymavlink import mavutil
except ImportError:
    print("ERROR: pymavlink is not installed. Run: pip install pymavlink")
    sys.exit(1)


# ============================================================================
# Tokyo Route Definition (WGS84)
# ============================================================================
# Tokyo Bay Open Harbor / Sea Surface (south of Rainbow Bridge — 100% open water, zero building clipping!)
TOKYO_TAKEOFF_HOME = (35.6195, 139.7605, 1.0, 355.0)  # lat, lon, alt_msl, heading_deg (North toward Rainbow Bridge)

# (Name, lat, lon, alt_agl_m)
TOKYO_LANDMARKS: List[Tuple[str, float, float, float]] = [
    ("Odaiba / Rainbow Bridge (台場・彩虹大橋)", 35.6366, 139.7632, 300.0),
    ("Tokyo Tower (東京鐵塔)", 35.6586, 139.7454, 420.0),
    ("Shinjuku Metropolitan Bldg (新宿都廳)", 35.6896, 139.6917, 420.0),
    ("Tokyo Imperial Palace / Station (皇居・東京車站)", 35.6812, 139.7671, 420.0),
    ("Tokyo Skytree (東京晴空塔)", 35.7101, 139.8107, 700.0),
]

CRUISE_AIRSPEED_MPS = 28.0  # ~100.8 km/h
LOITER_RADIUS_M = 250.0     # Optimal stable bank angle (~17.7 deg at 28 m/s) for ArduPlane
ORBIT_TURNS_PER_LANDMARK = 2
EARTH_RADIUS_M = 6371000.0


# ============================================================================
# Geometry & Distance Helpers
# ============================================================================
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def compute_route_legs() -> Tuple[List[Tuple[str, float]], float, float]:
    """Return (legs, transit_m, orbit_m) in meters."""
    legs: List[Tuple[str, float]] = []
    prev_name = "Tokyo Bay Open Sea (東京灣海面起飛)"
    prev_lat, prev_lon = TOKYO_TAKEOFF_HOME[0], TOKYO_TAKEOFF_HOME[1]
    transit_m = 0.0

    for name, lat, lon, _alt in TOKYO_LANDMARKS:
        dist_m = haversine_m(prev_lat, prev_lon, lat, lon)
        legs.append((f"{prev_name} -> {name}", dist_m))
        transit_m += dist_m
        prev_name, prev_lat, prev_lon = name, lat, lon

    # Each landmark gets ORBIT_TURNS_PER_LANDMARK full orbits of radius LOITER_RADIUS_M
    orbit_per_landmark_m = ORBIT_TURNS_PER_LANDMARK * 2.0 * math.pi * LOITER_RADIUS_M
    orbit_m = orbit_per_landmark_m * len(TOKYO_LANDMARKS)
    return legs, transit_m, orbit_m


def print_route_summary() -> float:
    legs, transit_m, orbit_m = compute_route_legs()
    total_m = transit_m + orbit_m
    total_km = total_m / 1000.0
    total_nm = total_km / 1.852
    est_min = (total_m / CRUISE_AIRSPEED_MPS) / 60.0
    orbit_each_km = (ORBIT_TURNS_PER_LANDMARK * 2.0 * math.pi * LOITER_RADIUS_M) / 1000.0

    print("\n" + "=" * 72)
    print("  Tokyo Fixed-Wing Virtual Flight Route Summary")
    print("=" * 72)
    for idx, (label, dist_m) in enumerate(legs, start=1):
        print(f"  Leg {idx}: {label}")
        print(f"         Transit: {dist_m / 1000.0:.2f} km + Orbit ({ORBIT_TURNS_PER_LANDMARK} turns @ R={LOITER_RADIUS_M:.0f}m): {orbit_each_km:.2f} km")
    print("-" * 72)
    print(f"  Transit Distance     : {transit_m / 1000.0:.2f} km")
    print(f"  Orbit Distance ({len(TOKYO_LANDMARKS)}x{ORBIT_TURNS_PER_LANDMARK}) : {orbit_m / 1000.0:.2f} km (R = {LOITER_RADIUS_M:.0f} m, bank ~17.7 deg)")
    print(f"  Total Route Distance : {total_km:.2f} km ({total_nm:.2f} NM)")
    print(f"  Cruise Airspeed      : {CRUISE_AIRSPEED_MPS:.1f} m/s (~{CRUISE_AIRSPEED_MPS * 3.6:.0f} km/h)")
    print(f"  Est. Flight Time     : ~{est_min:.1f} min (at 1x speedup)")
    print("=" * 72 + "\n")
    return total_km


# ============================================================================
# ArduPilot SITL Launcher & MAVLink Helpers
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


def ensure_sitl_binary(ardupilot_root: Path) -> Path:
    bin_path = ardupilot_root / "build" / "sitl" / "bin" / "arduplane"
    if bin_path.is_file():
        return bin_path
    waf = ardupilot_root / "modules" / "waf" / "waf-light"
    print("[BUILD] Building ArduPlane SITL binary (bin/arduplane)...")
    subprocess.run([sys.executable, str(waf), "configure", "--board", "sitl"], cwd=str(ardupilot_root), check=True)
    subprocess.run([sys.executable, str(waf), "build", "--target", "bin/arduplane"], cwd=str(ardupilot_root), check=True)
    return bin_path


def launch_plane_sitl(
    ardupilot_root: Path,
    home: Tuple[float, float, float, float],
    work_dir: Path,
    speedup: int = 1,
) -> Tuple[subprocess.Popen, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    bin_path = ensure_sitl_binary(ardupilot_root)
    lat, lon, alt, hdg = home
    loc_str = f"{lat:.6f},{lon:.6f},{alt:.1f},{hdg:.1f}"

    cmd = [
        str(bin_path),
        "-w",
        "--model", "plane",
        "--speedup", str(speedup),
        "--sysid", "1",
        "-I0",
        "--home", loc_str,
    ]
    log_path = work_dir / "sitl_tokyo_plane.log"
    print(f"[LAUNCH] Starting ArduPlane SITL at Haneda ({loc_str}), speedup={speedup}x")
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
    proc: Optional[subprocess.Popen] = None,
    log_path: Optional[Path] = None,
    timeout_s: float = 60.0,
) -> mavutil.mavfile:
    deadline = time.time() + timeout_s
    print(f"[CONNECT] Waiting for ArduPlane SITL on tcp:127.0.0.1:{port} ...")
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = ""
            if log_path and log_path.is_file():
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
            raise RuntimeError(f"ArduPlane SITL exited early (code {proc.returncode}):\n{tail}")
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=1.0)
            sock.close()
            conn = mavutil.mavlink_connection(f"tcp:127.0.0.1:{port}", source_system=254)
            conn.wait_heartbeat(timeout=10)
            print(f"[CONNECT] Heartbeat received on port {port}!")
            return conn
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for SITL on tcp:127.0.0.1:{port}")


def get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
        finally:
            s.close()
    except Exception:
        pass
    return "127.0.0.1"


class GcsBridge:
    """Bridges raw MAVLink bytes to local ArduDeck (UDP) and remote/mobile GCS (TCP/UDP)."""

    def __init__(self, local_udp_port: int, remote_tcp_port: int) -> None:
        self.lan_ip = get_lan_ip()
        self.local_target = ("127.0.0.1", local_udp_port)
        self.local_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.local_sock.setblocking(False)

        self.tcp_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_srv.bind(("0.0.0.0", remote_tcp_port))
        self.tcp_srv.listen(4)
        self.tcp_srv.setblocking(False)
        self.tcp_clients: List[socket.socket] = []

    def broadcast(self, raw_bytes: bytes) -> None:
        try:
            self.local_sock.sendto(raw_bytes, self.local_target)
        except Exception:
            pass
        self._accept_tcp()
        dead: List[socket.socket] = []
        for cli in self.tcp_clients:
            try:
                cli.sendall(raw_bytes)
            except Exception:
                dead.append(cli)
        for cli in dead:
            self.tcp_clients.remove(cli)
            try:
                cli.close()
            except Exception:
                pass

    def poll_incoming(self) -> List[bytes]:
        packets: List[bytes] = []
        while True:
            try:
                data, _ = self.local_sock.recvfrom(4096)
                if data:
                    packets.append(data)
            except BlockingIOError:
                break
            except Exception:
                break
        self._accept_tcp()
        for cli in list(self.tcp_clients):
            try:
                data = cli.recv(4096)
                if data:
                    packets.append(data)
            except BlockingIOError:
                pass
            except Exception:
                self.tcp_clients.remove(cli)
                try:
                    cli.close()
                except Exception:
                    pass
        return packets

    def _accept_tcp(self) -> None:
        while True:
            try:
                cli, addr = self.tcp_srv.accept()
                cli.setblocking(False)
                self.tcp_clients.append(cli)
                print(f"[BRIDGE] Remote GCS connected via TCP: {addr[0]}:{addr[1]}")
            except BlockingIOError:
                break
            except Exception:
                break


@dataclass
class PlaneState:
    lat: float = 0.0
    lon: float = 0.0
    alt_agl: float = 0.0
    groundspeed: float = 0.0
    armed: bool = False
    custom_mode: int = -1
    has_fix: bool = False
    mission_seq: int = 0


def set_param(conn: mavutil.mavfile, name: str, value: float) -> None:
    conn.mav.param_set_send(
        1, 1,
        name.encode("ascii"),
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )


def set_mode_ardupilot(conn: mavutil.mavfile, custom_mode: int) -> None:
    conn.mav.set_mode_send(
        1,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        custom_mode,
    )


def arm_plane(conn: mavutil.mavfile) -> None:
    conn.mav.command_long_send(
        1, 1,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0,
    )


def request_streams(conn: mavutil.mavfile, rate_hz: int = 20) -> None:
    for stream_id in (
        mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS,
        mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA3,
    ):
        conn.mav.request_data_stream_send(1, 1, stream_id, rate_hz, 1)


def upload_tokyo_mission(conn: mavutil.mavfile) -> int:
    """Upload Home + Tokyo landmarks (2-turn orbit each) + final Loiter to ArduPlane."""
    # Sequence:
    #   0: Home (Tokyo Bay sea surface)
    #   1..N: Tokyo Landmarks (MAV_CMD_NAV_LOITER_TURNS: 2 full turns at LOITER_RADIUS_M)
    #   N+1: Loiter Unlimited at Tokyo Skytree (NAV_LOITER_UNLIM)
    items = [
        (
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            TOKYO_TAKEOFF_HOME[0], TOKYO_TAKEOFF_HOME[1], 0.0,
        )
    ]
    for _name, lat, lon, alt_agl in TOKYO_LANDMARKS:
        items.append((mavutil.mavlink.MAV_CMD_NAV_LOITER_TURNS, lat, lon, alt_agl))

    last_name, last_lat, last_lon, last_alt = TOKYO_LANDMARKS[-1]
    items.append((mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM, last_lat, last_lon, last_alt))

    total = len(items)
    print(f"[MISSION] Uploading {total} waypoints (Tokyo Bay -> {last_name}, {ORBIT_TURNS_PER_LANDMARK} orbits/landmark)...")
    conn.mav.mission_clear_all_send(1, 1)
    # Consume the MISSION_ACK from mission_clear_all_send before starting upload
    clear_deadline = time.time() + 2.0
    while time.time() < clear_deadline:
        ack = conn.recv_match(type="MISSION_ACK", blocking=True, timeout=0.3)
        if ack is not None:
            break

    conn.mav.mission_count_send(1, 1, total)

    deadline = time.time() + 20.0
    sent = set()
    while time.time() < deadline:
        msg = conn.recv_match(type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"], blocking=True, timeout=1.0)
        if msg is None:
            continue
        mtype = msg.get_type()
        if mtype in ("MISSION_REQUEST_INT", "MISSION_REQUEST"):
            seq = msg.seq
            cmd, lat, lon, alt = items[seq]
            if cmd == mavutil.mavlink.MAV_CMD_NAV_LOITER_TURNS:
                param1 = float(ORBIT_TURNS_PER_LANDMARK)  # 2 full turns
                param2 = 0.0                              # heading required (0 = immediate exit after turns)
                param3 = float(LOITER_RADIUS_M)           # clockwise radius (m)
                param4 = 1.0                              # exit tangent toward next WP
            elif cmd == mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM:
                param1, param2, param3, param4 = 0.0, 0.0, float(LOITER_RADIUS_M), 0.0
            else:
                param1, param2, param3, param4 = 0.0, 120.0, 0.0, 0.0
            conn.mav.mission_item_int_send(
                1, 1,
                seq,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                cmd,
                1 if seq == 1 else 0,
                1,  # autocontinue
                param1, param2, param3, param4,
                int(lat * 1e7),
                int(lon * 1e7),
                float(alt),
            )
            sent.add(seq)
        elif mtype == "MISSION_ACK":
            if msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED and len(sent) == total:
                print(f"[MISSION] Upload accepted ({len(sent)}/{total} items)!")
                return total
            if msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED and len(sent) < total:
                # Ignore stale ACK from earlier command, wait for actual MISSION_REQUESTs
                continue
            raise RuntimeError(f"Mission rejected with MAV_MISSION_RESULT={msg.type} (sent {len(sent)}/{total})")
    raise TimeoutError(f"Timed out uploading Tokyo mission to ArduPlane (sent {len(sent)}/{total})")


def update_state(msg, state: PlaneState) -> None:
    mtype = msg.get_type()
    if mtype == "HEARTBEAT":
        state.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        state.custom_mode = int(msg.custom_mode)
    elif mtype == "STATUSTEXT":
        text = msg.text.rstrip("\x00")
        print(f"[FC] {text}")
    elif mtype == "GLOBAL_POSITION_INT":
        state.lat = msg.lat / 1e7
        state.lon = msg.lon / 1e7
        state.alt_agl = msg.relative_alt / 1000.0
        state.has_fix = True
    elif mtype == "VFR_HUD":
        state.groundspeed = float(msg.groundspeed)
    elif mtype == "MISSION_CURRENT":
        state.mission_seq = int(msg.seq)


def main() -> None:
    parser = argparse.ArgumentParser(description="ArduDeck Tokyo Fixed-Wing Sightseeing Demo")
    parser.add_argument("--ardupilot-dir", type=str, default=os.environ.get("ARDUPILOT_DIR"),
                        help="Path to ArduPilot repository root (e.g. ~/ardupilot)")
    parser.add_argument("--gcs-port", type=int, default=14550,
                        help="Local UDP port where ArduDeck listens (default: 14550)")
    parser.add_argument("--tcp-port", type=int, default=14560,
                        help="LAN TCP server port for mobile/remote GCS (default: 14560)")
    parser.add_argument("--speedup", type=int, default=1,
                        help="SITL simulation speedup factor (default: 1)")
    parser.add_argument("--route-only", action="store_true",
                        help="Only calculate and print the Tokyo route distance, then exit")
    args = parser.parse_args()

    total_km = print_route_summary()
    if args.route_only:
        return

    procs: List[subprocess.Popen] = []
    run_root = Path("/tmp/ardudeck_tokyo_sitl")
    run_root.mkdir(parents=True, exist_ok=True)

    def cleanup(*_sig) -> None:
        print("\n[SHUTDOWN] Stopping Tokyo ArduPlane SITL...")
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        subprocess.run(["killall", "-9", "arduplane"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    subprocess.run(["killall", "-9", "arduplane"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)

    ardupilot_root = find_ardupilot_root(args.ardupilot_dir)
    plane_proc, plane_log = launch_plane_sitl(
        ardupilot_root, TOKYO_TAKEOFF_HOME, run_root / "plane_0", speedup=args.speedup
    )
    procs.append(plane_proc)
    conn = connect_tcp_with_retry(5760, plane_proc, plane_log)

    bridge = GcsBridge(local_udp_port=args.gcs_port, remote_tcp_port=args.tcp_port)
    print(f"[BRIDGE] Forwarding ArduPlane telemetry to ArduDeck at udp://127.0.0.1:{args.gcs_port}")
    print(f"[BRIDGE] Remote / Mobile GCS TCP available at {bridge.lan_ip}:{args.tcp_port}")

    # Configure fixed-wing cruise, realistic 16Ah battery, & takeoff parameters
    # (Isolated to /tmp/ardudeck_tokyo_sitl with -w; does not affect any other SITL tests)
    set_param(conn, "SYSID_THISMAV", 1)
    set_param(conn, "AIRSPEED_CRUISE", CRUISE_AIRSPEED_MPS)
    set_param(conn, "TRIM_ARSPD_CM", CRUISE_AIRSPEED_MPS * 100.0)
    set_param(conn, "TKOFF_ALT", 80.0)
    set_param(conn, "WP_LOITER_RAD", LOITER_RADIUS_M)
    set_param(conn, "BATT_CAPACITY", 16000.0)  # 16,000 mAh pack (~50% used over 38.5 km tour)
    set_param(conn, "BATT_LOW_MAH", 3200.0)    # 20% low-battery reserve
    set_param(conn, "BATT_CRT_MAH", 1600.0)    # 10% critical reserve
    set_param(conn, "THR_FAILSAFE", 0)
    set_param(conn, "FS_GCS_ENABL", 0)
    set_param(conn, "FENCE_ENABLE", 0)
    set_param(conn, "RTL_AUTOLAND", 0)
    request_streams(conn, rate_hz=10)

    state = PlaneState()
    stage = "UPLOAD"  # UPLOAD -> INIT -> TAKEOFF -> AUTO
    last_arm_t = 0.0
    last_log_t = 0.0
    last_hb_t = 0.0
    last_seq_printed = -1
    flown_m = 0.0
    prev_pos: Optional[Tuple[float, float]] = None

    print(f"\n[DEMO] Tokyo Fixed-Wing Tour ready! Planned route: {total_km:.2f} km\n")

    while True:
        now = time.time()
        if now - last_hb_t >= 1.0:
            last_hb_t = now
            conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0,
            )

        while True:
            msg = conn.recv_match(blocking=False)
            if msg is None:
                break
            buf = msg.get_msgbuf()
            if buf:
                bridge.broadcast(bytes(buf))
            update_state(msg, state)

        for data in bridge.poll_incoming():
            conn.write(data)

        if state.has_fix:
            if prev_pos is not None and stage not in ("UPLOAD", "INIT"):
                step_m = haversine_m(prev_pos[0], prev_pos[1], state.lat, state.lon)
                if 0.1 < step_m < 500.0:
                    flown_m += step_m
            prev_pos = (state.lat, state.lon)

            if stage == "UPLOAD":
                upload_tokyo_mission(conn)
                conn.mav.mission_set_current_send(1, 1, 1)
                stage = "INIT"

            elif stage == "INIT":
                if state.armed:
                    print("[SYS 1] Armed over Tokyo Bay! Rolling for takeoff...")
                    stage = "TAKEOFF"
                elif now - last_arm_t >= 1.5:
                    last_arm_t = now
                    set_mode_ardupilot(conn, 13)  # TAKEOFF = 13
                    arm_plane(conn)

            elif stage == "TAKEOFF":
                if not state.armed:
                    stage = "INIT"
                elif state.alt_agl >= 80.0:
                    conn.mav.mission_set_current_send(1, 1, 1)
                    set_mode_ardupilot(conn, 10)  # AUTO = 10
                    print("[SYS 1] Airborne (>80m)! Switching to AUTO mode for Tokyo Landmark Tour.")
                    stage = "AUTO"
                elif now - last_log_t >= 2.0:
                    last_log_t = now
                    print(f"[TAKEOFF] Alt: {state.alt_agl:.1f}m AGL | Speed: {state.groundspeed:.1f} m/s")

            elif stage == "AUTO":
                if state.mission_seq != last_seq_printed:
                    last_seq_printed = state.mission_seq
                    if 1 <= state.mission_seq <= len(TOKYO_LANDMARKS):
                        wp_name = TOKYO_LANDMARKS[state.mission_seq - 1][0]
                        print(f"[TOUR] Heading to WP {state.mission_seq}/{len(TOKYO_LANDMARKS)}: {wp_name}")
                    elif state.mission_seq > len(TOKYO_LANDMARKS):
                        print(f"[TOUR] Reached final landmark! Loitering over Tokyo Skytree (Flown: {flown_m / 1000.0:.2f} km)")

                if now - last_log_t >= 5.0:
                    last_log_t = now
                    print(
                        f"[STATUS] WP {state.mission_seq} | Alt: {state.alt_agl:.0f}m AGL | "
                        f"GS: {state.groundspeed:.1f} m/s | Flown: {flown_m / 1000.0:.2f} km"
                    )

        time.sleep(0.02)


if __name__ == "__main__":
    main()
