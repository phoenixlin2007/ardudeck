/**
 * Synthetic-vision view: a Garmin-SVT-style 3D terrain world locked to one
 * vehicle's GPS position and attitude, with the same overlay stack as the live
 * camera feed drawn on top. Needs no configured camera source — only a position
 * fix — so it works for any vehicle, and is the natural fallback when a real
 * feed is unavailable.
 *
 * The Three.js scene runs its own rAF loop reading a pose ref (so high-rate
 * attitude updates drive the camera without re-mounting anything). Terrain is
 * streamed from the DEM around the vehicle and rebuilt as it travels.
 */

import { useEffect, useRef, useState, type ReactNode, type RefObject } from 'react';
import type { OsdLayers } from '../../../shared/camera-types';
import type { FleetVehicle } from '../../hooks/useFleet';
import { useTelemetryStore } from '../../stores/telemetry-store';
import { useSmoothTelemetry } from '../../perf/render-governor';
import { useConnectionStore } from '../../stores/connection-store';
import { getVehicleClass } from '../../../shared/telemetry-types';
import { useFleetTelemetryStore } from '../../stores/fleet-telemetry-store';
import { useActiveVehicleStore } from '../../stores/active-vehicle-store';
import { useVehicleAppearanceStore, resolveVehicleColor } from '../../stores/vehicle-appearance-store';
import { useCameraStore } from '../../stores/camera-store';
import { CameraOverlays } from './CameraOverlays';
import { createSvtScene, type SvtScene, type SvtPose, type SvtPeerVehicle } from './svt/svt-scene';
import {
  approxDistanceM,
  buildTerrainGeometry,
  loadElevationGrid,
  metersPerDegLon,
  M_PER_DEG_LAT,
  sampleElevation,
  SVT_QUALITY,
  SVT_REBUILD_DISTANCE_M,
  type ElevationGrid,
  type SvtQuality,
} from './svt/svt-terrain';
import { loadDrapeRings, recenterDistanceM, validImagery } from './svt/svt-satellite';
import { loadBuildings, buildBuildingsGeometry, cancelBuildingLoads, BUILDINGS_RECENTER_M } from './svt/svt-buildings';
import { validBuildings } from './svt/svt-building-sources';
import {
  eulerRatesFromBody,
  lerpAngle,
  predictAngle,
  predictPosition,
  pushSample,
  type Sample,
} from './svt/svt-pose-buffer';
import { useTranslation } from 'react-i18next';
import { observeSettledResize } from '../../utils/observe-settled-resize';
import { subscribeTerrainChanged, useTileSourceStore } from '../../stores/tile-source-store';
import { meshesFor, useSceneMeshStore } from '../../stores/scene-mesh-store';

/**
 * Module-level cache of the last loaded terrain grid per vehicle. Grids are plain
 * data (not GPU resources), so they survive this component unmounting/remounting
 * across screen switches — a remount rebuilds the mesh from the cached grid
 * instantly instead of re-fetching and re-decoding tiles. Bounded to a few
 * vehicles so it never grows unbounded.
 */
const gridCache = new Map<string, ElevationGrid>();
subscribeTerrainChanged(() => gridCache.clear());
const GRID_CACHE_MAX = 6;

function cacheGrid(key: string, grid: ElevationGrid): void {
  gridCache.delete(key);
  gridCache.set(key, grid);
  while (gridCache.size > GRID_CACHE_MAX) {
    const oldest = gridCache.keys().next().value;
    if (oldest === undefined) break;
    gridCache.delete(oldest);
  }
}

interface SyntheticVisionViewProps {
  vehicle: FleetVehicle | null;
  /** True when this view's vehicle is the active selection (enables the full HUD). */
  isPrimary: boolean;
  osd: OsdLayers;
  /** Grid mode: clicking the tile activates the vehicle. */
  onActivate?: () => void;
  /** Rendered inside the view with its canvas and container, for the stream window's publisher. */
  streamSlot?: (refs: { canvasRef: RefObject<HTMLCanvasElement>; containerRef: RefObject<HTMLDivElement> }) => ReactNode;
}

/** Beyond this a position sample is a teleport (new vehicle, first fix), not
 * motion, and the buffer restarts instead of gliding across the world. */
const SNAP_DIST_M = 250;

/** Terrain clearance below this raises the caution chip, the order of
 * magnitude general-aviation synthetic vision annunciates at. */
const TERRAIN_CAUTION_M = 30;

/** Render floor above the DEM: a rover's camera mast, an aircraft's clearance. */
const ROVER_EYE_M = 1;
const AIR_EYE_M = 2;

interface PosSample {
  lat: number;
  lon: number;
  altMsl: number;
  agl: number;
  /** NED velocity, m/s, used to carry the position forward between fixes. */
  vN: number;
  vE: number;
  vD: number;
}

/** How fast the shown position closes on the predicted one. Long enough to
 * absorb a fix correction without a visible jump. */
const POS_BLEND_TAU_S = 0.15;
interface AttSample {
  rollDeg: number;
  pitchDeg: number;
  headingDeg: number;
  /** Body rates, deg/s, used to carry the attitude forward between samples. */
  rollRate: number;
  pitchRate: number;
  yawRate: number;
}

/** How fast the shown attitude closes on the predicted one. Short enough to be
 * imperceptible, long enough to absorb the step when a new sample lands. */
const ATT_BLEND_TAU_S = 0.04;

/** ATTITUDE yaw (deg, may be signed) wrapped to a compass heading; falls back
 * to the VFR_HUD heading when yaw is not being streamed. */
function headingFrom(yawDeg: number, fallbackDeg: number): number {
  if (yawDeg === 0) return fallbackDeg;
  return ((yawDeg % 360) + 360) % 360;
}

function blendPos(a: PosSample, b: PosSample, u: number): PosSample {
  return {
    ...b,
    lat: a.lat + (b.lat - a.lat) * u,
    lon: a.lon + (b.lon - a.lon) * u,
    altMsl: a.altMsl + (b.altMsl - a.altMsl) * u,
    agl: a.agl + (b.agl - a.agl) * u,
  };
}

/** Position is dead-reckoned from the newest fix, never rendered in the past:
 * fixes arrive at a few hertz, so interpolating between them holds the world
 * still and then jumps it. */
function predictPos(sample: Sample<PosSample>, now: number): PosSample {
  const dt = (now - sample.t) / 1000;
  return {
    ...sample.v,
    ...predictPosition(sample.v, dt, M_PER_DEG_LAT, metersPerDegLon(sample.v.lat)),
  };
}

function lerpAtt(a: AttSample, b: AttSample, u: number): AttSample {
  return {
    ...b,
    rollDeg: lerpAngle(a.rollDeg, b.rollDeg, u),
    pitchDeg: lerpAngle(a.pitchDeg, b.pitchDeg, u),
    headingDeg: lerpAngle(a.headingDeg, b.headingDeg, u),
  };
}

/**
 * Attitude is predicted forward from the newest sample with the body rates,
 * never rendered in the past: angular lag is what a pilot notices first, and a
 * yaw drags the whole world with it.
 */
function predictAtt(sample: Sample<AttSample>, now: number): AttSample {
  const dt = (now - sample.t) / 1000;
  const v = sample.v;
  // Body rates are what the airframe measures; the camera's angles change at
  // the Euler rates, which differ as soon as it banks.
  const e = eulerRatesFromBody(v.rollDeg, v.pitchDeg, { p: v.rollRate, q: v.pitchRate, r: v.yawRate });
  return {
    ...v,
    rollDeg: predictAngle(v.rollDeg, e.rollRate, dt),
    pitchDeg: predictAngle(v.pitchDeg, e.pitchRate, dt),
    headingDeg: predictAngle(v.headingDeg, e.yawRate, dt),
  };
}

export function SyntheticVisionView({ vehicle, isPrimary, osd, onActivate, streamSlot }: SyntheticVisionViewProps) {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<SvtScene | null>(null);
  const poseRef = useRef<SvtPose | null>(null);
  const gridCenterRef = useRef<{ lat: number; lon: number } | null>(null);
  // Monotonic token so only the newest in-flight load applies — a cancelled or
  // superseded load can never wedge the view on "loading" (important in swarms,
  // where active-vehicle churn cancels loads for parked vehicles that then never
  // move enough to re-trigger).
  const loadTokenRef = useRef(0);
  // The scene is static between inputs: the loop repaints only when the pose,
  // viewport or terrain changed, instead of burning GPU at display rate.
  const dirtyRef = useRef(true);
  // Timestamped history per channel: the camera renders a fraction of a second
  // behind live and walks between samples at constant speed, instead of easing
  // toward the newest one (which surges then coasts once per tick). Position
  // and attitude arrive at different rates, so they buffer separately.
  const posBufRef = useRef<Array<Sample<PosSample>>>([]);
  const attBufRef = useRef<Array<Sample<AttSample>>>([]);
  const shownPosRef = useRef<PosSample | null>(null);
  const shownAttRef = useRef<AttSample | null>(null);
  const lastFrameRef = useRef(0);
  // Offset between the vehicle's altitude reference and the DEM's, measured
  // while it sits on the ground: SITL origins, baro drift and geoid models all
  // shift one against the other.
  const datumOffsetRef = useRef(0);
  const [clearanceM, setClearanceM] = useState(NaN);
  // Published to React at ~30 Hz for the world-locked HUD overlay, which has to
  // sit on the same interpolated terrain or its symbology swims.
  const shownRef = useRef<SvtPose | null>(null);
  const [shownPose, setShownPose] = useState<SvtPose | null>(null);
  const publishAtRef = useRef(0);
  const isPrimaryRef = useRef(isPrimary);
  isPrimaryRef.current = isPrimary;
  const [terrainStatus, setTerrainStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  // The SVT camera's vertical FOV, captured on mount so the world-locked HUD
  // overlay can drive its own camera with the exact same fov (zero calibration).
  const [svtFov, setSvtFov] = useState(62);
  const satellite = useCameraStore((s) => s.svtSatellite);
  const imagery = useCameraStore((s) => validImagery(s.svtImagery));
  const quality = useCameraStore((s) => s.svtQuality);
  const buildingsSource = useCameraStore((s) => validBuildings(s.svtBuildings));
  const roofImagery = useCameraStore((s) => s.svtRoofImagery);
  const facades = useCameraStore((s) => s.svtFacades);
  const google3dTiles = useCameraStore((s) => s.google3dTiles);
  const googleMapsApiKey = useCameraStore((s) => s.googleMapsApiKey);
  const setGoogleMapsApiKey = useCameraStore((s) => s.setGoogleMapsApiKey);
  const setGoogle3dTiles = useCameraStore((s) => s.setGoogle3dTiles);
  const [g3dStatus, setG3dStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [g3dErrorMsg, setG3dErrorMsg] = useState<string | undefined>(undefined);
  const [g3dAttribution, setG3dAttribution] = useState<string>('Google Photorealistic 3D Tiles');
  const [draftApiKey, setDraftApiKey] = useState('');
  useEffect(() => {
    sceneRef.current?.setBuildingStyle({ roofImagery, facades });
    dirtyRef.current = true;
  }, [roofImagery, facades]);
  useEffect(() => {
    const maxSSE = quality === 'high' ? 10 : quality === 'low' ? 26 : 16;
    sceneRef.current?.setGoogle3dTiles({
      enabled: google3dTiles,
      apiKey: googleMapsApiKey,
      maxScreenSpaceError: maxSSE,
      onStatusChange: (st, msg) => {
        setG3dStatus(st);
        setG3dErrorMsg(msg);
        dirtyRef.current = true;
      },
      onAttributionChange: (attr) => {
        if (attr) setG3dAttribution(`Google · ${attr}`);
      },
    });
    dirtyRef.current = true;
  }, [google3dTiles, googleMapsApiKey, quality]);
  const buildingsTokenRef = useRef(0);
  const [buildingsStatus, setBuildingsStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const buildingsShownRef = useRef<string | null>(null);
  const buildingsRequestedRef = useRef<string | null>(null);
  // The grid the drape follows: state, not a ref, so a new patch re-runs the
  // imagery effect. Quality changes and the toggle do the same.
  const [drapeGrid, setDrapeGrid] = useState<ElevationGrid | null>(null);
  const drapeTokenRef = useRef(0);
  // One imagery load at a time: overlapping runs hold several hundred MB of
  // mosaics at once. A request that arrives during one is remembered and run
  // after it, never dropped: a parked aircraft produces no further move to
  // re-trigger on, so a dropped one left the drape blank until a remount.
  const drapeBusyRef = useRef(false);
  const drapePendingRef = useRef(false);
  const [drapeRetry, setDrapeRetry] = useState(0);
  // Quality the live mesh was built at: changing the level has to rebuild it
  // even though the vehicle has not moved.
  const terrainQualityRef = useRef<SvtQuality | null>(null);

  const mavType = useConnectionStore((s) => s.connectionState.mavType);
  // A rover sits on the surface: terrain clearance is zero by definition, so
  // the CFIT caution is an aircraft idea and would latch red for a whole drive.
  const onSurface = getVehicleClass(mavType) === 'rover';
  // Read by the render loop, which never re-closes over props.
  const eyeFloorRef = useRef(ROVER_EYE_M);
  eyeFloorRef.current = onSurface ? ROVER_EYE_M : AIR_EYE_M;

  // Flies the camera off the live position: a thinned store reads as the
  // aircraft lurching backwards between samples.
  useSmoothTelemetry();

  const flatArmed = useTelemetryStore((s) => s.flight.armed);
  const flatClimb = useTelemetryStore((s) => s.vfrHud.climb);
  const flatGroundspeed = useTelemetryStore((s) => s.vfrHud.groundspeed);
  const flatAttitude = useTelemetryStore((s) => s.attitude);
  const flatAltMsl = useTelemetryStore((s) => s.position.alt);
  const flatAltAgl = useTelemetryStore((s) => s.position.relativeAlt);
  const fleetAttitude = useFleetTelemetryStore((s) => (vehicle ? s.byVehicle[vehicle.key]?.attitude : undefined));

  const vehicleKey = vehicle?.key ?? null;
  const datumByVehicleRef = useRef<Map<string, number>>(new Map());

  const position = vehicle?.position ?? null;
  const att = isPrimary ? flatAttitude : fleetAttitude;
  const armed = (isPrimary ? flatArmed : false) || (vehicle?.armed ?? false);
  const climbRate = isPrimary ? flatClimb : 0;
  const groundspeed = isPrimary ? Math.max(flatGroundspeed, vehicle?.groundspeed ?? 0) : (vehicle?.groundspeed ?? 0);

  // Calibrate the altitude datum while the vehicle is on the ground: there its
  // true eye height IS the terrain height, so the difference from its reported
  // MSL is the offset between the two references. Once airborne the offset is
  // held, so a climb raises the eye by exactly what the vehicle climbed and
  // rising ground rises into view as it does in the world.
  useEffect(() => {
    // Disarmed and stationary means it is standing on the ground, so its eye
    // height IS the terrain height. Also guard against transient store resets
    // during fleet vehicle switches where `armed` or `vfrHud` may briefly read
    // false/0 while `GLOBAL_POSITION_INT` already reports airborne altitude/velocity.
    if (armed || !drapeGrid || !position) return;
    if (Math.abs(climbRate) > 0.5 || groundspeed > 1) return;
    const agl = isPrimary
      ? Math.max(Math.abs(flatAltAgl), Math.abs(vehicle?.altitudeAgl ?? 0))
      : Math.abs(vehicle?.altitudeAgl ?? 0);
    if (agl > 3.0) return;
    const posState = isPrimary
      ? useTelemetryStore.getState().position
      : (vehicleKey ? useFleetTelemetryStore.getState().byVehicle[vehicleKey]?.position : undefined);
    if (posState && (Math.hypot(posState.vx, posState.vy) > 1.0 || Math.abs(posState.vz) > 0.5)) return;
    if (approxDistanceM(drapeGrid.centerLat, drapeGrid.centerLon, position[0], position[1]) > SVT_REBUILD_DISTANCE_M) return;
    const altMsl = isPrimary ? flatAltMsl : vehicle?.altitudeMsl ?? 0;
    if (altMsl === 0) return;
    const nextOffset = sampleElevation(drapeGrid, position[0], position[1]) - altMsl;
    datumOffsetRef.current = nextOffset;
    if (vehicleKey) datumByVehicleRef.current.set(vehicleKey, nextOffset);
    dirtyRef.current = true;
  }, [armed, drapeGrid, position, climbRate, groundspeed, vehicle?.altitudeMsl, vehicle?.altitudeAgl, flatAltMsl, flatAltAgl, isPrimary, vehicleKey]);

  // Latest pose for the render loop — assigned during render (cheap, idempotent).
  const nextPose = position
    ? {
        lat: position[0],
        lon: position[1],
        agl: vehicle?.altitudeAgl ?? 0,
        altMsl: isPrimary ? flatAltMsl : vehicle?.altitudeMsl ?? 0,
        eyeMsl: 0, // resolved against the datum offset in the render loop
        rollDeg: att?.roll ?? 0,
        pitchDeg: att?.pitch ?? 0,
        headingDeg: vehicle?.heading ?? 0,
      }
    : null;
  if ((nextPose === null) !== (poseRef.current === null)) {
    if (nextPose === null) {
      posBufRef.current = [];
      attBufRef.current = [];
      shownRef.current = null;
    }
    dirtyRef.current = true;
  }
  poseRef.current = nextPose;

  const vehicleKeyRef = useRef<string | null>(vehicleKey);
  vehicleKeyRef.current = vehicleKey;
  const ownSysidRef = useRef<number | null>(vehicle?.sysid ?? null);
  ownSysidRef.current = vehicle?.sysid ?? null;
  const peerSmoothRef = useRef<
    Map<
      string,
      {
        lastUpdateWall: number;
        posSample: Sample<PosSample>;
        attSample: Sample<AttSample>;
        shownPos: PosSample;
        shownAtt: AttSample;
      }
    >
  >(new Map());
  const hadPeersRef = useRef(false);

  // ─── Telemetry sampling ───────────────────────────────────────────────────
  // Straight off the store, not out of the render body: React coalesces updates
  // that land in the same frame and would stamp them with the render time, so
  // attitude lost samples and got uneven timestamps (it stepped while position
  // glided). A store subscription sees every update at its arrival instant.
  useEffect(() => {
    // Switching vehicles in the fleet bar: restore this vehicle's ground datum
    // offset and reset interpolation buffers so the camera snaps cleanly to the
    // newly selected vehicle at its true altitude.
    datumOffsetRef.current = (vehicleKey ? datumByVehicleRef.current.get(vehicleKey) : undefined) ?? 0;
    posBufRef.current = [];
    attBufRef.current = [];
    shownPosRef.current = null;
    shownAttRef.current = null;
    dirtyRef.current = true;

    const pushPos = (t: number, v: PosSample) => {
      const buf = posBufRef.current;
      const last = buf[buf.length - 1]?.v;
      if (
        last && last.lat === v.lat && last.lon === v.lon && last.altMsl === v.altMsl &&
        last.agl === v.agl && last.vN === v.vN && last.vE === v.vE && last.vD === v.vD
      ) return;
      if (last && approxDistanceM(last.lat, last.lon, v.lat, v.lon) > SNAP_DIST_M) posBufRef.current = [];
      pushSample(posBufRef.current, t, v);
      dirtyRef.current = true;
    };
    const pushAtt = (t: number, v: AttSample) => {
      const buf = attBufRef.current;
      const last = buf[buf.length - 1]?.v;
      if (
        last && last.rollDeg === v.rollDeg && last.pitchDeg === v.pitchDeg &&
        last.headingDeg === v.headingDeg && last.rollRate === v.rollRate &&
        last.pitchRate === v.pitchRate && last.yawRate === v.yawRate
      ) return;
      pushSample(attBufRef.current, t, v);
      dirtyRef.current = true;
    };

    // Seed buffers immediately from the selected vehicle's latest accumulated telemetry
    const seedNow = performance.now();
    const seedTel = vehicleKey ? useFleetTelemetryStore.getState().byVehicle[vehicleKey] : undefined;
    if (seedTel?.gps && seedTel.gps.fixType >= 2 && (seedTel.gps.lat !== 0 || seedTel.gps.lon !== 0) && seedTel.position) {
      pushPos(seedNow, {
        lat: seedTel.gps.lat,
        lon: seedTel.gps.lon,
        altMsl: seedTel.position.alt,
        agl: seedTel.position.relativeAlt,
        vN: seedTel.position.vx,
        vE: seedTel.position.vy,
        vD: seedTel.position.vz,
      });
    }
    if (seedTel?.attitude) {
      pushAtt(seedNow, {
        rollDeg: seedTel.attitude.roll,
        pitchDeg: seedTel.attitude.pitch,
        headingDeg: headingFrom(seedTel.attitude.yaw, seedTel.vfrHud?.heading ?? 0),
        rollRate: seedTel.attitude.rollSpeed,
        pitchRate: seedTel.attitude.pitchSpeed,
        yawRate: seedTel.attitude.yawSpeed,
      });
    }

    if (isPrimary) {
      return useTelemetryStore.subscribe((st, prevSt) => {
        const t = performance.now();
        if (st.gps !== prevSt.gps || st.position !== prevSt.position) {
          if (st.gps.fixType >= 2 && (st.gps.lat !== 0 || st.gps.lon !== 0)) {
            pushPos(t, {
              lat: st.gps.lat,
              lon: st.gps.lon,
              altMsl: st.position.alt,
              agl: st.position.relativeAlt,
              vN: st.position.vx,
              vE: st.position.vy,
              vD: st.position.vz,
            });
          }
        }
        if (st.attitude !== prevSt.attitude || st.vfrHud !== prevSt.vfrHud) {
          pushAtt(t, {
            rollDeg: st.attitude.roll,
            pitchDeg: st.attitude.pitch,
            // ATTITUDE's own yaw, not VFR_HUD's whole-degree heading: it shares
            // this message's timestamp and rates, so the prediction advances
            // continuously instead of resetting to a stale heading.
            headingDeg: headingFrom(st.attitude.yaw, st.vfrHud.heading),
            rollRate: st.attitude.rollSpeed,
            pitchRate: st.attitude.pitchSpeed,
            yawRate: st.attitude.yawSpeed,
          });
        }
      });
    }
    if (!vehicleKey) return;
    return useFleetTelemetryStore.subscribe((st, prevSt) => {
      const tel = st.byVehicle[vehicleKey];
      if (!tel || tel === prevSt.byVehicle[vehicleKey]) return;
      const t = performance.now();
      if (tel.gps && tel.gps.fixType >= 2 && tel.position) {
        pushPos(t, {
          lat: tel.gps.lat,
          lon: tel.gps.lon,
          altMsl: tel.position.alt,
          agl: tel.position.relativeAlt,
          vN: tel.position.vx,
          vE: tel.position.vy,
          vD: tel.position.vz,
        });
      }
      if (tel.attitude) {
        pushAtt(t, {
          rollDeg: tel.attitude.roll,
          pitchDeg: tel.attitude.pitch,
          headingDeg: headingFrom(tel.attitude.yaw, tel.vfrHud?.heading ?? 0),
          rollRate: tel.attitude.rollSpeed,
          pitchRate: tel.attitude.pitchSpeed,
          yawRate: tel.attitude.yawSpeed,
        });
      }
    });
  }, [isPrimary, vehicleKey]);

  // ─── Scene lifecycle (mount once) ─────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const scene = createSvtScene(canvas);
    sceneRef.current = scene;
    const cam = useCameraStore.getState();
    scene.setBuildingStyle({ roofImagery: cam.svtRoofImagery, facades: cam.svtFacades });
    scene.setGoogle3dTiles({
      enabled: cam.google3dTiles,
      apiKey: cam.googleMapsApiKey,
      maxScreenSpaceError: cam.svtQuality === 'high' ? 10 : cam.svtQuality === 'low' ? 26 : 16,
      onStatusChange: (st, msg) => {
        setG3dStatus(st);
        setG3dErrorMsg(msg);
        dirtyRef.current = true;
      },
      onAttributionChange: (attr) => {
        if (attr) setG3dAttribution(`Google · ${attr}`);
      },
    });
    setSvtFov(scene.getFov());

    let lastWidth = 0;
    let lastHeight = 0;
    const sizeToContainer = () => {
      const r = container.getBoundingClientRect();
      const width = Math.round(r.width);
      const height = Math.round(r.height);
      if (width === lastWidth && height === lastHeight) return;
      lastWidth = width;
      lastHeight = height;
      scene.resize(width, height);
      dirtyRef.current = true;
    };
    sizeToContainer();
    // Reallocating the WebGL buffer on every frame of a layout animation stalls it;
    // the canvas stretches via CSS until the resize settles.
    const stopObservingSize = observeSettledResize(container, sizeToContainer);

    let raf = 0;
    const loop = () => {
      const now = performance.now();
      const posBuf = posBufRef.current;
      const attBuf = attBufRef.current;
      const frameDt = lastFrameRef.current ? Math.min(0.1, (now - lastFrameRef.current) / 1000) : 0;
      lastFrameRef.current = now;
      const newestPos = posBuf[posBuf.length - 1];
      let pos: PosSample | null = null;
      if (newestPos) {
        const predicted = predictPos(newestPos, now);
        const held = shownPosRef.current;
        pos = held && approxDistanceM(held.lat, held.lon, predicted.lat, predicted.lon) <= SNAP_DIST_M
          ? blendPos(held, predicted, 1 - Math.exp(-frameDt / POS_BLEND_TAU_S))
          : predicted;
        shownPosRef.current = pos;
      } else {
        shownPosRef.current = null;
      }
      const newestAtt = attBuf[attBuf.length - 1];
      let att: AttSample | null = null;
      if (newestAtt) {
        const predicted = predictAtt(newestAtt, now);
        const held = shownAttRef.current;
        att = held ? lerpAtt(held, predicted, 1 - Math.exp(-frameDt / ATT_BLEND_TAU_S)) : predicted;
        shownAttRef.current = att;
      } else {
        shownAttRef.current = null;
      }
      const shown: SvtPose | null = pos && att
        ? { ...pos, ...att, eyeMsl: pos.altMsl + datumOffsetRef.current, eyeFloorM: eyeFloorRef.current }
        : null;

      // ─── Peer SITL / Fleet Vehicles (enabled with 3D GLB or Google 3D toggle) ─
      const showPeerVehicles = useSceneMeshStore.getState().enabled || useCameraStore.getState().google3dTiles;
      if (!showPeerVehicles) {
        if (hadPeersRef.current) {
          hadPeersRef.current = false;
          peerSmoothRef.current.clear();
          scene.setPeerVehicles?.([], 0);
          dirtyRef.current = true;
        }
      } else {
        const known = Object.values(useActiveVehicleStore.getState().knownVehicles);
        const byVehicle = useFleetTelemetryStore.getState().byVehicle;
        const overrides = useVehicleAppearanceStore.getState().overrides;
        const curKey = vehicleKeyRef.current;
        const curSysid = ownSysidRef.current;
        const peers: SvtPeerVehicle[] = [];
        const activePeerKeys = new Set<string>();

        for (const v of known) {
          if (curKey ? v.key === curKey : (curSysid != null && v.sysid === curSysid)) continue;
          const tel = byVehicle[v.key];
          if (!tel) continue;
          const g = tel.gps;
          const p = tel.position;
          let peerLat = 0;
          let peerLon = 0;
          if (g && (g.fixType ?? 0) >= 2 && (g.lat !== 0 || g.lon !== 0)) {
            peerLat = g.lat;
            peerLon = g.lon;
          } else if (p && (p.lat !== 0 || p.lon !== 0)) {
            peerLat = p.lat;
            peerLon = p.lon;
          } else {
            continue;
          }

          activePeerKeys.add(v.key);
          const rawPos: PosSample = {
            lat: peerLat,
            lon: peerLon,
            altMsl: p?.alt ?? 0,
            agl: p?.relativeAlt ?? 0,
            vN: p?.vx ?? 0,
            vE: p?.vy ?? 0,
            vD: p?.vz ?? 0,
          };
          const rawAtt: AttSample = {
            rollDeg: tel.attitude?.roll ?? 0,
            pitchDeg: tel.attitude?.pitch ?? 0,
            headingDeg: headingFrom(tel.attitude?.yaw ?? 0, tel.vfrHud?.heading ?? 0),
            rollRate: tel.attitude?.rollSpeed ?? 0,
            pitchRate: tel.attitude?.pitchSpeed ?? 0,
            yawRate: tel.attitude?.yawSpeed ?? 0,
          };

          let st = peerSmoothRef.current.get(v.key);
          if (!st) {
            st = {
              lastUpdateWall: tel.lastUpdate,
              posSample: { t: now, v: rawPos },
              attSample: { t: now, v: rawAtt },
              shownPos: rawPos,
              shownAtt: rawAtt,
            };
            peerSmoothRef.current.set(v.key, st);
          } else if (tel.lastUpdate !== st.lastUpdateWall) {
            st.lastUpdateWall = tel.lastUpdate;
            st.posSample = { t: now, v: rawPos };
            st.attSample = { t: now, v: rawAtt };
          }

          const dtPos = Math.min(0.5, Math.max(0, (now - st.posSample.t) / 1000));
          const predPos: PosSample = {
            ...st.posSample.v,
            ...predictPosition(st.posSample.v, dtPos, M_PER_DEG_LAT, metersPerDegLon(st.posSample.v.lat)),
          };
          st.shownPos = approxDistanceM(st.shownPos.lat, st.shownPos.lon, predPos.lat, predPos.lon) <= SNAP_DIST_M
            ? blendPos(st.shownPos, predPos, 1 - Math.exp(-frameDt / POS_BLEND_TAU_S))
            : predPos;

          const dtAtt = Math.min(0.25, Math.max(0, (now - st.attSample.t) / 1000));
          const eRates = eulerRatesFromBody(st.attSample.v.rollDeg, st.attSample.v.pitchDeg, {
            p: st.attSample.v.rollRate,
            q: st.attSample.v.pitchRate,
            r: st.attSample.v.yawRate,
          });
          const predAtt: AttSample = {
            ...st.attSample.v,
            rollDeg: predictAngle(st.attSample.v.rollDeg, eRates.rollRate, dtAtt),
            pitchDeg: predictAngle(st.attSample.v.pitchDeg, eRates.pitchRate, dtAtt),
            headingDeg: predictAngle(st.attSample.v.headingDeg, eRates.yawRate, dtAtt),
          };
          st.shownAtt = lerpAtt(st.shownAtt, predAtt, 1 - Math.exp(-frameDt / ATT_BLEND_TAU_S));

          peers.push({
            id: v.key,
            label: `SYS ${v.sysid}`,
            color: resolveVehicleColor(overrides, v.key, v.sysid),
            vehicleClass: getVehicleClass(v.mavType),
            lat: st.shownPos.lat,
            lon: st.shownPos.lon,
            altMsl: st.shownPos.altMsl,
            agl: st.shownPos.agl,
            rollDeg: st.shownAtt.rollDeg,
            pitchDeg: st.shownAtt.pitchDeg,
            headingDeg: st.shownAtt.headingDeg,
            armed: tel.flight?.armed ?? false,
          });
        }

        for (const k of peerSmoothRef.current.keys()) {
          if (!activePeerKeys.has(k)) peerSmoothRef.current.delete(k);
        }

        if (peers.length > 0 || hadPeersRef.current) {
          scene.setPeerVehicles?.(peers, datumOffsetRef.current);
          hadPeersRef.current = peers.length > 0;
          dirtyRef.current = true;
        }
      }

      const was = shownRef.current;
      const moved =
        (shown === null) !== (was === null) ||
        (shown !== null && was !== null && (
          shown.lat !== was.lat || shown.lon !== was.lon || shown.eyeMsl !== was.eyeMsl ||
          shown.rollDeg !== was.rollDeg || shown.pitchDeg !== was.pitchDeg ||
          shown.headingDeg !== was.headingDeg
        ));
      shownRef.current = shown;
      if (dirtyRef.current || moved) {
        dirtyRef.current = false;
        if (shown) scene.setPose(shown);
        scene.render();
        if (isPrimaryRef.current && now - publishAtRef.current > 33) {
          publishAtRef.current = now;
          setShownPose(shown);
          setClearanceM(scene.getClearanceM());
        }
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      stopObservingSize();
      scene.dispose();
      sceneRef.current = null;
    };
  }, []);

  // ─── Terrain streaming ────────────────────────────────────────────────────
  const lat = position?.[0] ?? null;
  const lon = position?.[1] ?? null;
  // Re-check on ~100 m moves; the distance gate below avoids needless reloads.
  const terrainVersion = useTileSourceStore((s) => s.version);
  const meshVersion = useSceneMeshStore((s) => s.version);
  // A new or changed module DEM invalidates the patch wherever it is; force the next load.
  useEffect(() => {
    gridCenterRef.current = null;
  }, [terrainVersion]);
  const latKey = lat != null ? Math.round(lat * 1000) : null;
  const lonKey = lon != null ? Math.round(lon * 1000) : null;


  useEffect(() => {
    if (lat == null || lon == null || vehicleKey == null) return;
    const scene = sceneRef.current;
    if (!scene) return;

    // Instant restore after a remount (e.g. screen switch): rebuild the mesh from
    // the cached grid if it still covers us — cheap, no network, no reload flash.
    const cacheKey = `${vehicleKey}:${quality}`;
    if (!scene.hasTerrain()) {
      const cached = gridCache.get(cacheKey);
      if (cached && approxDistanceM(cached.centerLat, cached.centerLon, lat, lon) <= SVT_REBUILD_DISTANCE_M) {
        scene.setTerrain(buildTerrainGeometry(cached), cached);
        dirtyRef.current = true;
        gridCenterRef.current = { lat: cached.centerLat, lon: cached.centerLon };
        terrainQualityRef.current = quality;
        setDrapeGrid(cached);
        setTerrainStatus('ready');
      }
    }

    const center = gridCenterRef.current;
    const needsLoad =
      !center ||
      terrainQualityRef.current !== quality ||
      approxDistanceM(center.lat, center.lon, lat, lon) > SVT_REBUILD_DISTANCE_M;
    if (!needsLoad) return;

    const token = ++loadTokenRef.current; // newest load wins; stale ones no-op
    setTerrainStatus((s) => (s === 'ready' ? s : 'loading'));
    void (async () => {
      try {
        const grid = await loadElevationGrid(lat, lon, quality);
        if (token !== loadTokenRef.current) return; // superseded by a newer load
        sceneRef.current?.setTerrain(buildTerrainGeometry(grid), grid);
        dirtyRef.current = true;
        gridCenterRef.current = { lat, lon };
        terrainQualityRef.current = quality;
        setDrapeGrid(grid);
        cacheGrid(cacheKey, grid);
        setTerrainStatus('ready');
      } catch {
        if (token === loadTokenRef.current) setTerrainStatus((s) => (s === 'ready' ? s : 'error'));
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latKey, lonKey, vehicleKey, quality, terrainVersion]);

  // Module meshes ride the terrain grid; re-place them on every mesh or terrain change.
  useEffect(() => {
    sceneRef.current?.setMeshes(meshesFor('svt'));
    dirtyRef.current = true;
  }, [meshVersion, terrainVersion, terrainStatus]);

  // ─── Satellite drape ──────────────────────────────────────────────────────
  // Rings follow the AIRCRAFT, not the terrain patch: the sharp inner ring is
  // small, so it is re-fetched every few hundred metres of travel (mostly cache
  // hits) instead of sitting at a patch centre kilometres behind.
  const drapeStep = recenterDistanceM() / 111_320;
  const drapeLatKey = lat != null ? Math.round(lat / drapeStep) : null;
  const drapeLonKey = lon != null ? Math.round(lon / drapeStep) : null;

  useEffect(() => {
    const scene = sceneRef.current;
    const grid = drapeGrid;
    if (!scene) return;
    if (!satellite || !grid || lat == null || lon == null) {
      scene.setDrape(null);
      dirtyRef.current = true;
      return;
    }
    if (drapeBusyRef.current) {
      drapePendingRef.current = true;
      return;
    }
    const token = ++drapeTokenRef.current;
    drapeBusyRef.current = true;
    void (async () => {
      try {
        const rings = await loadDrapeRings(grid, SVT_QUALITY[quality].outerRingTiles, { lat, lon }, undefined, undefined, imagery);
        if (token !== drapeTokenRef.current) {
          for (const ring of rings) ring.texture.dispose();
          return;
        }
        sceneRef.current?.setDrape(rings.length > 0 ? rings : null);
        dirtyRef.current = true;
      } catch {
        // Imagery is optional; the elevation ramp stays.
      } finally {
        drapeBusyRef.current = false;
        if (drapePendingRef.current) {
          drapePendingRef.current = false;
          setDrapeRetry((n) => n + 1);
        }
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [satellite, imagery, quality, drapeGrid, drapeLatKey, drapeLonKey, drapeRetry]);

  // ─── Buildings ────────────────────────────────────────────────────────────
  const buildingsStep = BUILDINGS_RECENTER_M / 111_320;
  const buildingsLatKey = lat != null ? Math.round(lat / buildingsStep) : null;
  const buildingsLonKey = lon != null ? Math.round(lon / buildingsStep) : null;

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const grid = drapeGrid;
    if (buildingsSource === 'off' || !grid || lat == null || lon == null) {
      if (buildingsRequestedRef.current !== null) cancelBuildingLoads();
      buildingsRequestedRef.current = null;
      ++buildingsTokenRef.current;
      buildingsShownRef.current = null;
      scene.setBuildings(null);
      setBuildingsStatus('idle');
      dirtyRef.current = true;
      return;
    }
    const token = ++buildingsTokenRef.current;
    // a new source clears at once so the switch is visible; moving on keeps the old set until the next arrives
    if (buildingsRequestedRef.current !== null && buildingsRequestedRef.current !== buildingsSource) cancelBuildingLoads();
    buildingsRequestedRef.current = buildingsSource;
    if (buildingsShownRef.current !== buildingsSource) {
      scene.setBuildings(null);
      dirtyRef.current = true;
    }
    setBuildingsStatus('loading');
    void (async () => {
      try {
        const list = await loadBuildings(buildingsSource, lat, lon);
        if (token !== buildingsTokenRef.current) return;
        const customOrigins = meshesFor('svt').map((m) => m.origin);
        sceneRef.current?.setBuildings(buildBuildingsGeometry(list, grid, { lat, lon }, undefined, customOrigins));
        buildingsShownRef.current = buildingsSource;
        setBuildingsStatus('ready');
        dirtyRef.current = true;
      } catch {
        // the last set stays up; buildings are optional
        if (token === buildingsTokenRef.current) setBuildingsStatus('error');
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [buildingsSource, drapeGrid, buildingsLatKey, buildingsLonKey, meshVersion]);

  const overlayAttitude = att ? { roll: att.roll, pitch: att.pitch } : null;
  // The 3D scene already shows a true banked horizon — drop the flat cyan line.
  const svtOsd: OsdLayers = { ...osd, artificialHorizon: false };

  // Drive the world-locked waypoint overlay with the SAME pose + fov the SVT
  // camera uses, so the symbology lands on the terrain. altMsl is the primary's
  // telemetry MSL (only the primary vehicle draws the overlay).
  const overlayPose = shownPose ?? nextPose;
  const worldOverlay = position && overlayPose
    ? {
        lat: overlayPose.lat,
        lon: overlayPose.lon,
        altMsl: overlayPose.altMsl !== 0 ? overlayPose.altMsl : flatAltMsl,
        yawDeg: overlayPose.headingDeg,
        pitchDeg: overlayPose.pitchDeg,
        rollDeg: overlayPose.rollDeg,
        fov: svtFov,
      }
    : null;

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full overflow-hidden bg-black"
      onClick={onActivate}
      title={onActivate ? t('camera:view.clickToActivate') : undefined}
    >
      <canvas ref={canvasRef} className="block h-full w-full" />
      {streamSlot?.({ canvasRef, containerRef })}

      {position && <CameraOverlays vehicle={vehicle} isPrimary={isPrimary} osd={svtOsd} attitude={overlayAttitude} worldOverlay={worldOverlay} />}

      {!position && (
        <Center>
          <div className="text-sm text-amber-300">{t('camera:svt.noFix')}</div>
          <div className="max-w-[80%] text-[11px] text-white/60">
            {t('camera:svt.needsFix', { vehicle: vehicle?.label ?? t('camera:svt.theVehicle') })}
          </div>
        </Center>
      )}

      {position && armed && !onSurface && Number.isFinite(clearanceM) && clearanceM < TERRAIN_CAUTION_M && (
        <div
          className={
            'absolute left-1/2 top-2 -translate-x-1/2 rounded px-2 py-1 text-[11px] font-semibold tracking-wide ' +
            (clearanceM <= 0 ? 'bg-red-600/90 text-white' : 'bg-amber-500/90 text-black')
          }
        >
          {/* i18n-exempt: GPWS-style callout */}
          {clearanceM <= 0 ? 'TERRAIN' : `TERRAIN ${Math.round(clearanceM)} m`}
        </div>
      )}

      {position && terrainStatus === 'loading' && (
        <div className="absolute bottom-2 left-2 flex items-center gap-2 rounded bg-black/55 px-2 py-1 text-[11px] text-white/80">
          <div className="h-3 w-3 animate-spin rounded-full border-2 border-white/30 border-t-white/90" />
          {t('camera:svt.loadingTerrain')}
        </div>
      )}
      {google3dTiles && (!googleMapsApiKey || g3dStatus === 'error') && (
        <div
          className="absolute left-1/2 top-10 z-20 flex -translate-x-1/2 flex-col gap-1.5 rounded-lg border border-sky-500/40 bg-slate-900/90 p-2.5 text-xs text-white shadow-xl backdrop-blur-md"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between gap-4">
            {/* i18n-exempt */}
            <span className="font-semibold text-sky-300">Google Map Tiles API — Photorealistic 3D Tiles</span>
            <button
              type="button"
              onClick={() => setGoogle3dTiles(false)}
              className="text-[11px] text-white/60 hover:text-white"
            >
              ✕
            </button>
          </div>
          {g3dStatus === 'error' && g3dErrorMsg && (
            <div className="flex items-center justify-between gap-2 text-[11px] text-amber-300">
              {/* i18n-exempt */}
              <span>載入失敗：{g3dErrorMsg}</span>
              {googleMapsApiKey && (
                <button
                  type="button"
                  onClick={() => {
                    setGoogleMapsApiKey('');
                    setDraftApiKey('');
                  }}
                  className="shrink-0 rounded border border-rose-400/40 bg-rose-500/20 px-1.5 py-0.5 text-[10px] text-rose-200 hover:bg-rose-500/30"
                >
                  {/* i18n-exempt */}
                  清除舊 Key
                </button>
              )}
            </div>
          )}
          <div className="flex items-center gap-1.5">
            <input
              type="password"
              value={draftApiKey}
              onChange={(e) => setDraftApiKey(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && draftApiKey.trim()) {
                  setGoogleMapsApiKey(draftApiKey.trim());
                }
              }}
              placeholder="貼上 Google Map Tiles API Key..." // i18n-exempt
              className="w-60 rounded border border-white/20 bg-black/60 px-2 py-1 text-[11px] font-mono text-white focus:border-sky-400 focus:outline-none"
            />
            <button
              type="button"
              onClick={() => {
                if (draftApiKey.trim()) setGoogleMapsApiKey(draftApiKey.trim());
              }}
              className="rounded bg-sky-600 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-sky-500"
            >
              {/* i18n-exempt */}
              套用並載入
            </button>
          </div>
        </div>
      )}
      {position && google3dTiles && g3dStatus === 'loading' && (
        <div className="absolute bottom-8 left-2 flex items-center gap-2 rounded bg-black/55 px-2 py-1 text-[11px] text-sky-300">
          <div className="h-3 w-3 animate-spin rounded-full border-2 border-sky-400/30 border-t-sky-300" />
          {/* i18n-exempt */}
          載入 Google 3D Tiles...
        </div>
      )}
      {position && google3dTiles && g3dStatus === 'ready' && (
        <div className="absolute bottom-1 right-2 text-[9px] text-white/60">
          {g3dAttribution}
        </div>
      )}
      {position && !google3dTiles && buildingsSource !== 'off' && (
        <div className={`absolute bottom-1 right-2 text-[9px] ${buildingsStatus === 'error' ? 'text-amber-300' : 'text-white/50'}`}>
          {buildingsStatus === 'error' ? t('camera:svt.buildingsUnavailable') : t('camera:svt.osmCredit')}
        </div>
      )}
      {position && buildingsStatus === 'loading' && terrainStatus !== 'loading' && (
        <div className="absolute bottom-2 left-2 flex items-center gap-2 rounded bg-black/55 px-2 py-1 text-[11px] text-white/80">
          <div className="h-3 w-3 animate-spin rounded-full border-2 border-white/30 border-t-white/90" />
          {t('camera:svt.loadingBuildings')}
        </div>
      )}
      {position && terrainStatus === 'error' && (
        <div className="absolute bottom-2 left-2 rounded bg-black/55 px-2 py-1 text-[11px] text-amber-300">
          {t('camera:svt.terrainUnavailable')}
        </div>
      )}
    </div>
  );
}

function Center({ children }: { children: React.ReactNode }) {
  return (
    <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/60 text-center">
      {children}
    </div>
  );
}
