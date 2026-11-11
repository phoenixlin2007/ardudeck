/**
 * Synthetic-vision Three.js scene: a daytime sky dome + sun, distance haze, a
 * far sea-level ground plane, and a replaceable vertex-colored terrain mesh. A
 * single perspective camera is locked to the vehicle's position and attitude so
 * the view banks and pitches with the airframe (Garmin SVT style).
 *
 * Sky/fog/lighting mirror the in-app simulator world (sim-world-scene) for a
 * consistent look. Coordinate + euler conventions also match it:
 *   three.x = East, three.y = Up (elevation), three.z = -North
 *   camera euler order 'YXZ': y = -heading, x = pitch, z = -roll  (radians)
 */

import * as THREE from 'three';
import type { DrapeRing } from './svt-satellite';
import type { SceneMeshEntry } from '../../../../shared/tile-source-types';
import { loadGltfScene, releaseGltf } from '../../../three/gltf-cache';
import { useConsoleStore } from '../../../stores/console-store';
import { modelKeyForClass, type ModelKey } from '../../sim/sim-models';
import { attachDrape } from './drape-material';
import {
  createGoogle3dTilesController,
  type Google3dTilesStatus,
} from './google-3d-tiles';
import {
  type ElevationGrid,
  buildTerrainGridGeometry,
  lonLatToLocal,
  sampleElevation,
  SVT_HALF_SIZE_M,
  SVT_REBUILD_DISTANCE_M,
} from './svt-terrain';

const DEG = Math.PI / 180;

// Floor under the camera, so a landed vehicle does not sit exactly ON the
// surface (the terrain would collapse to an edge-on sliver). It is a RENDER
// floor only: the true clearance is reported separately, or the display would
// quietly hide the aircraft being below the terrain it is drawing.
const MIN_EYE_M = 2;

// Subtle atmospheric haze near the far horizon only. Terrain reads crisply for
// most of the patch and fades into the sky just before its nearest possible
// edge (half-size minus how far the vehicle can drift from the patch centre),
// so the square boundary is never a visible wall.
const FOG_END = SVT_HALF_SIZE_M - SVT_REBUILD_DISTANCE_M - 1_000;
const FOG_START = Math.max(8_000, FOG_END - 9_000);

export interface SvtPose {
  lat: number;
  lon: number;
  /** Height above terrain at the vehicle (metres). */
  agl: number;
  /** Altitude above mean sea level (metres), as the vehicle reports it. */
  altMsl: number;
  /** Eye altitude in the DEM's own datum: the vehicle's MSL shifted by the
      measured offset between its altitude reference and the terrain data.
      This, not AGL, is what the camera flies at, so rising ground rises in
      the view exactly as it does in the world. */
  eyeMsl: number;
  /** Render floor above the terrain. A rover's camera sits under a metre, an
      aircraft's is held clear of the surface so the ground never goes edge-on. */
  eyeFloorM?: number;
  rollDeg: number;
  pitchDeg: number;
  /** True heading, degrees (0 = North, 90 = East). */
  headingDeg: number;
}

export interface SvtPeerVehicle {
  id: string;
  label: string;
  color?: string;
  vehicleClass?: string;
  lat: number;
  lon: number;
  altMsl: number;
  agl: number;
  rollDeg: number;
  pitchDeg: number;
  headingDeg: number;
  armed?: boolean;
}

interface PeerModelDef {
  url: string;
  yaw: number;
  spinAxis: 'x' | 'y' | 'z';
}

const PEER_MODEL_DEFS: Record<ModelKey, PeerModelDef> = {
  quad: { url: new URL('../../../assets/models/quadrotor-split.glb', import.meta.url).href, yaw: 0, spinAxis: 'y' },
  plane: { url: new URL('../../../assets/models/plane.glb', import.meta.url).href, yaw: Math.PI, spinAxis: 'z' },
  hexa: { url: new URL('../../../assets/models/hexacopter.glb', import.meta.url).href, yaw: 0, spinAxis: 'y' },
  rover: { url: new URL('../../../assets/models/rover.glb', import.meta.url).href, yaw: Math.PI / 2, spinAxis: 'y' },
};

const PEER_TARGET_SPAN = 4.0;
const PEER_PROP_SPIN = 1.1;

export interface SvtScene {
  resize: (width: number, height: number) => void;
  /** Swap in a freshly built terrain mesh; disposes the previous one. */
  setTerrain: (geometry: THREE.BufferGeometry, grid: ElevationGrid) => void;
  /** Module GLB meshes placed in the terrain grid's frame; entries not listed are removed. */
  setMeshes: (entries: SceneMeshEntry[]) => void;
  /** Other connected SITL / fleet vehicles rendered as 3D airframe GLB models in the FPV scene. */
  setPeerVehicles: (peers: SvtPeerVehicle[], datumOffsetM?: number) => void;
  /** Drape satellite imagery over the terrain; null returns to the elevation
      ramp. Disposes the textures of any previous rings. */
  setDrape: (rings: DrapeRing[] | null) => void;
  /** Swap in a building mesh (null removes it); disposes the previous one. */
  setBuildings: (geometry: THREE.BufferGeometry | null) => void;
  /** Roofs from the satellite drape, and a window pattern on the walls. */
  setBuildingStyle: (style: { roofImagery: boolean; facades: boolean }) => void;
  /** Enable/disable and configure Google Map Tiles API Photorealistic 3D Tiles. */
  setGoogle3dTiles: (opts: {
    enabled: boolean;
    apiKey: string;
    maxScreenSpaceError?: number;
    onStatusChange?: (status: Google3dTilesStatus, message?: string) => void;
    onAttributionChange?: (attribution: string) => void;
  }) => void;
  /** Position + orient the camera from a vehicle pose. */
  setPose: (pose: SvtPose) => void;
  /** Metres between the eye and the terrain directly below it, from the last
      setPose. Negative means the aircraft is below the DEM surface. NaN until
      terrain exists. */
  getClearanceM: () => number;
  /** The perspective camera's vertical FOV (degrees). The HUD world overlay uses
      the SAME fov so its world-locked symbology aligns with the SVT terrain. */
  getFov: () => number;
  hasTerrain: () => boolean;
  render: () => void;
  dispose: () => void;
}

export function createSvtScene(canvas: HTMLCanvasElement): SvtScene {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  // Without preventDefault a lost context is fatal to the whole renderer
  // process; swallowing it leaves a frozen view that recovers on restore.
  let contextLost = false;
  const onLost = (e: Event) => { e.preventDefault(); contextLost = true; };
  const onRestored = () => { contextLost = false; };
  canvas.addEventListener('webglcontextlost', onLost, false);
  canvas.addEventListener('webglcontextrestored', onRestored, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  const HORIZON = new THREE.Color(0xbcd4ec);
  scene.background = HORIZON.clone();
  // Haze fully hides the square terrain boundary before the vehicle can reach it.
  scene.fog = new THREE.Fog(HORIZON.getHex(), FOG_START, FOG_END);

  // Far plane clears the patch corners AND the camera-centred sky dome below.
  const camera = new THREE.PerspectiveCamera(62, 1, 1, SVT_HALF_SIZE_M * 2);
  camera.rotation.order = 'YXZ';

  // ─── Lighting (matches the sim world) ────────────────────────────────────
  const hemi = new THREE.HemisphereLight(0xcfe3ff, 0x4a5d3f, 1.0);
  const sun = new THREE.DirectionalLight(0xfff6e6, 1.4);
  sun.position.set(120, 200, 80);
  const fill = new THREE.AmbientLight(0xffffff, 0.25);
  scene.add(hemi, sun, fill);

  // ─── Sky dome ─────────────────────────────────────────────────────────────
  // Radius stays inside `far`; the dome is re-centred on the camera each frame
  // (see setPose) so it never crosses the far plane and clips into a "hemisphere".
  const skyGeom = new THREE.SphereGeometry(SVT_HALF_SIZE_M * 1.6, 32, 16);
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
    uniforms: {
      topColor: { value: new THREE.Color(0x3b74c4) },
      bottomColor: { value: HORIZON.clone() },
      sunDir: { value: sun.position.clone().normalize() },
      sunColor: { value: new THREE.Color(0xfff3da) },
    },
    vertexShader: `
      varying vec3 vDir;
      void main() {
        vDir = normalize((modelMatrix * vec4(position, 1.0)).xyz);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 topColor;
      uniform vec3 bottomColor;
      uniform vec3 sunDir;
      uniform vec3 sunColor;
      varying vec3 vDir;
      void main() {
        float h = clamp(vDir.y * 0.5 + 0.5, 0.0, 1.0);
        vec3 sky = mix(bottomColor, topColor, pow(h, 0.8));
        float s = max(dot(normalize(vDir), normalize(sunDir)), 0.0);
        sky += sunColor * pow(s, 350.0) * 1.2;
        sky += sunColor * pow(s, 6.0) * 0.10;
        gl_FragColor = vec4(sky, 1.0);
      }
    `,
  });
  const sky = new THREE.Mesh(skyGeom, skyMat);
  // No depth interaction — it's purely the backdrop; terrain always draws over it.
  sky.renderOrder = -1;
  scene.add(sky);

  // ─── Replaceable terrain mesh ─────────────────────────────────────────────
  // DoubleSide on purpose: with FrontSide a camera that dips below the surface
  // sees straight through it into the sky dome, which reads as the world
  // vanishing rather than as a clipped view.
  const terrainMat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 1,
    metalness: 0,
    side: THREE.DoubleSide,
  });

  const drape = attachDrape(terrainMat);

  let terrainMesh: THREE.Mesh | null = null;
  let buildingsMesh: THREE.Mesh | null = null;
  // both sides lit: footprint winding varies between OSM sources
  const buildingsMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  const buildingStyle = { uRoofImagery: { value: 1 }, uFacades: { value: 1 } };
  drape.share(buildingsMat, 'vSvtRoof > 0.5 && uRoofImagery > 0.5', (shader) => {
    Object.assign(shader.uniforms, buildingStyle);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float aRoof;\nattribute vec2 aWall;\nvarying float vSvtRoof;\nvarying vec2 vSvtWall;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvSvtRoof = aRoof;\nvSvtWall = aWall;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vSvtRoof;\nvarying vec2 vSvtWall;\nuniform float uRoofImagery;\nuniform float uFacades;')
      .replace(
        '#include <color_fragment>',
        `#include <color_fragment>
        // windows: one row per 3 m floor, one column per 3.5 m of wall, none in the top half metre
        if (uFacades > 0.5 && vSvtRoof < 0.5 && vSvtWall.y > 0.8) {
          vec2 cell = fract(vSvtWall / vec2(3.5, 3.0));
          float win = step(0.22, cell.x) * step(cell.x, 0.78) * step(0.32, cell.y) * step(cell.y, 0.78);
          diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.18, 0.22, 0.28), win * 0.75);
        }`,
      );
  });
  let grid: ElevationGrid | null = null;
  const meshGroups = new Map<string, THREE.Group>();
  const meshUrls = new Map<string, { url: string; version: number }>();
  let lastMeshes: SceneMeshEntry[] = [];
  // A cargo mesh that fails to load says so once in the in-app console.
  const loggedFailures = new Set<string>();
  const placeMeshes = (entries: SceneMeshEntry[]) => {
    lastMeshes = entries;
    const wanted = new Set(entries.map((e) => `${e.key}@${e.version ?? 0}`));
    for (const [k, g] of meshGroups) {
      if (!wanted.has(k)) {
        scene.remove(g);
        meshGroups.delete(k);
        const gone = meshUrls.get(k);
        meshUrls.delete(k);
        if (gone) void releaseGltf(gone.url, gone.version);
      }
    }
    for (const e of entries) meshUrls.set(`${e.key}@${e.version ?? 0}`, { url: e.url, version: e.version ?? 0 });
    if (!grid) return;
    for (const e of entries) {
      const k = `${e.key}@${e.version ?? 0}`;
      const place = (g: THREE.Group) => {
        const { x, z } = lonLatToLocal(grid!, e.origin.lat, e.origin.lng);
        const groundY = sampleElevation(grid!, e.origin.lat, e.origin.lng);
        const y = Number.isFinite(groundY) ? Math.max(e.origin.altMsl, groundY) : e.origin.altMsl;
        g.position.set(x, y, z);
        g.rotation.y = -((e.yawDeg ?? 0) * Math.PI) / 180;
        g.visible = !google3d.hasLoadedTiles();
      };
      const existing = meshGroups.get(k);
      if (existing) {
        place(existing);
        continue;
      }
      void loadGltfScene(e.url, e.version ?? 0)
        .then((g) => {
          if (!wanted.has(k) || meshGroups.has(k) || !grid) return;
          place(g);
          meshGroups.set(k, g);
          scene.add(g);
          if (!contextLost) renderer.render(scene, camera);
        })
        .catch((err: unknown) => {
          // A bad file is the module's problem; the scene stays usable, but say why.
          if (loggedFailures.has(k)) return;
          loggedFailures.add(k);
          const reason = err instanceof Error ? err.message : String(err);
          useConsoleStore.getState().addLog({ id: Date.now(), timestamp: Date.now(), level: 'warn', message: `[SceneMesh] could not load ${e.url}: ${reason}` }); // i18n-exempt: diagnostic log
        });
    }
  };

  // ─── Peer SITL / Fleet Vehicles (3D airframe GLB models in FPV) ───────────
  interface PeerTemplate {
    group: THREE.Group;
    spinAxis: 'x' | 'y' | 'z';
    topY: number;
  }
  interface PeerVehicleObj {
    id: string;
    group: THREE.Group;
    frame: THREE.Group;
    modelKey: ModelKey;
    usesModel: boolean;
    spinAxis: 'x' | 'y' | 'z';
    props: THREE.Object3D[];
    bodyMat: THREE.MeshStandardMaterial;
    ledMat: THREE.MeshStandardMaterial;
    labelSprite: THREE.Sprite;
    labelCanvas: HTMLCanvasElement;
    labelTexture: THREE.CanvasTexture;
    labelMat: THREE.SpriteMaterial;
    labelText: string;
    colorHex: string;
    lastDistM: number;
    armed: boolean;
  }

  const peerTemplates = new Map<ModelKey, PeerTemplate>();
  const peerLoadingModels = new Set<ModelKey>();
  const peerVehicles = new Map<string, PeerVehicleObj>();
  let lastPeers: SvtPeerVehicle[] = [];
  let lastPeerDatumOffsetM = 0;

  const peerBodyGeom = new THREE.BoxGeometry(1.1, 0.38, 1.3);
  const peerArmGeom = new THREE.BoxGeometry(3.0, 0.14, 0.28);
  const peerPropGeom = new THREE.CylinderGeometry(0.85, 0.85, 0.04, 16);
  const peerLedGeom = new THREE.SphereGeometry(0.22, 12, 10);
  const peerDarkMat = new THREE.MeshStandardMaterial({ color: 0x222831, roughness: 0.6, metalness: 0.3 });
  const peerPropMat = new THREE.MeshStandardMaterial({ color: 0x0f1115, roughness: 0.4, transparent: true, opacity: 0.55 });
  const peerSharedDisposables: Array<{ dispose: () => void }> = [
    peerBodyGeom, peerArmGeom, peerPropGeom, peerLedGeom, peerDarkMat, peerPropMat,
  ];

  const drawPeerLabel = (
    canvas: HTMLCanvasElement,
    tex: THREE.CanvasTexture,
    label: string,
    distM: number,
    colorHex: string,
  ) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const text = distM > 0 ? `${label} · ${distM}m` : label;
    ctx.font = 'bold 28px system-ui, -apple-system, sans-serif';
    const textW = Math.min(w - 24, Math.max(110, ctx.measureText(text).width + 32));
    const pillX = (w - textW) / 2;
    const pillY = 10;
    const pillH = 44;
    const r = 10;

    ctx.fillStyle = 'rgba(15, 23, 42, 0.82)';
    ctx.strokeStyle = colorHex;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(pillX + r, pillY);
    ctx.lineTo(pillX + textW - r, pillY);
    ctx.quadraticCurveTo(pillX + textW, pillY, pillX + textW, pillY + r);
    ctx.lineTo(pillX + textW, pillY + pillH - r);
    ctx.quadraticCurveTo(pillX + textW, pillY + pillH, pillX + textW - r, pillY + pillH);
    ctx.lineTo(pillX + r, pillY + pillH);
    ctx.quadraticCurveTo(pillX, pillY + pillH, pillX, pillY + pillH - r);
    ctx.lineTo(pillX, pillY + r);
    ctx.quadraticCurveTo(pillX, pillY, pillX + r, pillY);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#f8fafc';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, w / 2, pillY + pillH / 2 + 1);

    // Subtle target pointer / diamond below the pill pointing to the airframe
    const cx = w / 2;
    const cy = pillY + pillH + 18;
    ctx.strokeStyle = colorHex;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy - 10);
    ctx.lineTo(cx + 10, cy);
    ctx.lineTo(cx, cy + 10);
    ctx.lineTo(cx - 10, cy);
    ctx.closePath();
    ctx.stroke();

    tex.needsUpdate = true;
  };

  const buildPeerProceduralFrame = (
    frame: THREE.Group,
    bodyMat: THREE.MeshStandardMaterial,
    ledMat: THREE.MeshStandardMaterial,
    props: THREE.Object3D[],
  ) => {
    frame.add(new THREE.Mesh(peerBodyGeom, bodyMat));
    for (const ang of [Math.PI / 4, -Math.PI / 4]) {
      const arm = new THREE.Mesh(peerArmGeom, peerDarkMat);
      arm.rotation.y = ang;
      frame.add(arm);
    }
    const r = 1.05;
    for (const [sx, sz, front] of [[1, -1, true], [-1, -1, true], [1, 1, false], [-1, 1, false]] as Array<[number, number, boolean]>) {
      const prop = new THREE.Mesh(peerPropGeom, front ? ledMat : peerPropMat);
      prop.position.set(sx * r, 0.25, sz * r);
      props.push(prop);
      frame.add(prop);
    }
    const led = new THREE.Mesh(peerLedGeom, ledMat);
    led.position.set(0, 0.45, 0);
    frame.add(led);
  };

  const installPeerTemplate = (v: PeerVehicleObj, tpl: PeerTemplate) => {
    for (let i = v.frame.children.length - 1; i >= 0; i--) v.frame.remove(v.frame.children[i]!);
    v.props.length = 0;
    const model = tpl.group.clone(true);
    v.frame.add(model);
    model.traverse((o) => {
      if (!/^prop_/i.test(o.name)) return;
      for (let p = o.parent; p && p !== model; p = p.parent) {
        if (/^prop_/i.test(p.name)) return;
      }
      v.props.push(o);
    });
    const led = new THREE.Mesh(peerLedGeom, v.ledMat);
    led.position.set(0, tpl.topY + 0.25, 0);
    v.frame.add(led);
    v.usesModel = true;
    v.spinAxis = tpl.spinAxis;
  };

  const ensurePeerModelLoaded = (key: ModelKey) => {
    if (peerTemplates.has(key) || peerLoadingModels.has(key)) return;
    peerLoadingModels.add(key);
    const def = PEER_MODEL_DEFS[key];
    void loadGltfScene(def.url, 0)
      .then((rawModel) => {
        const wrapper = new THREE.Group();
        rawModel.rotation.y = def.yaw;
        rawModel.updateMatrixWorld(true);
        const box = new THREE.Box3().setFromObject(rawModel);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const s = PEER_TARGET_SPAN / Math.max(size.x, size.z, 1e-3);
        rawModel.scale.setScalar(s);
        rawModel.position.set(-center.x * s, -center.y * s, -center.z * s);
        wrapper.add(rawModel);
        const tpl: PeerTemplate = { group: wrapper, spinAxis: def.spinAxis, topY: (size.y * s) * 0.5 };
        peerTemplates.set(key, tpl);
        for (const v of peerVehicles.values()) {
          if (v.modelKey === key && !v.usesModel) {
            installPeerTemplate(v, tpl);
          }
        }
        if (!contextLost) renderer.render(scene, camera);
      })
      .catch(() => {
        // Procedural fallback stays active if GLB cannot load
      });
  };

  const disposePeerVehicle = (v: PeerVehicleObj) => {
    scene.remove(v.group, v.labelSprite);
    v.bodyMat.dispose();
    v.ledMat.dispose();
    v.labelMat.dispose();
    v.labelTexture.dispose();
  };

  const placePeerVehicles = (peers: SvtPeerVehicle[], datumOffsetM: number) => {
    lastPeers = peers;
    lastPeerDatumOffsetM = datumOffsetM;
    const wanted = new Set(peers.map((p) => p.id));
    for (const [id, v] of peerVehicles) {
      if (!wanted.has(id)) {
        disposePeerVehicle(v);
        peerVehicles.delete(id);
      }
    }
    if (!grid || peers.length === 0) return;

    for (const p of peers) {
      const colorHex = p.color ?? '#38bdf8';
      const wantKey = modelKeyForClass(p.vehicleClass);
      ensurePeerModelLoaded(wantKey);
      let v = peerVehicles.get(p.id);
      if (!v) {
        const group = new THREE.Group();
        group.rotation.order = 'YXZ';
        const frame = new THREE.Group();
        group.add(frame);

        const identity = new THREE.Color(colorHex);
        const accent = p.armed ? 0xf97316 : 0x38bdf8;
        const bodyMat = new THREE.MeshStandardMaterial({
          color: identity.clone().multiplyScalar(0.8),
          emissive: identity.clone().multiplyScalar(0.45),
          emissiveIntensity: 0.7,
          roughness: 0.45,
          metalness: 0.3,
        });
        const ledMat = new THREE.MeshStandardMaterial({
          color: accent,
          emissive: accent,
          emissiveIntensity: 0.95,
        });
        const props: THREE.Object3D[] = [];

        const labelCanvas = document.createElement('canvas');
        labelCanvas.width = 320;
        labelCanvas.height = 96;
        const labelTexture = new THREE.CanvasTexture(labelCanvas);
        labelTexture.colorSpace = THREE.SRGBColorSpace;
        const labelMat = new THREE.SpriteMaterial({
          map: labelTexture,
          transparent: true,
          depthTest: false,
          depthWrite: false,
        });
        const labelSprite = new THREE.Sprite(labelMat);
        labelSprite.renderOrder = 999;

        v = {
          id: p.id,
          group,
          frame,
          modelKey: wantKey,
          usesModel: false,
          spinAxis: 'y',
          props,
          bodyMat,
          ledMat,
          labelSprite,
          labelCanvas,
          labelTexture,
          labelMat,
          labelText: p.label,
          colorHex,
          lastDistM: -1,
          armed: !!p.armed,
        };

        const tpl = peerTemplates.get(wantKey);
        if (tpl) {
          installPeerTemplate(v, tpl);
        } else {
          buildPeerProceduralFrame(frame, bodyMat, ledMat, props);
        }

        drawPeerLabel(labelCanvas, labelTexture, p.label, 0, colorHex);
        scene.add(group, labelSprite);
        peerVehicles.set(p.id, v);
      } else {
        if (v.modelKey !== wantKey) {
          v.modelKey = wantKey;
          v.usesModel = false;
          const tpl = peerTemplates.get(wantKey);
          if (tpl) installPeerTemplate(v, tpl);
        }
        if (!!p.armed !== v.armed) {
          v.armed = !!p.armed;
          const accent = v.armed ? 0xf97316 : 0x38bdf8;
          v.ledMat.color.setHex(accent);
          v.ledMat.emissive.setHex(accent);
        }
        if (v.colorHex !== colorHex || v.labelText !== p.label) {
          v.colorHex = colorHex;
          v.labelText = p.label;
          const identity = new THREE.Color(colorHex);
          v.bodyMat.color.copy(identity).multiplyScalar(0.8);
          v.bodyMat.emissive.copy(identity).multiplyScalar(0.45);
          v.lastDistM = -1;
        }
      }

      const { x, z } = lonLatToLocal(grid, p.lat, p.lon);
      const groundY = sampleElevation(grid, p.lat, p.lon);
      const rawY = p.altMsl !== 0
        ? p.altMsl + datumOffsetM
        : (Number.isFinite(groundY) ? groundY + p.agl : p.agl);
      const y = Number.isFinite(groundY) ? Math.max(rawY, groundY + 0.5) : rawY;

      v.group.position.set(x, y, z);
      v.group.rotation.set(p.pitchDeg * DEG, -p.headingDeg * DEG, -p.rollDeg * DEG);
      v.labelSprite.position.set(x, y + PEER_TARGET_SPAN * 0.75, z);
    }
  };

  // ─── Terrain-draped reference grid ────────────────────────────────────────
  // Rebuilt from each ElevationGrid in setTerrain so the grid lines ride the
  // hills (a flat plane grid just buries itself in the terrain). Gives the pilot
  // a topographic reference and the lines converge to the horizon = depth cue.
  const GRID_OPACITY = 0.3;
  // Over imagery the reference grid is a distraction, not a depth cue.
  const GRID_OPACITY_DRAPED = 0.12;
  const terrainGridMat = new THREE.LineBasicMaterial({ color: 0xbfe6c8, transparent: true, opacity: GRID_OPACITY, depthWrite: false });
  let terrainGrid: THREE.LineSegments | null = null;
  let clearanceM = NaN;
  let viewportHeight = 720;

  // ─── Google Map Tiles API Photorealistic 3D Tiles ─────────────────────────
  let g3dStatusCb: ((status: Google3dTilesStatus, message?: string) => void) | undefined;
  let g3dAttrCb: ((attribution: string) => void) | undefined;
  const syncTerrainBackdropWith3dTiles = () => {
    const has3d = google3d.hasLoadedTiles();
    if (buildingsMesh) buildingsMesh.visible = !has3d;
    if (terrainGrid) terrainGrid.visible = !has3d;
    if (terrainMesh) terrainMesh.position.y = has3d ? -28 : 0;
    for (const g of meshGroups.values()) g.visible = !has3d;
  };
  const google3d = createGoogle3dTilesController({
    onStatusChange: (st, msg) => {
      syncTerrainBackdropWith3dTiles();
      g3dStatusCb?.(st, msg);
    },
    onAttributionChange: (attr) => {
      g3dAttrCb?.(attr);
    },
    onDirty: () => {
      syncTerrainBackdropWith3dTiles();
      if (!contextLost) renderer.render(scene, camera);
    },
  });
  scene.add(google3d.group);


  return {
    resize(width: number, height: number) {
      viewportHeight = Math.max(1, height);
      renderer.setSize(width, height, false);
      camera.aspect = width / Math.max(1, height);
      camera.updateProjectionMatrix();
    },

    setTerrain(geometry: THREE.BufferGeometry, nextGrid: ElevationGrid) {
      if (terrainMesh) {
        scene.remove(terrainMesh);
        terrainMesh.geometry.dispose();
      }
      terrainMesh = new THREE.Mesh(geometry, terrainMat);
      terrainMesh.frustumCulled = false;
      scene.add(terrainMesh);
      grid = nextGrid;
      google3d.setOrigin(nextGrid.centerLat, nextGrid.centerLon, 0);

      // Rebuild the draped reference grid so it conforms to (rides) the new
      // terrain surface instead of hovering as a flat plane.
      if (terrainGrid) { scene.remove(terrainGrid); terrainGrid.geometry.dispose(); }
      terrainGrid = new THREE.LineSegments(buildTerrainGridGeometry(nextGrid), terrainGridMat);
      terrainGrid.frustumCulled = false;
      terrainGrid.renderOrder = 1;
      scene.add(terrainGrid);
      syncTerrainBackdropWith3dTiles();
      placeMeshes(lastMeshes);
      placePeerVehicles(lastPeers, lastPeerDatumOffsetM);
    },

    setMeshes(entries: SceneMeshEntry[]) {
      placeMeshes(entries);
    },

    setPeerVehicles(peers: SvtPeerVehicle[], datumOffsetM = 0) {
      placePeerVehicles(peers, datumOffsetM);
    },

    setBuildings(geometry: THREE.BufferGeometry | null) {
      if (buildingsMesh) {
        scene.remove(buildingsMesh);
        buildingsMesh.geometry.dispose();
        buildingsMesh = null;
      }
      if (!geometry) return;
      buildingsMesh = new THREE.Mesh(geometry, buildingsMat);
      buildingsMesh.visible = !google3d.hasLoadedTiles();
      scene.add(buildingsMesh);
    },
    setBuildingStyle(style) {
      buildingStyle.uRoofImagery.value = style.roofImagery ? 1 : 0;
      buildingStyle.uFacades.value = style.facades ? 1 : 0;
    },
    setGoogle3dTiles(opts) {
      g3dStatusCb = opts.onStatusChange;
      g3dAttrCb = opts.onAttributionChange;
      google3d.configure({
        enabled: opts.enabled,
        apiKey: opts.apiKey,
        maxScreenSpaceError: opts.maxScreenSpaceError,
      });
      syncTerrainBackdropWith3dTiles();
    },
    setDrape(rings: DrapeRing[] | null) {
      drape.setDrape(rings, renderer);
      terrainGridMat.opacity = drape.hasDrape() ? GRID_OPACITY_DRAPED : GRID_OPACITY;
    },

    setPose(pose: SvtPose) {
      if (!grid && (pose.lat !== 0 || pose.lon !== 0)) {
        google3d.setOrigin(pose.lat, pose.lon, 0);
      }
      const local = grid ? lonLatToLocal(grid, pose.lat, pose.lon) : { x: 0, z: 0 };
      const groundElev = grid ? sampleElevation(grid, pose.lat, pose.lon) : 0;
      clearanceM = grid ? pose.eyeMsl - groundElev : NaN;
      camera.position.set(local.x, Math.max(pose.eyeMsl, groundElev + (pose.eyeFloorM ?? MIN_EYE_M)), local.z);
      camera.rotation.set(pose.pitchDeg * DEG, -pose.headingDeg * DEG, -pose.rollDeg * DEG);
      // Keep the sky dome centred on the camera so it never crosses the far plane.
      sky.position.copy(camera.position);
      google3d.updateForCamera(camera, viewportHeight);
      syncTerrainBackdropWith3dTiles();
    },

    getFov() {
      return camera.fov;
    },

    getClearanceM() {
      return clearanceM;
    },

    hasTerrain() {
      return terrainMesh !== null;
    },

    render() {
      if (contextLost) return;
      for (const v of peerVehicles.values()) {
        if (v.armed) {
          for (let i = 0; i < v.props.length; i++) {
            v.props[i]!.rotation[v.spinAxis] += i % 2 === 0 ? PEER_PROP_SPIN : -PEER_PROP_SPIN;
          }
        }
        const dist = camera.position.distanceTo(v.group.position);
        const distM = Math.max(1, Math.round(dist));
        if (distM !== v.lastDistM) {
          v.lastDistM = distM;
          drawPeerLabel(v.labelCanvas, v.labelTexture, v.labelText, distM, v.colorHex);
        }
        const s = Math.min(90, Math.max(3.2, dist * 0.14));
        v.labelSprite.scale.set(s, s * 0.3, 1);
        v.labelSprite.position.y = v.group.position.y + PEER_TARGET_SPAN * 0.65 + s * 0.12;
      }
      renderer.render(scene, camera);
    },

    dispose() {
      canvas.removeEventListener('webglcontextlost', onLost);
      canvas.removeEventListener('webglcontextrestored', onRestored);
      google3d.dispose();
      drape.dispose();
      for (const v of peerVehicles.values()) disposePeerVehicle(v);
      peerVehicles.clear();
      for (const d of peerSharedDisposables) d.dispose();
      if (terrainMesh) {
        terrainMesh.geometry.dispose();
        terrainMesh = null;
      }
      terrainMat.dispose();
      buildingsMesh?.geometry.dispose();
      buildingsMat.dispose();
      if (terrainGrid) terrainGrid.geometry.dispose();
      terrainGridMat.dispose();
      skyGeom.dispose();
      skyMat.dispose();
      renderer.dispose();
    },
  };
}
