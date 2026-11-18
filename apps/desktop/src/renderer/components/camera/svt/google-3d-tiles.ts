/**
 * Google Map Tiles API — Photorealistic 3D Tiles loader & LOD manager for Three.js.
 *
 * Implements an OGC 3D Tiles 1.1 hierarchical octree client tailored for
 * `https://tile.googleapis.com/v1/3dtiles/root.json?key=YOUR_API_KEY`:
 *  - Propagates Google's `session` token and `key` across child `.json` & `.glb` URIs.
 *  - Decodes `.glb` payloads with Three.js `GLTFLoader` + `DRACOLoader`.
 *  - Zero-vertex-loop 64-bit Relative-To-Center (RTC) matrix composition: keeps vertex
 *    buffers untouched on the GPU while cancelling large ECEF translations in 64-bit
 *    JavaScript `Matrix4` math before uploading `modelViewMatrix` to WebGL.
 *  - Performs camera frustum culling (`THREE.Frustum`) + Screen-Space Error (SSE) LOD
 *    selection and staggered `.glb` instantiation for smooth 60 FPS flight.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';

const DEG = Math.PI / 180;
const WGS84_A = 6378137.0;
const WGS84_F = 1.0 / 298.257223563;
const WGS84_E2 = 2 * WGS84_F - WGS84_F * WGS84_F;

const ROOT_ENDPOINT = 'https://tile.googleapis.com/v1/3dtiles/root.json';

// Shared Draco + GLTF loader instances across scenes
let sharedGltfLoader: GLTFLoader | null = null;
function getGltfLoader(): GLTFLoader {
  if (!sharedGltfLoader) {
    const draco = new DRACOLoader();
    const baseHref = typeof window !== 'undefined' ? window.location.href : 'http://localhost/';
    draco.setDecoderPath(new URL('./draco/', baseHref).href);
    draco.setWorkerLimit(2);
    sharedGltfLoader = new GLTFLoader();
    sharedGltfLoader.setDRACOLoader(draco);
  }
  return sharedGltfLoader;
}

/**
 * Convert WGS84 geodetic coordinates (latDeg, lonDeg, ellipsoidHeightM) to ECEF (X, Y, Z) metres.
 */
export function geodeticToEcef(latDeg: number, lonDeg: number, heightM: number): [number, number, number] {
  const lat = latDeg * DEG;
  const lon = lonDeg * DEG;
  const sinLat = Math.sin(lat);
  const cosLat = Math.cos(lat);
  const sinLon = Math.sin(lon);
  const cosLon = Math.cos(lon);
  const n = WGS84_A / Math.sqrt(1.0 - WGS84_E2 * sinLat * sinLat);
  const x = (n + heightM) * cosLat * cosLon;
  const y = (n + heightM) * cosLat * sinLon;
  const z = (n * (1.0 - WGS84_E2) + heightM) * sinLat;
  return [x, y, z];
}

/**
 * Estimate EGM96 geoid undulation N (metres: h_ellipsoid - H_MSL).
 * Uses a regional high-accuracy fit for Taiwan / East Asia and a low-degree
 * spherical harmonic approximation globally so 3D Tiles sit at true MSL height.
 */
export function egm96GeoidUndulationM(latDeg: number, lonDeg: number): number {
  if (latDeg >= 21.0 && latDeg <= 26.0 && lonDeg >= 119.0 && lonDeg <= 123.0) {
    const dLat = latDeg - 25.0375;
    const dLon = lonDeg - 121.5637;
    return 19.3 - 0.35 * dLat + 1.15 * dLon;
  }
  const phi = latDeg * DEG;
  const lam = lonDeg * DEG;
  const cosPhi = Math.cos(phi);
  const sinPhi = Math.sin(phi);
  const p20 = 0.5 * (3 * sinPhi * sinPhi - 1);
  const p22 = 3 * cosPhi * cosPhi;
  return (
    -4.5 * p20 +
    4.2 * p22 * Math.cos(2 * (lam - 75 * DEG)) -
    3.1 * Math.sin(2 * phi) * Math.sin(lam + 20 * DEG)
  );
}

interface RawBoundingVolume {
  box?: number[];
  sphere?: number[];
  region?: number[];
}

interface RawTileContent {
  uri?: string;
  url?: string;
}

interface RawTileNode {
  boundingVolume: RawBoundingVolume;
  geometricError: number;
  refine?: 'ADD' | 'REPLACE';
  transform?: number[];
  content?: RawTileContent;
  contents?: RawTileContent[];
  children?: RawTileNode[];
}

interface RawTilesetJson {
  asset?: { version?: string };
  geometricError?: number;
  root: RawTileNode;
}

interface PreparedMeshEntry {
  mesh: THREE.Mesh;
  /** 64-bit matrix mapping from the mesh's (RTC-centered) local space to WGS84 ECEF. */
  meshToEcef: THREE.Matrix4;
}

interface TileNode {
  id: string;
  centerEcef: [number, number, number];
  radiusM: number;
  geometricError: number;
  refine: 'ADD' | 'REPLACE';
  /** Accumulated 4x4 column-major transform from tileset root to this tile in ECEF. */
  worldTransform: THREE.Matrix4;
  contentUrl: string | null;
  contentKind: 'none' | 'json' | 'glb';
  children: TileNode[];
  state: 'unloaded' | 'loading' | 'loaded' | 'failed';
  group: THREE.Group | null;
  entries: PreparedMeshEntry[];
  lastVisitedFrame: number;
}

export type Google3dTilesStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface Google3dTilesController {
  /** Three.js Group containing all active 3D tiles; add this to your scene. */
  readonly group: THREE.Group;
  /** Configure whether enabled and which API key / quality to use. */
  configure: (opts: {
    enabled: boolean;
    apiKey: string;
    maxScreenSpaceError?: number;
    maxRangeM?: number;
  }) => void;
  /** Set the local scene's geographic origin and vertical offset from MSL (`y = H_MSL + mslOffsetY`). */
  setOrigin: (lat: number, lon: number, mslOffsetY?: number) => void;
  /** Update LOD selection for the current camera pose in local scene coordinates. */
  updateForCamera: (camera: THREE.PerspectiveCamera, viewportHeightPx: number) => void;
  /** True when enabled and at least one `.glb` tile mesh is currently visible. */
  hasLoadedTiles: () => boolean;
  /** Release all GPU geometries, materials, textures, and abort pending requests. */
  dispose: () => void;
}

export interface Google3dTilesCallbacks {
  onStatusChange?: (status: Google3dTilesStatus, message?: string) => void;
  onAttributionChange?: (attribution: string) => void;
  onDirty?: () => void;
}

// Standard 3D Tiles glTF Y-up -> ECEF Z-up rotation (+90° around X)
const GLTF_Y_UP_TO_ECEF_Z_UP = new THREE.Matrix4().makeRotationX(Math.PI / 2);

function computeBoundingSphere(
  bv: RawBoundingVolume,
  transform: THREE.Matrix4,
): { centerEcef: [number, number, number]; radiusM: number } {
  const vCenter = new THREE.Vector3();

  if (bv.box && bv.box.length >= 12) {
    vCenter.set(bv.box[0]!, bv.box[1]!, bv.box[2]!).applyMatrix4(transform);
    let rSq = 0;
    const te = transform.elements;
    for (let i = 3; i < 12; i += 3) {
      const bx = bv.box[i]!;
      const by = bv.box[i + 1]!;
      const bz = bv.box[i + 2]!;
      const sx = bx * te[0]! + by * te[4]! + bz * te[8]!;
      const sy = bx * te[1]! + by * te[5]! + bz * te[9]!;
      const sz = bx * te[2]! + by * te[6]! + bz * te[10]!;
      rSq += sx * sx + sy * sy + sz * sz;
    }
    return { centerEcef: [vCenter.x, vCenter.y, vCenter.z], radiusM: Math.sqrt(rSq) };
  }

  if (bv.sphere && bv.sphere.length >= 4) {
    vCenter.set(bv.sphere[0]!, bv.sphere[1]!, bv.sphere[2]!).applyMatrix4(transform);
    const maxScale = transform.getMaxScaleOnAxis();
    return { centerEcef: [vCenter.x, vCenter.y, vCenter.z], radiusM: bv.sphere[3]! * maxScale };
  }

  if (bv.region && bv.region.length >= 6) {
    const west = bv.region[0]!;
    const south = bv.region[1]!;
    const east = bv.region[2]!;
    const north = bv.region[3]!;
    const minH = bv.region[4]!;
    const maxH = bv.region[5]!;
    const midLatDeg = ((south + north) * 0.5) / DEG;
    const midLonDeg = ((west + east) * 0.5) / DEG;
    const midH = (minH + maxH) * 0.5;
    const centerEcef = geodeticToEcef(midLatDeg, midLonDeg, midH);
    const cornerEcef = geodeticToEcef(north / DEG, east / DEG, maxH);
    const radiusM = Math.hypot(
      cornerEcef[0] - centerEcef[0],
      cornerEcef[1] - centerEcef[1],
      cornerEcef[2] - centerEcef[2],
    );
    return { centerEcef, radiusM };
  }

  return { centerEcef: [0, 0, 0], radiusM: WGS84_A * 2 };
}

function resolveChildUrl(
  rawUri: string,
  baseUrl: string,
  apiKey: string,
  sessionToken: string | null,
): { url: string; session: string | null } {
  const u = new URL(rawUri, baseUrl);
  const foundSession = u.searchParams.get('session') ?? sessionToken;
  if (foundSession && !u.searchParams.has('session')) {
    u.searchParams.set('session', foundSession);
  }
  if (apiKey) {
    u.searchParams.set('key', apiKey);
  }
  return { url: u.toString(), session: foundSession };
}

function disposeObject3D(obj: THREE.Object3D): void {
  obj.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.geometry?.dispose();
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const m of mats) {
      if (!m) continue;
      for (const val of Object.values(m as unknown as Record<string, unknown>)) {
        if (val instanceof THREE.Texture) val.dispose();
      }
      m.dispose();
    }
  });
}

export function createGoogle3dTilesController(callbacks: Google3dTilesCallbacks = {}): Google3dTilesController {
  const rootGroup = new THREE.Group();
  rootGroup.name = 'GooglePhotorealistic3DTiles';

  let enabled = false;
  let apiKey = '';
  let maxSSE = 28;
  let maxRangeM = 2800;
  const maxConcurrent = 5;

  let originLat: number | null = null;
  let originLon: number | null = null;
  let mslOffsetY = 0;

  // ECEF <-> Local Three.js (East, Up, South) transformation state (64-bit float)
  let originEcef: [number, number, number] = [0, 0, 0];
  let eastVec: [number, number, number] = [1, 0, 0];
  let upVec: [number, number, number] = [0, 1, 0];
  let southVec: [number, number, number] = [0, 0, 1];
  const ecefToLocalMat = new THREE.Matrix4();

  let sessionToken: string | null = null;
  let rootNode: TileNode | null = null;
  let rootFetching = false;
  let rootFailed = false;
  let generation = 0;
  let frameCounter = 0;
  let activeRequests = 0;
  let loadedGlbCount = 0;
  let currentStatus: Google3dTilesStatus = 'idle';
  const attributions = new Set<string>();
  const allLoadedGlbNodes = new Set<TileNode>();
  let nodeSeq = 0;

  let lastCamUpdatePos = new THREE.Vector3(Infinity, Infinity, Infinity);
  let lastCamUpdateRot = new THREE.Euler();
  let lastTraversalTime = 0;
  let pendingTraversalTimer: ReturnType<typeof setTimeout> | null = null;
  let lastCameraRef: THREE.PerspectiveCamera | null = null;
  let lastViewportHeight = 720;

  // Reusable math objects for frustum culling
  const projScreenMatrix = new THREE.Matrix4();
  const frustum = new THREE.Frustum();
  const tmpSphere = new THREE.Sphere();

  const setStatus = (next: Google3dTilesStatus, msg?: string) => {
    if (currentStatus === next && !msg) return;
    currentStatus = next;
    callbacks.onStatusChange?.(next, msg);
  };

  const updateNodeLocalMatrices = (node: TileNode) => {
    for (const entry of node.entries) {
      entry.mesh.matrix.multiplyMatrices(ecefToLocalMat, entry.meshToEcef);
      entry.mesh.matrixWorldNeedsUpdate = true;
    }
  };

  const recomputeOriginBasis = () => {
    if (originLat === null || originLon === null) return;
    const geoidM = egm96GeoidUndulationM(originLat, originLon);
    const h0 = geoidM - mslOffsetY;
    originEcef = geodeticToEcef(originLat, originLon, h0);

    const phi = originLat * DEG;
    const lam = originLon * DEG;
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);
    const sinLam = Math.sin(lam);
    const cosLam = Math.cos(lam);

    eastVec = [-sinLam, cosLam, 0];
    upVec = [cosPhi * cosLam, cosPhi * sinLam, sinPhi];
    southVec = [sinPhi * cosLam, sinPhi * sinLam, -cosPhi];

    // 64-bit affine matrix mapping any ECEF point P -> Local (East, Up, South)
    const tx = -(eastVec[0] * originEcef[0] + eastVec[1] * originEcef[1] + eastVec[2] * originEcef[2]);
    const ty = -(upVec[0] * originEcef[0] + upVec[1] * originEcef[1] + upVec[2] * originEcef[2]);
    const tz = -(southVec[0] * originEcef[0] + southVec[1] * originEcef[1] + southVec[2] * originEcef[2]);

    ecefToLocalMat.set(
      eastVec[0], eastVec[1], eastVec[2], tx,
      upVec[0], upVec[1], upVec[2], ty,
      southVec[0], southVec[1], southVec[2], tz,
      0, 0, 0, 1,
    );

    for (const node of allLoadedGlbNodes) {
      updateNodeLocalMatrices(node);
    }
    callbacks.onDirty?.();
  };

  const localToEcef = (local: THREE.Vector3): [number, number, number] => {
    const x = originEcef[0] + eastVec[0] * local.x + upVec[0] * local.y + southVec[0] * local.z;
    const y = originEcef[1] + eastVec[1] * local.x + upVec[1] * local.y + southVec[1] * local.z;
    const z = originEcef[2] + eastVec[2] * local.x + upVec[2] * local.y + southVec[2] * local.z;
    return [x, y, z];
  };

  const ecefToLocalPoint = (ecef: [number, number, number], out: THREE.Vector3): THREE.Vector3 => {
    const dx = ecef[0] - originEcef[0];
    const dy = ecef[1] - originEcef[1];
    const dz = ecef[2] - originEcef[2];
    out.set(
      eastVec[0] * dx + eastVec[1] * dy + eastVec[2] * dz,
      upVec[0] * dx + upVec[1] * dy + upVec[2] * dz,
      southVec[0] * dx + southVec[1] * dy + southVec[2] * dz,
    );
    return out;
  };

  /**
   * Prepare a loaded glTF scene without expensive per-vertex matrix/normal loops.
   * Uses 64-bit Matrix4 multiplication to cancel Earth-radius ECEF translations.
   */
  const prepareGltfScene = (
    gltfScene: THREE.Group,
    tileWorldTransform: THREE.Matrix4,
    tileCenterEcef: [number, number, number],
  ): { wrapper: THREE.Group; entries: PreparedMeshEntry[] } => {
    gltfScene.updateMatrixWorld(true);

    const meshes: THREE.Mesh[] = [];
    gltfScene.traverse((obj) => {
      const m = obj as THREE.Mesh;
      if (m.isMesh) meshes.push(m);
    });

    // O(1) single-vertex orientation check (glTF Y-up vs ECEF Z-up)
    let needsYUpRotation = true;
    const firstMesh = meshes[0];
    if (firstMesh) {
      const posAttr = firstMesh.geometry.getAttribute('position') as THREE.BufferAttribute | undefined;
      if (posAttr && posAttr.count > 0) {
        const sampleWorld = new THREE.Vector3(posAttr.getX(0), posAttr.getY(0), posAttr.getZ(0)).applyMatrix4(
          firstMesh.matrixWorld,
        );
        const testYUp = sampleWorld.clone().applyMatrix4(GLTF_Y_UP_TO_ECEF_Z_UP).applyMatrix4(tileWorldTransform);
        const testDirect = sampleWorld.clone().applyMatrix4(tileWorldTransform);
        const distYUp = Math.hypot(
          testYUp.x - tileCenterEcef[0],
          testYUp.y - tileCenterEcef[1],
          testYUp.z - tileCenterEcef[2],
        );
        const distDirect = Math.hypot(
          testDirect.x - tileCenterEcef[0],
          testDirect.y - tileCenterEcef[1],
          testDirect.z - tileCenterEcef[2],
        );
        needsYUpRotation = distYUp <= distDirect;
      }
    }

    const assetCopyright = (gltfScene.userData as { copyright?: string } | undefined)?.copyright;
    if (assetCopyright && attributions.size < 6) {
      for (const part of String(assetCopyright).split(';')) {
        const trimmed = part.trim();
        if (trimmed) attributions.add(trimmed);
      }
      callbacks.onAttributionChange?.(Array.from(attributions).slice(0, 4).join(' · '));
    }

    const rotToEcef = new THREE.Matrix4().copy(tileWorldTransform);
    if (needsYUpRotation) {
      rotToEcef.multiply(GLTF_Y_UP_TO_ECEF_Z_UP);
    }

    const wrapper = new THREE.Group();
    const entries: PreparedMeshEntry[] = [];

    for (const mesh of meshes) {
      const geom = mesh.geometry;
      const posAttr = geom.getAttribute('position') as THREE.BufferAttribute | undefined;
      const meshToEcef = new THREE.Matrix4().multiplyMatrices(rotToEcef, mesh.matrixWorld);

      // Only if raw vertex buffer itself contains un-centered planetary coordinates (>100 km),
      // shift in-place by vertex[0] and fold translation into the 64-bit meshToEcef matrix.
      if (posAttr && posAttr.count > 0) {
        const cx = posAttr.getX(0);
        const cy = posAttr.getY(0);
        const cz = posAttr.getZ(0);
        if (cx * cx + cy * cy + cz * cz > 1e10) {
          const arr = posAttr.array as Float32Array;
          for (let i = 0; i < arr.length; i += 3) {
            arr[i]! -= cx;
            arr[i + 1]! -= cy;
            arr[i + 2]! -= cz;
          }
          posAttr.needsUpdate = true;
          meshToEcef.multiply(new THREE.Matrix4().makeTranslation(cx, cy, cz));
        }
      }

      // Convert material to lightweight unlit FrontSide MeshBasicMaterial without mipmap generation stalls
      const convertMat = (m: THREE.Material): THREE.Material => {
        const std = m as THREE.MeshStandardMaterial;
        if (std.map) {
          std.map.colorSpace = THREE.SRGBColorSpace;
          std.map.generateMipmaps = false;
          std.map.minFilter = THREE.LinearFilter;
          std.map.magFilter = THREE.LinearFilter;
          const basic = new THREE.MeshBasicMaterial({
            map: std.map,
            side: THREE.FrontSide,
            toneMapped: false,
          });
          std.dispose();
          return basic;
        }
        return m;
      };
      const convertedMat = Array.isArray(mesh.material)
        ? mesh.material.map(convertMat)
        : convertMat(mesh.material);

      const cleanMesh = new THREE.Mesh(geom, convertedMat);
      cleanMesh.matrixAutoUpdate = false;
      cleanMesh.matrix.multiplyMatrices(ecefToLocalMat, meshToEcef);
      cleanMesh.matrixWorldNeedsUpdate = true;
      cleanMesh.frustumCulled = true;

      wrapper.add(cleanMesh);
      entries.push({ mesh: cleanMesh, meshToEcef });
    }

    return { wrapper, entries };
  };

  const buildTileNode = (
    raw: RawTileNode,
    parentTransform: THREE.Matrix4,
    baseUrl: string,
    inheritedRefine: 'ADD' | 'REPLACE',
  ): TileNode => {
    const worldTransform = parentTransform.clone();
    if (raw.transform && raw.transform.length === 16) {
      const localMat = new THREE.Matrix4().fromArray(raw.transform);
      worldTransform.multiply(localMat);
    }

    const { centerEcef, radiusM } = computeBoundingSphere(raw.boundingVolume, worldTransform);
    const refine = raw.refine ?? inheritedRefine;
    const rawContent = raw.content ?? raw.contents?.[0];
    const rawUri = rawContent?.uri ?? rawContent?.url ?? null;

    let contentUrl: string | null = null;
    let contentKind: 'none' | 'json' | 'glb' = 'none';
    if (rawUri) {
      const resolved = resolveChildUrl(rawUri, baseUrl, apiKey, sessionToken);
      contentUrl = resolved.url;
      if (resolved.session) sessionToken = resolved.session;
      const pathname = new URL(contentUrl).pathname.toLowerCase();
      if (pathname.endsWith('.json')) contentKind = 'json';
      else if (pathname.endsWith('.glb') || pathname.endsWith('.gltf')) contentKind = 'glb';
    }

    const node: TileNode = {
      id: `t_${++nodeSeq}`,
      centerEcef,
      radiusM,
      geometricError: raw.geometricError ?? 0,
      refine,
      worldTransform,
      contentUrl,
      contentKind,
      children: [],
      state: contentKind === 'none' ? 'loaded' : 'unloaded',
      group: null,
      entries: [],
      lastVisitedFrame: 0,
    };

    if (raw.children && raw.children.length > 0) {
      node.children = raw.children.map((c) => buildTileNode(c, worldTransform, baseUrl, refine));
    }
    return node;
  };

  const scheduleRevisit = () => {
    if (pendingTraversalTimer !== null) return;
    pendingTraversalTimer = setTimeout(() => {
      pendingTraversalTimer = null;
      if (lastCameraRef && enabled) {
        traverseAndSelect(lastCameraRef, lastViewportHeight);
      }
    }, 110);
  };

  const loadNodeContent = (node: TileNode, gen: number) => {
    if (node.state !== 'unloaded' || !node.contentUrl) return;
    if (activeRequests >= maxConcurrent) return;

    node.state = 'loading';
    activeRequests++;

    const finalUrl = resolveChildUrl(node.contentUrl, ROOT_ENDPOINT, apiKey, sessionToken).url;

    if (node.contentKind === 'json') {
      void fetch(finalUrl)
        .then(async (res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return (await res.json()) as RawTilesetJson;
        })
        .then((json) => {
          if (gen !== generation) return;
          if (json.root) {
            const childRoot = buildTileNode(json.root, node.worldTransform, finalUrl, node.refine);
            node.children.push(childRoot);
          }
          node.state = 'loaded';
        })
        .catch(() => {
          if (gen !== generation) return;
          node.state = 'failed';
        })
        .finally(() => {
          if (gen === generation) {
            activeRequests = Math.max(0, activeRequests - 1);
            scheduleRevisit();
          }
        });
      return;
    }

    if (node.contentKind === 'glb') {
      const loader = getGltfLoader();
      loader.load(
        finalUrl,
        (gltf) => {
          if (gen !== generation) {
            disposeObject3D(gltf.scene);
            return;
          }
          const copyright = (gltf.parser?.json?.asset as { copyright?: string } | undefined)?.copyright;
          if (copyright) {
            (gltf.scene.userData as { copyright?: string }).copyright = copyright;
          }
          const { wrapper, entries } = prepareGltfScene(gltf.scene, node.worldTransform, node.centerEcef);
          node.group = wrapper;
          node.entries = entries;
          node.state = 'loaded';
          allLoadedGlbNodes.add(node);
          rootGroup.add(wrapper);
          loadedGlbCount++;
          activeRequests = Math.max(0, activeRequests - 1);
          setStatus('ready');
          callbacks.onDirty?.();
          scheduleRevisit();
        },
        undefined,
        () => {
          if (gen !== generation) return;
          node.state = 'failed';
          activeRequests = Math.max(0, activeRequests - 1);
          scheduleRevisit();
        },
      );
    }
  };

  const ensureRootLoaded = () => {
    if (!enabled || !apiKey || rootNode || rootFetching || rootFailed) return;
    rootFetching = true;
    const gen = generation;
    setStatus('loading');

    const rootUrl = `${ROOT_ENDPOINT}?key=${encodeURIComponent(apiKey)}`;
    void fetch(rootUrl)
      .then(async (res) => {
        if (!res.ok) {
          let detail = `HTTP ${res.status}`;
          try {
            const errJson = (await res.json()) as { error?: { message?: string } };
            if (errJson?.error?.message) detail = errJson.error.message;
          } catch {
            // ignore body parse failure
          }
          throw new Error(detail);
        }
        return (await res.json()) as RawTilesetJson;
      })
      .then((json) => {
        if (gen !== generation) return;
        rootFetching = false;
        rootFailed = false;
        rootNode = buildTileNode(json.root, new THREE.Matrix4(), rootUrl, 'REPLACE');
        scheduleRevisit();
      })
      .catch((err: unknown) => {
        if (gen !== generation) return;
        rootFetching = false;
        rootFailed = true;
        const msg = err instanceof Error ? err.message : String(err);
        setStatus('error', msg);
      });
  };

  const traverseAndSelect = (camera: THREE.PerspectiveCamera, viewportHeightPx: number) => {
    if (!enabled || !rootNode || originLat === null || originLon === null) return;
    frameCounter++;
    lastTraversalTime = performance.now();

    camera.updateMatrixWorld();
    projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    frustum.setFromProjectionMatrix(projScreenMatrix);

    const camEcef = localToEcef(camera.position);
    const fovRad = (camera.fov || 60) * DEG;
    const sseFactor = Math.max(240, viewportHeightPx) / (2.0 * Math.tan(fovRad * 0.5));

    for (const n of allLoadedGlbNodes) {
      if (n.group) n.group.visible = false;
    }

    const loadQueue: Array<{ node: TileNode; priority: number }> = [];

    const visit = (node: TileNode): boolean => {
      node.lastVisitedFrame = frameCounter;

      const distCamCenter = Math.hypot(
        node.centerEcef[0] - camEcef[0],
        node.centerEcef[1] - camEcef[1],
        node.centerEcef[2] - camEcef[2],
      );
      const distCamSurface = Math.max(0.0, distCamCenter - node.radiusM);

      // Distance culling around the camera only
      if (distCamSurface > maxRangeM) {
        return false;
      }

      // Frustum culling: for tiles not enveloping the camera (radius < 3000m and dist > 200m),
      // check if their bounding sphere intersects the camera frustum.
      if (node.radiusM < 3000 && distCamSurface > 200) {
        ecefToLocalPoint(node.centerEcef, tmpSphere.center);
        tmpSphere.radius = node.radiusM * 1.15;
        if (!frustum.intersectsSphere(tmpSphere)) {
          return false;
        }
      }

      const d = Math.max(1.0, distCamSurface);
      const sse = (node.geometricError * sseFactor) / d;
      const shouldRefine = sse > maxSSE;

      if (node.contentKind === 'json') {
        if (node.state === 'unloaded') {
          loadQueue.push({ node, priority: sse });
          return false;
        }
        if (node.state === 'loading' || node.state === 'failed') {
          return false;
        }
        let anyRendered = false;
        for (const child of node.children) {
          if (visit(child)) anyRendered = true;
        }
        return anyRendered;
      }

      // Skip downloading planetary/city-wide low-res GLB blobs when refining deeper
      const isCoarseAncestor = node.geometricError > 140 && node.children.length > 0;
      if (!isCoarseAncestor && node.contentKind === 'glb' && node.state === 'unloaded') {
        loadQueue.push({ node, priority: sse });
      }

      if (shouldRefine && node.children.length > 0) {
        let allVisibleChildrenReady = true;
        let anyChildRendered = false;

        for (const child of node.children) {
          const cDistCam = Math.max(
            0,
            Math.hypot(
              child.centerEcef[0] - camEcef[0],
              child.centerEcef[1] - camEcef[1],
              child.centerEcef[2] - camEcef[2],
            ) - child.radiusM,
          );
          if (cDistCam > maxRangeM) continue;

          if (child.radiusM < 3000 && cDistCam > 200) {
            ecefToLocalPoint(child.centerEcef, tmpSphere.center);
            tmpSphere.radius = child.radiusM * 1.15;
            if (!frustum.intersectsSphere(tmpSphere)) continue;
          }

          const childRendered = visit(child);
          if (childRendered) {
            anyChildRendered = true;
          } else {
            allVisibleChildrenReady = false;
          }
        }

        if (anyChildRendered && (allVisibleChildrenReady || !node.group)) {
          return true;
        }
      }

      if (node.group) {
        node.group.visible = true;
        return true;
      }
      return false;
    };

    visit(rootNode);

    if (loadQueue.length > 0 && activeRequests < maxConcurrent) {
      loadQueue.sort((a, b) => b.priority - a.priority);
      // Dispatch at most 3 new requests per traversal tick so GLB parsing spreads across frames
      let dispatched = 0;
      for (const item of loadQueue) {
        if (activeRequests >= maxConcurrent || dispatched >= 3) break;
        loadNodeContent(item.node, generation);
        dispatched++;
      }
    }

    // Tighter GPU cache eviction (max 95 GLB tiles in memory)
    if (allLoadedGlbNodes.size > 95) {
      const stale = Array.from(allLoadedGlbNodes)
        .filter((n) => n.lastVisitedFrame < frameCounter - 20 && (!n.group || !n.group.visible))
        .sort((a, b) => a.lastVisitedFrame - b.lastVisitedFrame);
      const toDrop = Math.min(stale.length, allLoadedGlbNodes.size - 70);
      for (let i = 0; i < toDrop; i++) {
        const victim = stale[i]!;
        if (victim.group) {
          rootGroup.remove(victim.group);
          disposeObject3D(victim.group);
          victim.group = null;
        }
        victim.entries = [];
        victim.state = 'unloaded';
        allLoadedGlbNodes.delete(victim);
        loadedGlbCount = Math.max(0, loadedGlbCount - 1);
      }
    }
    callbacks.onDirty?.();
  };

  const clearAllTiles = () => {
    generation++;
    if (pendingTraversalTimer !== null) {
      clearTimeout(pendingTraversalTimer);
      pendingTraversalTimer = null;
    }
    for (const n of allLoadedGlbNodes) {
      if (n.group) {
        rootGroup.remove(n.group);
        disposeObject3D(n.group);
        n.group = null;
      }
      n.entries = [];
    }
    allLoadedGlbNodes.clear();
    rootNode = null;
    rootFetching = false;
    rootFailed = false;
    sessionToken = null;
    activeRequests = 0;
    loadedGlbCount = 0;
    attributions.clear();
  };

  return {
    group: rootGroup,

    configure(opts) {
      const nextKey = opts.apiKey.trim();
      const keyChanged = nextKey !== apiKey;
      const enabledChanged = opts.enabled !== enabled;

      enabled = opts.enabled;
      apiKey = nextKey;
      if (opts.maxScreenSpaceError !== undefined) maxSSE = opts.maxScreenSpaceError;
      if (opts.maxRangeM !== undefined) maxRangeM = opts.maxRangeM;

      rootGroup.visible = enabled;

      if (!enabled) {
        clearAllTiles();
        setStatus('idle');
        callbacks.onDirty?.();
        return;
      }

      if (keyChanged) {
        clearAllTiles();
      }

      if (!apiKey) {
        setStatus('idle');
        return;
      }

      ensureRootLoaded();
      if ((enabledChanged || keyChanged) && lastCameraRef) {
        scheduleRevisit();
      }
    },

    setOrigin(lat: number, lon: number, nextMslOffsetY = 0) {
      if (
        originLat !== null &&
        originLon !== null &&
        Math.abs(lat - originLat) < 1e-7 &&
        Math.abs(lon - originLon) < 1e-7 &&
        Math.abs(nextMslOffsetY - mslOffsetY) < 1e-3
      ) {
        return;
      }
      originLat = lat;
      originLon = lon;
      mslOffsetY = nextMslOffsetY;
      recomputeOriginBasis();
      if (enabled && apiKey) {
        ensureRootLoaded();
        scheduleRevisit();
      }
    },

    updateForCamera(camera: THREE.PerspectiveCamera, viewportHeightPx: number) {
      lastCameraRef = camera;
      lastViewportHeight = viewportHeightPx;
      if (!enabled || !apiKey || originLat === null || originLon === null) return;
      ensureRootLoaded();
      if (!rootNode) return;

      const now = performance.now();
      // Throttle LOD tree traversal to at most ~10 Hz (every 100 ms) during continuous flight
      if (now - lastTraversalTime < 100) return;

      const movedDistSq = camera.position.distanceToSquared(lastCamUpdatePos);
      const rotDelta =
        Math.abs(camera.rotation.x - lastCamUpdateRot.x) +
        Math.abs(camera.rotation.y - lastCamUpdateRot.y) +
        Math.abs(camera.rotation.z - lastCamUpdateRot.z);

      // Re-evaluate LOD when the camera moves > 15 m or rotates > ~4 degrees
      if (movedDistSq > 225 || rotDelta > 0.07) {
        lastCamUpdatePos.copy(camera.position);
        lastCamUpdateRot.copy(camera.rotation);
        traverseAndSelect(camera, viewportHeightPx);
      }
    },

    hasLoadedTiles() {
      return enabled && loadedGlbCount > 0;
    },

    dispose() {
      clearAllTiles();
    },
  };
}
