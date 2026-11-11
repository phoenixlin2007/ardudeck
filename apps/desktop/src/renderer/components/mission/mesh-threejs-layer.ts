import maplibregl from 'maplibre-gl';
import * as THREE from 'three';
import type { SceneMeshEntry } from '../../../shared/tile-source-types';
import { loadGltfScene, releaseGltf } from '../../three/gltf-cache';

interface Ref { x: number; y: number; z: number; s: number }

/** Local metres from the reference: x east, y up (times exaggeration), z south. */
export function meshLocalPlacement(origin: { lat: number; lng: number; altMsl: number }, ref: Ref, yawDeg: number, exaggeration: number): { x: number; y: number; z: number; rotY: number } {
  const mc = maplibregl.MercatorCoordinate.fromLngLat([origin.lng, origin.lat], 0);
  return { x: (mc.x - ref.x) / ref.s, y: origin.altMsl * exaggeration, z: (mc.y - ref.y) / ref.s, rotY: -(yawDeg * Math.PI) / 180 };
}

export interface MeshThreeJsLayer {
  layer: maplibregl.CustomLayerInterface;
  setMeshes(entries: SceneMeshEntry[], exaggeration: number): void;
  dispose(): void;
}

/** Same frame as mission-threejs-layer: Translation * Scale(s, -s, s) * RotateX(pi/2). */
export function createMeshThreeJsLayer(): MeshThreeJsLayer {
  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x4a5d3f, 0.9));
  const sun = new THREE.DirectionalLight(0xfff6e6, 1.3);
  sun.position.set(120, 220, 90);
  scene.add(sun);
  const camera = new THREE.Camera();
  let renderer: THREE.WebGLRenderer | null = null;
  let map: maplibregl.Map | null = null;
  let ref: Ref | null = null;
  const groups = new Map<string, THREE.Group>();
  const urls = new Map<string, { url: string; version: number }>();
  const rotationX = new THREE.Matrix4().makeRotationAxis(new THREE.Vector3(1, 0, 0), Math.PI / 2);

  const layer: maplibregl.CustomLayerInterface = {
    id: 'module-scene-meshes',
    type: 'custom',
    renderingMode: '3d',
    onAdd(m, gl) {
      map = m;
      renderer = new THREE.WebGLRenderer({ canvas: m.getCanvas(), context: gl, antialias: true });
      renderer.autoClear = false;
    },
    render(_gl, args) {
      if (!renderer || !map || !ref || groups.size === 0) return;
      const l = new THREE.Matrix4().makeTranslation(ref.x, ref.y, ref.z).scale(new THREE.Vector3(ref.s, -ref.s, ref.s)).multiply(rotationX);
      const main = (args as { defaultProjectionData: { mainMatrix: number[] } }).defaultProjectionData.mainMatrix;
      camera.projectionMatrix = new THREE.Matrix4().fromArray(main).multiply(l);
      renderer.resetState();
      renderer.render(scene, camera);
    },
  };

  return {
    layer,
    setMeshes(entries, exaggeration) {
      const wanted = new Set(entries.map((e) => `${e.key}@${e.version ?? 0}`));
      for (const [k, g] of groups) {
        if (!wanted.has(k)) {
          scene.remove(g);
          groups.delete(k);
          const gone = urls.get(k);
          urls.delete(k);
          if (gone) void releaseGltf(gone.url, gone.version);
        }
      }
      for (const e of entries) urls.set(`${e.key}@${e.version ?? 0}`, { url: e.url, version: e.version ?? 0 });
      if (entries.length === 0) {
        ref = null;
        map?.triggerRepaint();
        return;
      }
      const first = entries[0]!.origin;
      const mc = maplibregl.MercatorCoordinate.fromLngLat([first.lng, first.lat], 0);
      ref = { x: mc.x, y: mc.y, z: mc.z, s: mc.meterInMercatorCoordinateUnits() };
      for (const e of entries) {
        const k = `${e.key}@${e.version ?? 0}`;
        const place = (g: THREE.Group) => {
          const p = meshLocalPlacement(e.origin, ref!, e.yawDeg ?? 0, exaggeration);
          g.position.set(p.x, p.y, p.z);
          g.rotation.y = p.rotY;
          g.scale.set(1, exaggeration, 1);
        };
        const existing = groups.get(k);
        if (existing) {
          place(existing);
          continue;
        }
        void loadGltfScene(e.url, e.version ?? 0)
          .then((g) => {
            if (!wanted.has(k) || groups.has(k)) return;
            place(g);
            groups.set(k, g);
            scene.add(g);
            map?.triggerRepaint();
          })
          .catch((err: unknown) => {
            // A bad file is the module's problem; the scene stays usable, but say why.
            console.warn(`[SceneMesh] could not load ${e.url}:`, err);
          });
      }
      map?.triggerRepaint();
    },
    dispose() {
      for (const g of groups.values()) scene.remove(g);
      groups.clear();
      renderer?.dispose();
    },
  };
}
