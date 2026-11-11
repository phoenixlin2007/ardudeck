import { app } from 'electron';
import { existsSync, readFileSync, statSync, watch } from 'node:fs';
import { dirname, isAbsolute, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { registerSceneMesh, unregisterSceneMeshesFor } from './scene-mesh-registry.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const LOCAL_MODELS_SLUG = 'local-landmarks';

export interface LocalModelConfigEntry {
  id: string;
  name: string;
  /** Filename inside `apps/desktop/resources/models/` or an absolute path to a `.glb` file. */
  file: string;
  enabled?: boolean;
  origin: {
    lat: number;
    lng: number;
    altMsl: number;
  };
  yawDeg?: number;
  scenes?: Array<'svt' | 'mission3d'>;
}

function getModelsDir(): string {
  const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
  return isDev
    ? join(__dirname, '../../resources/models')
    : join(app.getAppPath(), 'resources/models');
}

let reloadVersion = 1;

export function reloadCustomModels(): void {
  const modelsDir = getModelsDir();
  const configPath = join(modelsDir, 'models.json');
  unregisterSceneMeshesFor(LOCAL_MODELS_SLUG);

  if (!existsSync(configPath)) return;

  try {
    const raw = readFileSync(configPath, 'utf-8');
    const parsed = JSON.parse(raw) as LocalModelConfigEntry[];
    if (!Array.isArray(parsed)) return;

    reloadVersion += 1;
    for (const item of parsed) {
      if (item.enabled === false) continue;
      if (!item.id || !item.file || !item.origin) continue;

      const glbPath = isAbsolute(item.file) ? item.file : join(modelsDir, item.file);
      if (!existsSync(glbPath)) {
        console.warn(`[CustomModels] GLB file not found for "${item.id}": ${glbPath}`);
        continue;
      }

      // Combine file mtime with reloadVersion so replacing a .glb file on disk invalidates the renderer GLTF cache
      let version = reloadVersion;
      try {
        version = Math.floor(statSync(glbPath).mtimeMs);
      } catch {
        // fallback to reloadVersion
      }

      registerSceneMesh(LOCAL_MODELS_SLUG, {
        id: item.id,
        name: item.name || item.id,
        glb: glbPath,
        origin: {
          lat: Number(item.origin.lat),
          lng: Number(item.origin.lng),
          altMsl: Number(item.origin.altMsl ?? 0),
        },
        yawDeg: Number(item.yawDeg ?? 0),
        scenes: item.scenes ?? ['svt', 'mission3d'],
        version,
      });
      console.log(
        `[CustomModels] Loaded "${item.name || item.id}" (${glbPath}) at (${item.origin.lat}, ${item.origin.lng}, ${item.origin.altMsl}m)`,
      );
    }
  } catch (err) {
    console.error('[CustomModels] Failed to load models.json:', err);
  }
}

export function setupCustomModelsLoader(): void {
  reloadCustomModels();

  const modelsDir = getModelsDir();
  if (!existsSync(modelsDir)) return;

  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  try {
    watch(modelsDir, (_eventType, filename) => {
      if (!filename || (!filename.endsWith('.json') && !filename.endsWith('.glb'))) return;
      if (debounceTimer) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        reloadCustomModels();
      }, 250);
    });
  } catch (err) {
    console.warn('[CustomModels] Could not watch models directory:', err);
  }
}

