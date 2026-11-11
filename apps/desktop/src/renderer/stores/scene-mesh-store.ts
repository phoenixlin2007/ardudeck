import { create } from 'zustand';
import type { SceneMeshEntry } from '../../shared/tile-source-types';

const STORAGE_KEY = 'ardudeck:custom-glb-enabled';

function loadInitialEnabled(): boolean {
  try {
    if (typeof localStorage !== 'undefined') {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved === 'false') return false;
      if (saved === 'true') return true;
    }
  } catch {
    // ignore storage errors
  }
  return true;
}

interface SceneMeshStore {
  entries: SceneMeshEntry[];
  version: number;
  enabled: boolean;
  setEnabled(enabled: boolean): void;
  toggleEnabled(): void;
  refresh(): Promise<void>;
  start(): () => void;
}

export const useSceneMeshStore = create<SceneMeshStore>((set, get) => ({
  entries: [],
  version: 0,
  enabled: loadInitialEnabled(),
  setEnabled: (enabled: boolean) => {
    try {
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem(STORAGE_KEY, String(enabled));
      }
    } catch {
      // ignore storage errors
    }
    set({ enabled, version: get().version + 1 });
  },
  toggleEnabled: () => {
    get().setEnabled(!get().enabled);
  },
  refresh: async () => {
    try {
      set({ entries: await window.electronAPI.sceneMeshesList(), version: get().version + 1 });
    } catch {
      // older main process: no meshes
    }
  },
  start: () => {
    void get().refresh();
    return window.electronAPI.onSceneMeshesChanged((entries) => set({ entries, version: get().version + 1 }));
  },
}));

export function meshesFor(scene: 'svt' | 'mission3d'): SceneMeshEntry[] {
  const state = useSceneMeshStore.getState();
  if (!state.enabled) return [];
  return state.entries.filter((e) => !e.scenes || e.scenes.includes(scene));
}
