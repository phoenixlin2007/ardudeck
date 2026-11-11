import { describe, expect, it } from 'vitest';
import { meshesFor, useSceneMeshStore } from './scene-mesh-store';

describe('scene-mesh-store', () => {
  it('filters by scene with both as the default, and respects enabled toggle', () => {
    useSceneMeshStore.setState({
      entries: [
        { slug: 'a', key: 'a::both', id: 'both', name: 'b', glb: '/b', url: 'u', origin: { lat: 0, lng: 0, altMsl: 0 } },
        { slug: 'a', key: 'a::svt', id: 'svt', name: 's', glb: '/s', url: 'u', origin: { lat: 0, lng: 0, altMsl: 0 }, scenes: ['svt'] },
      ],
      version: 1,
      enabled: true,
    });
    expect(meshesFor('svt').map((m) => m.id)).toEqual(['both', 'svt']);
    expect(meshesFor('mission3d').map((m) => m.id)).toEqual(['both']);

    useSceneMeshStore.getState().setEnabled(false);
    expect(meshesFor('svt')).toEqual([]);
    expect(meshesFor('mission3d')).toEqual([]);

    useSceneMeshStore.getState().toggleEnabled();
    expect(meshesFor('svt').map((m) => m.id)).toEqual(['both', 'svt']);
  });
});
