import { describe, expect, it } from 'vitest';
import {
  geodeticToEcef,
  egm96GeoidUndulationM,
  createGoogle3dTilesController,
} from '../google-3d-tiles';

describe('google-3d-tiles', () => {
  it('converts Taipei 101 geodetic coordinates to WGS84 ECEF accurately', () => {
    const [x, y, z] = geodeticToEcef(25.033964, 121.564468, 0);
    // Earth radius at 25°N is ~6374 km
    const r = Math.hypot(x, y, z);
    expect(r).toBeGreaterThan(6_370_000);
    expect(r).toBeLessThan(6_380_000);
    // Taipei is in the +Y (East ~121.5°), -X (past 90°E), +Z (North) octant
    expect(x).toBeLessThan(0);
    expect(y).toBeGreaterThan(0);
    expect(z).toBeGreaterThan(0);
  });

  it('estimates EGM96 geoid undulation in Taipei around +19.3m', () => {
    const n = egm96GeoidUndulationM(25.0375, 121.5637);
    expect(n).toBeCloseTo(19.3, 1);
  });

  it('initializes controller and toggles visibility cleanly', () => {
    const ctrl = createGoogle3dTilesController();
    expect(ctrl.hasLoadedTiles()).toBe(false);
    ctrl.setOrigin(25.0375, 121.5637, 0);
    ctrl.configure({ enabled: true, apiKey: '' });
    expect(ctrl.group.visible).toBe(true);
    ctrl.configure({ enabled: false, apiKey: '' });
    expect(ctrl.group.visible).toBe(false);
    ctrl.dispose();
  });
});
