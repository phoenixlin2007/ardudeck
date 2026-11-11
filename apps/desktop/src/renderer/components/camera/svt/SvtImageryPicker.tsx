import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Check, Globe } from 'lucide-react';
import { useCameraStore } from '../../../stores/camera-store';
import { useSceneMeshStore } from '../../../stores/scene-mesh-store';
import { mapLayerName } from '../../../../shared/map-layers';
import { SVT_IMAGERY_LAYERS, validImagery } from './svt-satellite';
import { SVT_BUILDING_SOURCES, validBuildings } from './svt-building-sources';

/** Shared Google Map Tiles API Photorealistic 3D Tiles toggle + API Key input section. */
export function Google3dTilesSettingsSection() {
  const google3dTiles = useCameraStore((s) => s.google3dTiles);
  const setGoogle3dTiles = useCameraStore((s) => s.setGoogle3dTiles);
  const googleMapsApiKey = useCameraStore((s) => s.googleMapsApiKey);
  const setGoogleMapsApiKey = useCameraStore((s) => s.setGoogleMapsApiKey);
  const [draftKey, setDraftKey] = useState(googleMapsApiKey);
  const [editingKey, setEditingKey] = useState(false);

  const saveKey = () => {
    setGoogleMapsApiKey(draftKey);
    setEditingKey(false);
    if (draftKey.trim() && !google3dTiles) {
      setGoogle3dTiles(true);
    }
  };

  return (
    <div className="mb-1 border-b border-subtle pb-1.5">
      <label className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-[11px] font-medium text-content hover:bg-surface-raised">
        <input
          type="checkbox"
          checked={google3dTiles}
          onChange={(e) => setGoogle3dTiles(e.target.checked)}
          className="accent-blue-500"
        />
        <Globe className="h-3 w-3 text-sky-400 shrink-0" />
        {/* i18n-exempt */}
        <span>Google 3D Tiles</span>
      </label>
      {(google3dTiles || editingKey || !googleMapsApiKey) && (
        <div className="mt-1 px-1.5">
          {!editingKey && googleMapsApiKey ? (
            <div className="flex items-center justify-between gap-1 text-[10px] text-content-tertiary">
              {/* i18n-exempt */}
              <span className="truncate font-mono">Key: ••••{googleMapsApiKey.slice(-4)}</span>
              <div className="flex items-center gap-1 shrink-0">
                <button
                  type="button"
                  onClick={() => {
                    setDraftKey(googleMapsApiKey);
                    setEditingKey(true);
                  }}
                  className="rounded px-1 py-0.5 text-[10px] text-sky-400 hover:bg-surface-raised"
                >
                  {/* i18n-exempt */}
                  變更
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setGoogleMapsApiKey('');
                    setDraftKey('');
                    setEditingKey(true);
                  }}
                  className="rounded px-1 py-0.5 text-[10px] text-rose-400 hover:bg-surface-raised"
                >
                  {/* i18n-exempt */}
                  清除
                </button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              <input
                type="password"
                value={draftKey}
                onChange={(e) => setDraftKey(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') saveKey();
                }}
                placeholder="Google Map Tiles API Key" // i18n-exempt
                className="w-full rounded border border-subtle bg-surface-input px-1.5 py-1 text-[10px] font-mono text-content focus:border-sky-500 focus:outline-none"
              />
              <div className="flex justify-end gap-1">
                {editingKey && (
                  <button
                    type="button"
                    onClick={() => setEditingKey(false)}
                    className="rounded px-1.5 py-0.5 text-[10px] text-content-secondary hover:bg-surface-raised"
                  >
                    {/* i18n-exempt */}
                    取消
                  </button>
                )}
                <button
                  type="button"
                  onClick={saveKey}
                  className="rounded bg-sky-600 px-2 py-0.5 text-[10px] font-medium text-white hover:bg-sky-500"
                >
                  {/* i18n-exempt */}
                  套用 Key
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Which of the map's satellite layers textures the synthetic-vision terrain. */
export function SvtImageryPicker({ disabled = false }: { disabled?: boolean }) {
  const { t } = useTranslation();
  const imagery = useCameraStore((s) => validImagery(s.svtImagery));
  const setImagery = useCameraStore((s) => s.setSvtImagery);
  return (
    <div className={disabled ? 'pointer-events-none opacity-50' : ''}>
      <div className="px-1.5 pb-0.5 pt-1 text-[10px] uppercase tracking-wide text-content-tertiary">{t('camera:panel.imagerySource')}</div>
      {SVT_IMAGERY_LAYERS.map((key) => (
        <button
          key={key}
          type="button"
          onClick={() => setImagery(key)}
          className={`flex w-full items-center gap-2 rounded px-1.5 py-1 text-left text-[11px] transition-colors hover:bg-surface-raised ${imagery === key ? 'text-content' : 'text-content-secondary'}`}
        >
          <Check className={`h-3 w-3 shrink-0 ${imagery === key ? 'text-blue-400' : 'invisible'}`} />
          {mapLayerName(key)}
        </button>
      ))}
    </div>
  );
}

/** Where the synthetic-vision buildings come from; independent of the imagery toggle. */
export function SvtBuildingsPicker() {
  const { t } = useTranslation();
  const buildings = useCameraStore((s) => validBuildings(s.svtBuildings));
  const setBuildings = useCameraStore((s) => s.setSvtBuildings);
  const roofImagery = useCameraStore((s) => s.svtRoofImagery);
  const setRoofImagery = useCameraStore((s) => s.setSvtRoofImagery);
  const satellite = useCameraStore((s) => s.svtSatellite);
  const facades = useCameraStore((s) => s.svtFacades);
  const setFacades = useCameraStore((s) => s.setSvtFacades);
  const customGlbEnabled = useSceneMeshStore((s) => s.enabled);
  const setCustomGlbEnabled = useSceneMeshStore((s) => s.setEnabled);
  const off = buildings === 'off';
  return (
    <div>
      <Google3dTilesSettingsSection />
      <label className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-[11px] text-content hover:bg-surface-raised">
        <input
          type="checkbox"
          checked={customGlbEnabled}
          onChange={(e) => setCustomGlbEnabled(e.target.checked)}
          className="accent-blue-500"
        />
        {/* i18n-exempt */}
        3D GLB Landmarks
      </label>
      <div className="px-1.5 pb-0.5 pt-1 text-[10px] uppercase tracking-wide text-content-tertiary">{t('camera:panel.buildingsSource')}</div>
      {SVT_BUILDING_SOURCES.map((key) => (
        <button
          key={key}
          type="button"
          onClick={() => setBuildings(key)}
          data-tip={key === 'off' ? undefined : t(`camera:panel.buildings.${key}Tip`)}
          className={`flex w-full items-center gap-2 rounded px-1.5 py-1 text-left text-[11px] transition-colors hover:bg-surface-raised ${buildings === key ? 'text-content' : 'text-content-secondary'}`}
        >
          <Check className={`h-3 w-3 shrink-0 ${buildings === key ? 'text-blue-400' : 'invisible'}`} />
          {t(`camera:panel.buildings.${key}`)}
        </button>
      ))}
      <div className={off ? 'pointer-events-none opacity-50' : ''}>
        <label
          className={`flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-[11px] hover:bg-surface-raised ${satellite ? 'text-content' : 'text-content-tertiary'}`}
          data-tip={satellite ? undefined : t('camera:panel.roofImageryNeedsSatellite')}
        >
          <input type="checkbox" checked={roofImagery} onChange={(e) => setRoofImagery(e.target.checked)} className="accent-blue-500" />
          {t('camera:panel.roofImagery')}
        </label>
        <label className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-[11px] text-content hover:bg-surface-raised">
          <input type="checkbox" checked={facades} onChange={(e) => setFacades(e.target.checked)} className="accent-blue-500" />
          {t('camera:panel.facades')}
        </label>
      </div>
    </div>
  );
}
