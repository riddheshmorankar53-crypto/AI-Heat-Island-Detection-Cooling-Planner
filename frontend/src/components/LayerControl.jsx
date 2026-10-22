import React from 'react';
import { Layers, Flame, Trees, AlertTriangle, Eye, ShieldAlert } from 'lucide-react';

export default function LayerControl({
  activeLayer,
  onSelectLayer,
  showInterventions,
  onToggleInterventions,
  showRoute,
  onToggleRoute,
  gridOpacity,
  onChangeOpacity,
}) {
  const layerOptions = [
    { id: 'lst', label: 'Land Surface Temp (LST)', icon: Flame, color: '#f43f5e', desc: 'Landsat, Mar-May 2024' },
    { id: 'ndvi', label: 'Vegetation Index (NDVI)', icon: Trees, color: '#10b981', desc: 'Tree Canopy & Greenery' },
    { id: 'hvi', label: 'Heat Vulnerability (HVI)', icon: AlertTriangle, color: '#f59e0b', desc: 'Demographics + Heat Stress' },
    { id: 'hotspots', label: 'Critical Hotspots (Top 15%)', icon: ShieldAlert, color: '#ef4444', desc: 'High Temp + High Risk Core' },
  ];

  return (
    <div className="glass-panel" style={{
      padding: '14px',
      display: 'flex',
      flexDirection: 'column',
      gap: '12px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', fontWeight: 600 }}>
          <Layers size={16} color="#60a5fa" />
          <span>GIS Map Choropleth</span>
        </div>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>100m Grid</span>
      </div>

      {/* Layer selector pills */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
        {layerOptions.map((opt) => {
          const Icon = opt.icon;
          const isSelected = activeLayer === opt.id;
          return (
            <button
              key={opt.id}
              onClick={() => onSelectLayer(opt.id)}
              style={{
                background: isSelected ? 'rgba(255, 255, 255, 0.12)' : 'rgba(255, 255, 255, 0.03)',
                border: isSelected ? `1px solid ${opt.color}` : '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: '8px',
                padding: '8px 10px',
                cursor: 'pointer',
                textAlign: 'left',
                display: 'flex',
                flexDirection: 'column',
                gap: '2px',
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Icon size={14} color={opt.color} />
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color: isSelected ? '#ffffff' : 'var(--text-secondary)' }}>
                  {opt.id.toUpperCase()}
                </span>
              </div>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
                {opt.desc}
              </span>
            </button>
          );
        })}
      </div>

      {/* Opacity slider */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '2px' }}>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Layer Opacity</span>
        <input
          type="range"
          min="0.2"
          max="0.9"
          step="0.05"
          value={gridOpacity}
          onChange={(e) => onChangeOpacity(parseFloat(e.target.value))}
          style={{ width: '120px', cursor: 'pointer' }}
        />
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', width: '30px', textAlign: 'right' }}>
          {Math.round(gridOpacity * 100)}%
        </span>
      </div>

      {/* Layer Toggles */}
      <div style={{
        borderTop: '1px solid rgba(255, 255, 255, 0.08)',
        paddingTop: '10px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '0.78rem',
      }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showInterventions}
            onChange={(e) => onToggleInterventions(e.target.checked)}
          />
          <span>Show Interventions</span>
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showRoute}
            onChange={(e) => onToggleRoute(e.target.checked)}
          />
          <span>Show Walking Routes</span>
        </label>
      </div>
    </div>
  );
}
