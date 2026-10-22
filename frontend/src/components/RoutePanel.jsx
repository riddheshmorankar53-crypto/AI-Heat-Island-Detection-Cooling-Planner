import React, { useState } from 'react';
import { Navigation, Compass, Shield, Footprints, ArrowRight, SunMedium, Umbrella } from 'lucide-react';

export default function RoutePanel({
  landmarks,
  onCalculateRoute,
  isLoading,
  routeResult,
}) {
  const [originId, setOriginId] = useState('SHANIWAR_WADA');
  const [destId, setDestId] = useState('PUNE_STATION');
  const [heatWeight, setHeatWeight] = useState(2.5);

  const handleRoute = () => {
    const orig = landmarks?.find((l) => l.id === originId);
    const dest = landmarks?.find((l) => l.id === destId);
    if (orig && dest) {
      onCalculateRoute({
        origin: { lon: orig.lon, lat: orig.lat },
        destination: { lon: dest.lon, lat: dest.lat },
        heat_avoidance_weight: heatWeight,
      });
    }
  };

  return (
    <div className="glass-panel" style={{
      padding: '16px',
      display: 'flex',
      flexDirection: 'column',
      gap: '14px',
      maxHeight: 'calc(100vh - 180px)',
      overflowY: 'auto',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Compass size={18} color="#06b6d4" />
          <h2 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Heat-Safe Walking Route</h2>
        </div>
        <span className="badge badge-blue">NetworkX Routing</span>
      </div>

      {/* Origin Selection */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Origin (A)</label>
        <select
          value={originId}
          onChange={(e) => setOriginId(e.target.value)}
          style={{
            background: 'rgba(255, 255, 255, 0.06)',
            color: 'var(--text-primary)',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            borderRadius: '6px',
            padding: '8px 10px',
            fontSize: '0.82rem',
            outline: 'none',
          }}
        >
          {landmarks?.map((lm) => (
            <option key={lm.id} value={lm.id} style={{ background: '#111827', color: '#fff' }}>
              {lm.name} ({lm.ambient_lst}°C)
            </option>
          ))}
        </select>
      </div>

      {/* Destination Selection */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Destination (B)</label>
        <select
          value={destId}
          onChange={(e) => setDestId(e.target.value)}
          style={{
            background: 'rgba(255, 255, 255, 0.06)',
            color: 'var(--text-primary)',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            borderRadius: '6px',
            padding: '8px 10px',
            fontSize: '0.82rem',
            outline: 'none',
          }}
        >
          {landmarks?.map((lm) => (
            <option key={lm.id} value={lm.id} style={{ background: '#111827', color: '#fff' }}>
              {lm.name} ({lm.ambient_lst}°C)
            </option>
          ))}
        </select>
      </div>

      {/* Heat Avoidance Sensitivity */}
      <div className="glass-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>Heat Avoidance Sensitivity</span>
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#06b6d4' }}>
            {heatWeight <= 1.0 ? 'Direct/Fast' : heatWeight <= 2.5 ? 'Balanced Shade' : 'Maximum Coolness'}
          </span>
        </div>
        <input
          type="range"
          min="0.5"
          max="4.5"
          step="0.5"
          value={heatWeight}
          onChange={(e) => setHeatWeight(parseFloat(e.target.value))}
          style={{ width: '100%', cursor: 'pointer' }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: '4px' }}>
          <span>Fastest</span>
          <span>Balanced</span>
          <span>Max Shade</span>
        </div>
      </div>

      {/* Find Route Button */}
      <button
        className="btn-primary"
        onClick={handleRoute}
        disabled={isLoading || originId === destId}
        style={{ background: 'linear-gradient(135deg, #06b6d4 0%, #0284c7 100%)' }}
      >
        <Navigation size={16} />
        <span>{isLoading ? 'Computing Routes...' : 'Calculate Heat-Safe Route'}</span>
      </button>

      {/* Route Comparison Output */}
      {routeResult && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '6px' }}>
          <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            Route Thermal Stress Analysis
          </div>

          {/* Side by side comparison */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            {/* Fastest Route */}
            <div className="glass-card" style={{ borderLeft: '3px solid #f97316' }}>
              <div style={{ fontSize: '0.72rem', color: '#f97316', fontWeight: 600 }}>FASTEST ROUTE</div>
              <div style={{ fontSize: '0.9rem', fontWeight: 700, marginTop: '2px' }}>
                {routeResult.fastest_route.properties.distance_meters} m
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                {routeResult.fastest_route.properties.walk_time_minutes} mins
              </div>
              <div style={{ fontSize: '0.72rem', color: '#f97316', marginTop: '4px' }}>
                Shade: {routeResult.fastest_route.properties.shade_percentage}%
              </div>
            </div>

            {/* Heat-Safe Route */}
            <div className="glass-card" style={{ borderLeft: '3px solid #10b981' }}>
              <div style={{ fontSize: '0.72rem', color: '#10b981', fontWeight: 600 }}>HEAT-SAFE ROUTE</div>
              <div style={{ fontSize: '0.9rem', fontWeight: 700, marginTop: '2px' }}>
                {routeResult.heat_safe_route.properties.distance_meters} m
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                {routeResult.heat_safe_route.properties.walk_time_minutes} mins
              </div>
              <div style={{ fontSize: '0.72rem', color: '#34d399', marginTop: '4px' }}>
                Shade: {routeResult.heat_safe_route.properties.shade_percentage}%
              </div>
            </div>
          </div>

          {/* Benefits badge */}
          <div className="glass-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Shield size={18} color="#34d399" />
              <div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Heat Stress Reduction</div>
                <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#34d399' }}>
                  -{routeResult.comparison.heat_stress_reduction_pct}% thermal load
                </div>
              </div>
            </div>
            <div style={{ textAlign: 'right', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              +{routeResult.comparison.extra_distance_meters}m detour
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
