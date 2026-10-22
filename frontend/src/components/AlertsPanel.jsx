import React from 'react';
import { Siren, Thermometer, Users, MapPin } from 'lucide-react';

const SEVERITY = {
  extreme: { label: 'Extreme', badge: 'badge-rose', color: '#f43f5e' },
  high: { label: 'High', badge: 'badge-amber', color: '#f59e0b' },
  moderate: { label: 'Moderate', badge: 'badge-blue', color: '#3b82f6' },
};

export default function AlertsPanel({ alertsData, selectedAlert, onSelectAlert }) {
  if (!alertsData) {
    return (
      <div className="glass-panel" style={{ padding: '16px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
        Loading heat alerts...
      </div>
    );
  }

  return (
    <div className="glass-panel" style={{
      padding: '16px',
      display: 'flex',
      flexDirection: 'column',
      gap: '12px',
      maxHeight: 'calc(100vh - 180px)',
      overflowY: 'auto',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Siren size={18} color="#f43f5e" />
          <h2 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Heat Hazard Alerts</h2>
        </div>
        <span className="badge badge-rose">{alertsData.total_alerts} hotspots</span>
      </div>

      <div style={{ display: 'flex', gap: '6px' }}>
        {Object.entries(SEVERITY).map(([key, s]) => (
          <div key={key} className="glass-card" style={{ flex: 1, padding: '8px', textAlign: 'center' }}>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: s.color }}>
              {alertsData.counts?.[key] ?? 0}
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>{s.label}</div>
          </div>
        ))}
      </div>

      {alertsData.forecast ? (
        <div className="glass-card" style={{
          padding: '8px 10px',
          fontSize: '0.74rem',
          borderLeft: `3px solid ${alertsData.escalation_levels > 0 ? '#f43f5e' : '#10b981'}`,
        }}>
          <div style={{ fontWeight: 600, marginBottom: '4px' }}>
            {alertsData.escalation_levels > 0
              ? `Heatwave forecast: severity raised ${alertsData.escalation_levels} level(s)`
              : 'No heatwave forecast (all days below 40°C)'}
          </div>
          <div style={{ display: 'flex', gap: '10px', color: 'var(--text-secondary)' }}>
            {alertsData.forecast.days.map((d) => (
              <span key={d.date}>{d.date.slice(5)}: <strong>{d.tmax_celsius}°C</strong></span>
            ))}
          </div>
        </div>
      ) : (
        <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          Weather forecast unavailable; severity from satellite data only.
        </p>
      )}

      <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
        Top {alertsData.alerts.length} cells by Heat Vulnerability Index. Click to locate on map.
      </p>

      {alertsData.alerts.map((a) => {
        const s = SEVERITY[a.severity];
        return (
          <div key={a.cell_id} className="glass-card" onClick={() => onSelectAlert(a)} style={{
            padding: '10px 12px',
            borderLeft: `3px solid ${s.color}`,
            cursor: 'pointer',
            outline: selectedAlert?.cell_id === a.cell_id ? `1px solid ${s.color}` : 'none',
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.74rem' }}>{a.cell_id}</span>
              <span className={`badge ${s.badge}`}>{s.label}</span>
            </div>
            <div style={{ display: 'flex', gap: '12px', fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
                <Thermometer size={12} /> {a.lst_celsius.toFixed(1)}°C (+{a.lst_anomaly_celsius.toFixed(1)})
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
                <Users size={12} /> {a.population}
              </span>
              <span>HVI {a.heat_vulnerability_index.toFixed(2)}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '3px', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              <MapPin size={11} /> {a.lat.toFixed(4)}, {a.lon.toFixed(4)}
            </div>
            <ul style={{ margin: 0, paddingLeft: '16px', fontSize: '0.72rem', color: 'var(--text-primary)' }}>
              {a.actions.map((act) => <li key={act}>{act}</li>)}
            </ul>
          </div>
        );
      })}
    </div>
  );
}
