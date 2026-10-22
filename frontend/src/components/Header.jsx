import React from 'react';
import { Flame, ShieldCheck, MapPin, Database, Award, BarChart3 } from 'lucide-react';

export default function Header({ summary, dbStatus, onOpenMetrics }) {
  return (
    <header style={{
      height: '64px',
      background: 'rgba(11, 15, 25, 0.95)',
      borderBottom: '1px solid var(--border-color)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 20px',
      zIndex: 1000,
    }}>
      {/* Title & Pilot Info */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <div style={{
          width: '40px',
          height: '40px',
          borderRadius: '10px',
          background: 'linear-gradient(135deg, #f43f5e 0%, #10b981 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 4px 12px rgba(244, 63, 94, 0.3)',
        }}>
          <Flame size={22} color="#ffffff" />
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <h1 style={{ fontSize: '1.15rem', fontWeight: 700, letterSpacing: '-0.02em' }}>
              Urban Cooling Planner
            </h1>
            <span className="badge badge-emerald">AI + GIS + Graph ML</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
              <MapPin size={12} color="#10b981" /> Pilot City: Pune, Maharashtra (Central Core AOI)
            </span>
            <span>•</span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
              <Database size={12} color="#60a5fa" /> {dbStatus?.mode || 'Local Spatial Engine'}
            </span>
          </div>
        </div>
      </div>

      {/* Live Stats Badges */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {summary && (
          <>
            <div style={{
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '8px',
              padding: '6px 12px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'flex-start',
            }}>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Total Grid</span>
              <span style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                {summary.total_cells?.toLocaleString()} cells (100m)
              </span>
            </div>

            <div style={{
              background: 'rgba(244, 63, 94, 0.08)',
              border: '1px solid rgba(244, 63, 94, 0.25)',
              borderRadius: '8px',
              padding: '6px 12px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'flex-start',
            }}>
              <span style={{ fontSize: '0.68rem', color: '#fb7185', textTransform: 'uppercase' }}>Peak Summer LST</span>
              <span style={{ fontSize: '0.88rem', fontWeight: 700, color: '#f43f5e' }}>
                {summary.max_lst_celsius}°C (Hotspots: {summary.critical_hotspot_cells})
              </span>
            </div>

            <div style={{
              background: 'rgba(16, 185, 129, 0.08)',
              border: '1px solid rgba(16, 185, 129, 0.25)',
              borderRadius: '8px',
              padding: '6px 12px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'flex-start',
            }}>
              <span style={{ fontSize: '0.68rem', color: '#34d399', textTransform: 'uppercase' }}>Pop. at Risk</span>
              <span style={{ fontSize: '0.88rem', fontWeight: 700, color: '#10b981' }}>
                {summary.vulnerable_population_at_risk?.toLocaleString()} residents
              </span>
            </div>
          </>
        )}

        <button className="btn-secondary" onClick={onOpenMetrics}>
          <BarChart3 size={15} color="#34d399" />
          <span>Model Metrics & Transparency</span>
        </button>
      </div>
    </header>
  );
}
