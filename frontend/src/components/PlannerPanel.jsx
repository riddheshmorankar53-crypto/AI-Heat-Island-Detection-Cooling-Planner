import React, { useState } from 'react';
import { Sliders, Cpu, CheckCircle2, TrendingDown, Users, DollarSign, RefreshCw } from 'lucide-react';

export default function PlannerPanel({
  catalog,
  onOptimize,
  isLoading,
  optimizationResult,
}) {
  const [budget, setBudget] = useState(1500000); // 15 Lakh INR
  const [enabledTypes, setEnabledTypes] = useState([
    'trees',
    'cool_roofs',
    'water_points',
    'shaded_stops',
    'cooling_centers',
  ]);
  const [priorityHotspotsOnly, setPriorityHotspotsOnly] = useState(false);

  const toggleType = (key) => {
    setEnabledTypes((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    );
  };

  const handleRunOptimizer = () => {
    onOptimize({
      budget_inr: budget,
      enabled_types: enabledTypes,
      max_interventions_per_cell: 1,
      priority_hotspots_only: priorityHotspotsOnly,
    });
  };

  const formatLakhs = (val) => {
    if (val >= 10000000) return `₹${(val / 10000000).toFixed(1)} Cr`;
    if (val >= 100000) return `₹${(val / 100000).toFixed(1)} Lakh`;
    return `₹${val.toLocaleString('en-IN')}`;
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
          <Cpu size={18} color="#10b981" />
          <h2 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Cooling Intervention Optimizer</h2>
        </div>
        <span className="badge badge-emerald">OR-Tools MILP</span>
      </div>

      {/* Budget Slider */}
      <div className="glass-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Municipal Budget</span>
          <span style={{ fontSize: '0.95rem', fontWeight: 700, color: '#10b981' }}>
            {formatLakhs(budget)}
          </span>
        </div>
        <input
          type="range"
          min="300000"
          max="5000000"
          step="100000"
          value={budget}
          onChange={(e) => setBudget(parseFloat(e.target.value))}
          style={{ width: '100%', cursor: 'pointer' }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: '4px' }}>
          <span>₹3 Lakh</span>
          <span>₹25 Lakh</span>
          <span>₹50 Lakh</span>
        </div>
      </div>

      {/* Intervention Types Selection */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
          Available Intervention Types
        </span>
        {catalog && Object.entries(catalog).map(([key, item]) => {
          const isEnabled = enabledTypes.includes(key);
          return (
            <div
              key={key}
              onClick={() => toggleType(key)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '8px 10px',
                borderRadius: '8px',
                background: isEnabled ? 'rgba(255, 255, 255, 0.05)' : 'rgba(255, 255, 255, 0.02)',
                border: isEnabled ? `1px solid ${item.color}` : '1px solid rgba(255, 255, 255, 0.06)',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: item.color }} />
                <span style={{ fontSize: '0.78rem', fontWeight: 500 }}>{item.name}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.72rem' }}>
                <span style={{ color: '#34d399' }} title={item.basis}>
                  -{item.delta_celsius}°C
                  <span style={{
                    marginLeft: '4px',
                    fontSize: '0.62rem',
                    color: item.basis?.startsWith('Measured') ? '#34d399' : 'var(--text-muted)',
                  }}>
                    {item.basis?.startsWith('Measured') ? 'measured' : 'assumed'}
                  </span>
                </span>
                <span style={{ color: 'var(--text-muted)' }}>{formatLakhs(item.cost_inr)}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Hotspots priority filter */}
      <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.78rem', cursor: 'pointer' }}>
        <input
          type="checkbox"
          checked={priorityHotspotsOnly}
          onChange={(e) => setPriorityHotspotsOnly(e.target.checked)}
        />
        <span>Prioritize Critical Hotspots Only (Top 15%)</span>
      </label>

      {/* Run button */}
      <button
        className="btn-primary"
        onClick={handleRunOptimizer}
        disabled={isLoading || enabledTypes.length === 0}
      >
        {isLoading ? (
          <>
            <RefreshCw size={16} className="spin" />
            <span>Solving MILP Program...</span>
          </>
        ) : (
          <>
            <Cpu size={16} />
            <span>Run AI Cooling Optimizer</span>
          </>
        )}
      </button>

      {/* Optimization Results */}
      {optimizationResult && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '6px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Optimal Placement Solution
            </span>
            <span className="badge badge-emerald">{optimizationResult.status}</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            <div className="glass-card">
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Total Cost</span>
              <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#10b981' }}>
                {formatLakhs(optimizationResult.summary.total_cost_inr)}
              </div>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
                {optimizationResult.summary.budget_utilization_pct}% of budget
              </span>
            </div>

            <div className="glass-card">
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Interventions Placed</span>
              <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#60a5fa' }}>
                {optimizationResult.summary.interventions_count} sites
              </div>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
                Avg {optimizationResult.summary.avg_cooling_impact_celsius}°C cooling
              </span>
            </div>
          </div>

          <div className="glass-card" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Users size={20} color="#fb7185" />
            <div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Population Protected</div>
              <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f43f5e' }}>
                {optimizationResult.summary.total_population_protected?.toLocaleString()} residents
              </div>
            </div>
          </div>

          {/* Type breakdown list */}
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {Object.entries(optimizationResult.summary.type_breakdown).map(([k, count]) => {
              if (count === 0) return null;
              return (
                <span key={k} style={{
                  background: 'rgba(255, 255, 255, 0.05)',
                  padding: '3px 6px',
                  borderRadius: '4px',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                }}>
                  {k}: <strong>{count}</strong>
                </span>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
