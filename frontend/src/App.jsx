import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import MapView from './components/MapView';
import LayerControl from './components/LayerControl';
import PlannerPanel from './components/PlannerPanel';
import RoutePanel from './components/RoutePanel';
import MetricsModal from './components/MetricsModal';
import AlertsPanel from './components/AlertsPanel';
import {
  fetchHealth,
  fetchAOI,
  fetchHotspotsSummary,
  fetchGrid,
  fetchInterventionCatalog,
  optimizeInterventions,
  fetchRoutingLandmarks,
  calculateHeatSafeRoute,
  fetchMLMetrics,
  fetchHeatAlerts,
} from './services/api';
import { Cpu, Navigation, AlertCircle, Siren } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('planner'); // 'planner' | 'routing' | 'alerts'
  const [activeLayer, setActiveLayer] = useState('lst'); // 'lst' | 'ndvi' | 'hvi' | 'hotspots'
  const [gridOpacity, setGridOpacity] = useState(0.70);
  const [showInterventions, setShowInterventions] = useState(true);
  const [showRoute, setShowRoute] = useState(true);

  // Data states
  const [gridGeoJson, setGridGeoJson] = useState(null);
  const [summary, setSummary] = useState(null);
  const [dbStatus, setDbStatus] = useState(null);
  const [aoiMetadata, setAoiMetadata] = useState(null);
  const [catalog, setCatalog] = useState(null);
  const [landmarks, setLandmarks] = useState(null);
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [routeResult, setRouteResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [alertsData, setAlertsData] = useState(null);
  const [isMetricsOpen, setIsMetricsOpen] = useState(false);

  // Loading & error states
  const [isLoadingGrid, setIsLoadingGrid] = useState(true);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [isRouting, setIsRouting] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Initial load
  useEffect(() => {
    async function init() {
      try {
        setIsLoadingGrid(true);
        const [healthData, aoiData, summaryData, catalogData, landmarksData, metricsData, alerts] =
          await Promise.all([
            fetchHealth().catch(() => null),
            fetchAOI().catch(() => null),
            fetchHotspotsSummary().catch(() => null),
            fetchInterventionCatalog().catch(() => null),
            fetchRoutingLandmarks().catch(() => null),
            fetchMLMetrics().catch(() => null),
            fetchHeatAlerts().catch(() => null),
          ]);

        if (healthData) setDbStatus(healthData.database);
        if (aoiData) setAoiMetadata(aoiData);
        if (summaryData) setSummary(summaryData);
        if (catalogData) setCatalog(catalogData);
        if (landmarksData) setLandmarks(landmarksData);
        if (metricsData) setMetrics(metricsData);
        if (alerts) setAlertsData(alerts);

        // Fetch initial grid GeoJSON
        const gridData = await fetchGrid();
        setGridGeoJson(gridData);

        // Pre-run default optimization so map is immediately populated with cooling interventions
        if (catalogData) {
          const defaultOpt = await optimizeInterventions({
            budget_inr: 1500000,
            enabled_types: Object.keys(catalogData),
            max_interventions_per_cell: 1,
            priority_hotspots_only: false,
          });
          setOptimizationResult(defaultOpt);
        }

        // Pre-calculate default route between Shaniwar Wada and Pune Station
        if (landmarksData && landmarksData.length >= 2) {
          const defaultRoute = await calculateHeatSafeRoute({
            origin: { lon: 73.8553, lat: 18.5196 },
            destination: { lon: 73.8744, lat: 18.5284 },
            heat_avoidance_weight: 2.5,
          });
          setRouteResult(defaultRoute);
        }
      } catch (err) {
        console.error('Initialization error:', err);
        setErrorMessage(err.message || 'Failed to initialize application.');
      } finally {
        setIsLoadingGrid(false);
      }
    }

    init();
  }, []);

  // Handle running OR-Tools optimizer
  const handleOptimize = async (payload) => {
    try {
      setIsOptimizing(true);
      setErrorMessage(null);
      const res = await optimizeInterventions(payload);
      setOptimizationResult(res);
      setShowInterventions(true);
    } catch (err) {
      console.error('Optimization error:', err);
      setErrorMessage(`Optimization failed: ${err.message}`);
    } finally {
      setIsOptimizing(false);
    }
  };

  // Handle calculating heat-safe walking route
  const handleRoute = async (payload) => {
    try {
      setIsRouting(true);
      setErrorMessage(null);
      const res = await calculateHeatSafeRoute(payload);
      setRouteResult(res);
      setShowRoute(true);
    } catch (err) {
      console.error('Routing error:', err);
      setErrorMessage(`Routing failed: ${err.message}`);
    } finally {
      setIsRouting(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', overflow: 'hidden' }}>
      {/* Header */}
      <Header
        summary={summary}
        dbStatus={dbStatus}
        onOpenMetrics={() => setIsMetricsOpen(true)}
      />

      {/* Main Workspace Layout */}
      <div style={{ display: 'flex', flex: 1, position: 'relative', overflow: 'hidden' }}>
        {/* Floating Sidebar Controls */}
        <aside style={{
          position: 'absolute',
          top: '16px',
          left: '16px',
          zIndex: 1000,
          width: '380px',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px',
        }}>
          {/* Navigation Tab Switcher */}
          <div className="glass-panel" style={{ padding: '6px', display: 'flex', gap: '6px' }}>
            <button
              onClick={() => setActiveTab('planner')}
              style={{
                flex: 1,
                padding: '8px 12px',
                borderRadius: '8px',
                border: 'none',
                background: activeTab === 'planner' ? 'linear-gradient(135deg, #10b981 0%, #059669 100%)' : 'transparent',
                color: activeTab === 'planner' ? '#ffffff' : 'var(--text-secondary)',
                fontWeight: 600,
                fontSize: '0.82rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px',
                transition: 'all 0.15s ease',
              }}
            >
              <Cpu size={15} />
              <span>Cooling Planner</span>
            </button>

            <button
              onClick={() => setActiveTab('routing')}
              style={{
                flex: 1,
                padding: '8px 12px',
                borderRadius: '8px',
                border: 'none',
                background: activeTab === 'routing' ? 'linear-gradient(135deg, #06b6d4 0%, #0284c7 100%)' : 'transparent',
                color: activeTab === 'routing' ? '#ffffff' : 'var(--text-secondary)',
                fontWeight: 600,
                fontSize: '0.82rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px',
                transition: 'all 0.15s ease',
              }}
            >
              <Navigation size={15} />
              <span>Heat-Safe Routes</span>
            </button>

            <button
              onClick={() => setActiveTab('alerts')}
              style={{
                flex: 1,
                padding: '8px 12px',
                borderRadius: '8px',
                border: 'none',
                background: activeTab === 'alerts' ? 'linear-gradient(135deg, #f43f5e 0%, #e11d48 100%)' : 'transparent',
                color: activeTab === 'alerts' ? '#ffffff' : 'var(--text-secondary)',
                fontWeight: 600,
                fontSize: '0.82rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px',
                transition: 'all 0.15s ease',
              }}
            >
              <Siren size={15} />
              <span>Alerts</span>
            </button>
          </div>

          {/* Layer Controls */}
          <LayerControl
            activeLayer={activeLayer}
            onSelectLayer={setActiveLayer}
            showInterventions={showInterventions}
            onToggleInterventions={setShowInterventions}
            showRoute={showRoute}
            onToggleRoute={setShowRoute}
            gridOpacity={gridOpacity}
            onChangeOpacity={setGridOpacity}
          />

          {/* Active Feature Panel */}
          {activeTab === 'planner' && (
            <PlannerPanel
              catalog={catalog}
              onOptimize={handleOptimize}
              isLoading={isOptimizing}
              optimizationResult={optimizationResult}
            />
          )}

          {activeTab === 'routing' && (
            <RoutePanel
              landmarks={landmarks}
              onCalculateRoute={handleRoute}
              isLoading={isRouting}
              routeResult={routeResult}
            />
          )}

          {activeTab === 'alerts' && <AlertsPanel alertsData={alertsData} />}

          {/* Error Banner if any */}
          {errorMessage && (
            <div className="glass-panel" style={{
              padding: '10px 14px',
              borderLeft: '4px solid #f43f5e',
              fontSize: '0.78rem',
              color: '#fb7185',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}>
              <AlertCircle size={16} />
              <span>{errorMessage}</span>
            </div>
          )}
        </aside>

        {/* Map View */}
        <main style={{ flex: 1, height: '100%', width: '100%' }}>
          <MapView
            gridGeoJson={gridGeoJson}
            activeLayer={activeLayer}
            gridOpacity={gridOpacity}
            interventions={optimizationResult}
            showInterventions={showInterventions}
            routeResult={routeResult}
            showRoute={showRoute}
            aoiMetadata={aoiMetadata}
          />
        </main>
      </div>

      {/* Model Transparency & Metrics Modal */}
      <MetricsModal
        isOpen={isMetricsOpen}
        onClose={() => setIsMetricsOpen(false)}
        metrics={metrics}
      />
    </div>
  );
}
