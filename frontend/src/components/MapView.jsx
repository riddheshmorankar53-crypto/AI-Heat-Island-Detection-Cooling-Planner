import React, { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, GeoJSON, Circle, CircleMarker, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import L from 'leaflet';

// Pune Center Coordinates
const PUNE_CENTER = [18.5175, 73.8550];

// Color mapping functions
// Bins span the real Landsat pre-monsoon range for central Pune (~34-50°C)
function getLSTColor(temp) {
  if (temp >= 45.0) return '#b91c1c'; // deep crimson
  if (temp >= 43.0) return '#ef4444'; // red
  if (temp >= 41.5) return '#f97316'; // orange
  if (temp >= 40.0) return '#eab308'; // amber/yellow
  if (temp >= 38.5) return '#84cc16'; // lime green
  if (temp >= 37.0) return '#10b981'; // emerald
  return '#06b6d4'; // cool cyan
}

const ALERT_COLORS = { extreme: '#f43f5e', high: '#f59e0b', moderate: '#3b82f6' };

function getNDVIColor(ndvi) {
  if (ndvi >= 0.50) return '#065f46'; // deep forest
  if (ndvi >= 0.38) return '#059669'; // rich emerald
  if (ndvi >= 0.28) return '#10b981'; // green
  if (ndvi >= 0.20) return '#84cc16'; // lime
  if (ndvi >= 0.12) return '#eab308'; // yellow
  return '#78716c'; // arid/built stone
}

function getHVIColor(hvi) {
  if (hvi >= 0.75) return '#7f1d1d'; // dark maroon
  if (hvi >= 0.65) return '#dc2626'; // high red
  if (hvi >= 0.50) return '#f97316'; // moderate orange
  if (hvi >= 0.35) return '#facc15'; // low yellow
  return '#10b981'; // safe green
}

// Custom Leaflet DivIcons for interventions
function createInterventionIcon(color, label) {
  return L.divIcon({
    className: 'custom-intervention-marker',
    html: `
      <div style="
        background: ${color};
        width: 24px;
        height: 24px;
        border-radius: 50%;
        border: 2px solid #ffffff;
        box-shadow: 0 0 10px ${color}, 0 2px 6px rgba(0,0,0,0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-size: 11px;
        font-weight: bold;
      ">
        ${label}
      </div>
    `,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  });
}

function MapUpdater({ aoiCenter }) {
  const map = useMap();
  useEffect(() => {
    if (aoiCenter && aoiCenter.lat && aoiCenter.lon) {
      map.setView([aoiCenter.lat, aoiCenter.lon], 13);
    }
  }, [aoiCenter, map]);
  return null;
}

function FlyToAlert({ alert }) {
  const map = useMap();
  useEffect(() => {
    if (alert) map.flyTo([alert.lat, alert.lon], 16, { duration: 0.8 });
  }, [alert, map]);
  return null;
}

export default function MapView({
  gridGeoJson,
  activeLayer,
  gridOpacity,
  interventions,
  showInterventions,
  routeResult,
  showRoute,
  aoiMetadata,
  alerts,
  selectedAlert,
}) {
  const geoJsonRef = useRef(null);

  // Style function for 100m grid cells
  const gridStyle = (feature) => {
    const props = feature.properties;
    let fillColor = '#ffffff';
    let fillOpacity = gridOpacity;
    let strokeColor = 'rgba(255,255,255,0.06)';
    let weight = 0.5;

    if (activeLayer === 'lst') {
      fillColor = getLSTColor(props.lst_celsius);
    } else if (activeLayer === 'ndvi') {
      fillColor = getNDVIColor(props.ndvi);
    } else if (activeLayer === 'hvi') {
      fillColor = getHVIColor(props.heat_vulnerability_index);
    } else if (activeLayer === 'hotspots') {
      if (props.is_critical_hotspot) {
        fillColor = '#dc2626';
        strokeColor = '#f87171';
        weight = 1.2;
        fillOpacity = Math.min(1.0, gridOpacity + 0.2);
      } else {
        fillColor = 'transparent';
        fillOpacity = 0.05;
      }
    }

    return {
      fillColor,
      fillOpacity,
      color: strokeColor,
      weight,
    };
  };

  // Click & tooltip handler
  const onEachCell = (feature, layer) => {
    const p = feature.properties;
    const content = `
      <div style="font-family: 'Inter', sans-serif; min-width: 190px;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 4px; margin-bottom: 6px;">
          <strong style="font-size: 0.85rem; color: #f9fafb;">${p.cell_id}</strong>
          ${p.is_critical_hotspot ? '<span style="background: rgba(239,68,68,0.2); color: #f87171; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">HOTSPOT</span>' : ''}
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 0.75rem;">
          <div><span style="color: #9ca3af;">LST:</span> <strong style="color: #f43f5e;">${p.lst_celsius}°C</strong></div>
          <div><span style="color: #9ca3af;">NDVI:</span> <strong style="color: #34d399;">${p.ndvi}</strong></div>
          <div><span style="color: #9ca3af;">Canopy:</span> <strong>${p.tree_canopy_pct}%</strong></div>
          <div><span style="color: #9ca3af;">Impervious:</span> <strong>${p.impervious_pct}%</strong></div>
          <div><span style="color: #9ca3af;">Pop Density:</span> <strong>${p.pop_density_sqkm}/km²</strong></div>
          <div><span style="color: #9ca3af;">HVI Score:</span> <strong style="color: #f59e0b;">${p.heat_vulnerability_index}</strong></div>
        </div>
      </div>
    `;
    layer.bindPopup(content);
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <MapContainer
        center={PUNE_CENTER}
        zoom={13}
        style={{ width: '100%', height: '100%' }}
        scrollWheelZoom={true}
      >
        {/* Base dark satellite / canvas tiles */}
        <TileLayer
          attribution='Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
          maxNativeZoom={16}
          maxZoom={19}
        />

        <MapUpdater aoiCenter={aoiMetadata?.center} />

        {/* 100m Grid Choropleth Layer */}
        {gridGeoJson && (
          <GeoJSON
            key={`${activeLayer}-${gridOpacity}`}
            ref={geoJsonRef}
            data={gridGeoJson}
            style={gridStyle}
            onEachFeature={onEachCell}
          />
        )}

        {/* Cooling Interventions Layer */}
        {showInterventions && interventions?.features?.map((item, idx) => {
          const coords = [item.geometry.coordinates[1], item.geometry.coordinates[0]];
          const p = item.properties;
          const letter = p.intervention_type[0].toUpperCase();

          return (
            <React.Fragment key={`intervention-${idx}`}>
              {/* Cooling Influence Radius Circle */}
              <Circle
                center={coords}
                radius={p.influence_radius_m}
                pathOptions={{
                  color: p.color,
                  fillColor: p.color,
                  fillOpacity: 0.18,
                  weight: 1.5,
                  dashArray: '3, 4',
                }}
              />

              {/* Central Marker */}
              <Marker position={coords} icon={createInterventionIcon(p.color, letter)}>
                <Popup>
                  <div style={{ fontFamily: "'Inter', sans-serif" }}>
                    <div style={{ fontWeight: 700, fontSize: '0.9rem', color: p.color }}>{p.intervention_name}</div>
                    <div style={{ fontSize: '0.75rem', color: '#9ca3af', margin: '3px 0 6px 0' }}>Cell: {p.cell_id}</div>
                    <div style={{ fontSize: '0.78rem', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                      <div>Cost: <strong>₹{p.cost_inr.toLocaleString('en-IN')}</strong></div>
                      <div>Cooling Impact: <strong style={{ color: '#34d399' }}>-{p.cooling_delta_celsius}°C</strong></div>
                      <div>Influence Radius: <strong>{p.influence_radius_m} meters</strong></div>
                      <div>Current Cell LST: <strong>{p.current_lst}°C</strong></div>
                    </div>
                  </div>
                </Popup>
              </Marker>
            </React.Fragment>
          );
        })}

        {/* Heat Hazard Alerts Layer */}
        {alerts?.map((a) => (
          <CircleMarker
            key={a.cell_id}
            center={[a.lat, a.lon]}
            radius={a.cell_id === selectedAlert?.cell_id ? 11 : 7}
            pathOptions={{
              color: '#ffffff',
              weight: a.cell_id === selectedAlert?.cell_id ? 3 : 1.5,
              fillColor: ALERT_COLORS[a.severity],
              fillOpacity: 0.9,
            }}
          >
            <Popup>
              <div style={{ fontFamily: "'Inter', sans-serif", fontSize: '0.78rem' }}>
                <strong style={{ color: ALERT_COLORS[a.severity] }}>
                  {a.severity.toUpperCase()} heat alert
                </strong>
                <div>{a.cell_id} · {a.lst_celsius.toFixed(1)}°C · {a.population} people</div>
              </div>
            </Popup>
          </CircleMarker>
        ))}
        <FlyToAlert alert={selectedAlert} />

        {/* Walking Routes Layer */}
        {showRoute && routeResult && (
          <>
            {/* Fastest Route (Dashed Orange) */}
            <Polyline
              positions={routeResult.fastest_route.geometry.coordinates.map((c) => [c[1], c[0]])}
              pathOptions={{
                color: '#f97316',
                weight: 4,
                dashArray: '8, 8',
                opacity: 0.85,
              }}
            >
              <Popup>
                <div>
                  <strong>Fastest Route</strong><br />
                  Distance: {routeResult.fastest_route.properties.distance_meters} m<br />
                  Shade: {routeResult.fastest_route.properties.shade_percentage}%
                </div>
              </Popup>
            </Polyline>

            {/* Heat-Safe Route (Solid Emerald with Glow) */}
            <Polyline
              positions={routeResult.heat_safe_route.geometry.coordinates.map((c) => [c[1], c[0]])}
              pathOptions={{
                color: '#10b981',
                weight: 6,
                opacity: 0.95,
              }}
            >
              <Popup>
                <div>
                  <strong style={{ color: '#10b981' }}>Heat-Safe Cool Route</strong><br />
                  Distance: {routeResult.heat_safe_route.properties.distance_meters} m<br />
                  Shade: {routeResult.heat_safe_route.properties.shade_percentage}%<br />
                  Stress Reduction: -{routeResult.comparison.heat_stress_reduction_pct}%
                </div>
              </Popup>
            </Polyline>

            {/* Origin & Destination Waypoints */}
            <Marker
              position={[routeResult.origin.lat, routeResult.origin.lon]}
              icon={createInterventionIcon('#3b82f6', 'A')}
            >
              <Popup><strong>Origin (A)</strong><br />{routeResult.origin.node_id}</Popup>
            </Marker>
            <Marker
              position={[routeResult.destination.lat, routeResult.destination.lon]}
              icon={createInterventionIcon('#ec4899', 'B')}
            >
              <Popup><strong>Destination (B)</strong><br />{routeResult.destination.node_id}</Popup>
            </Marker>
          </>
        )}
      </MapContainer>

      {/* Map Legend Overlay */}
      <div className="glass-panel" style={{
        position: 'absolute',
        bottom: '28px',
        right: '20px',
        padding: '10px 14px',
        zIndex: 1000,
        fontSize: '0.72rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
      }}>
        <strong style={{ color: 'var(--text-primary)', fontSize: '0.75rem' }}>
          {activeLayer === 'lst' && 'Land Surface Temp (°C)'}
          {activeLayer === 'ndvi' && 'Vegetation Index (NDVI)'}
          {activeLayer === 'hvi' && 'Heat Vulnerability Index'}
          {activeLayer === 'hotspots' && 'Critical Hotspot Status'}
        </strong>

        {activeLayer === 'lst' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ color: 'var(--text-muted)' }}>&le;37°C</span>
            <div style={{
              width: '120px',
              height: '8px',
              borderRadius: '4px',
              background: 'linear-gradient(90deg, #06b6d4 0%, #10b981 25%, #eab308 50%, #f97316 75%, #b91c1c 100%)',
            }} />
            <span style={{ color: 'var(--text-muted)' }}>&ge;45°C</span>
          </div>
        )}

        {activeLayer === 'ndvi' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ color: 'var(--text-muted)' }}>Low (0.05)</span>
            <div style={{
              width: '120px',
              height: '8px',
              borderRadius: '4px',
              background: 'linear-gradient(90deg, #78716c 0%, #eab308 30%, #10b981 70%, #065f46 100%)',
            }} />
            <span style={{ color: 'var(--text-muted)' }}>High (0.65)</span>
          </div>
        )}

        {activeLayer === 'hvi' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ color: 'var(--text-muted)' }}>0.1</span>
            <div style={{
              width: '120px',
              height: '8px',
              borderRadius: '4px',
              background: 'linear-gradient(90deg, #10b981 0%, #facc15 35%, #f97316 70%, #7f1d1d 100%)',
            }} />
            <span style={{ color: 'var(--text-muted)' }}>0.9</span>
          </div>
        )}

        {activeLayer === 'hotspots' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: '12px', height: '12px', background: '#dc2626', borderRadius: '2px' }} />
            <span>Critical Hotspot (Top 15% severity)</span>
          </div>
        )}
      </div>
    </div>
  );
}
