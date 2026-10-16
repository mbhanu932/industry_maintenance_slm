/**
 * Fleet Heatmap Component
 * Production-ready visual fleet monitoring with interactive machine grid
 */

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MapPin,
  AlertTriangle,
  Settings,
  ZoomIn,
  ZoomOut,
  Filter,
  Download,
  Maximize2,
  RefreshCw,
  Info,
} from 'lucide-react';
import { useAppStore, useFleetHeatmap, useIsLoading } from '../store/useAppStore';
import { api } from '../services/api';
import { RiskLevel, MachineStatus, HeatmapMachine, FleetZone } from '../types';

interface FleetHeatmapProps {
  className?: string;
}

interface HeatmapFilters {
  riskLevels: RiskLevel[];
  statuses: MachineStatus[];
  zones: string[];
  showOffline: boolean;
  showMaintenanceDue: boolean;
}

const DEFAULT_FILTERS: HeatmapFilters = {
  riskLevels: ['CRITICAL', 'HIGH', 'ATTENTION', 'NORMAL'],
  statuses: ['online', 'offline', 'maintenance', 'error'],
  zones: [],
  showOffline: true,
  showMaintenanceDue: true,
};

// Risk level colors with enhanced visual hierarchy
const RISK_COLORS: Record<RiskLevel, string> = {
  CRITICAL: '#ef4444',  // Red
  HIGH: '#f59e0b',      // Orange
  ATTENTION: '#f97316', // Orange-red
  NORMAL: '#22c55e',    // Green
  LOW: '#6b7280',       // Gray
};

const STATUS_COLORS: Record<MachineStatus, string> = {
  online: '#22c55e',      // Green
  offline: '#6b7280',     // Gray
  maintenance: '#3b82f6', // Blue
  error: '#ef4444',       // Red
};

const ZONE_COLORS = [
  '#8b5cf6', '#06b6d4', '#84cc16', '#f59e0b', 
  '#ef4444', '#ec4899', '#10b981', '#f97316'
];

export const FleetHeatmap: React.FC<FleetHeatmapProps> = ({ className = '' }) => {
  const {
    updateFleetHeatmap,
    selectMachine,
    selectedMachine,
    setView,
  } = useAppStore();
  
  const fleetHeatmap = useFleetHeatmap();
  const isLoading = useIsLoading();
  
  // Local state
  const [filters, setFilters] = useState<HeatmapFilters>(DEFAULT_FILTERS);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showFilters, setShowFilters] = useState(false);
  const [hoveredMachine, setHoveredMachine] = useState<HeatmapMachine | null>(null);
  const [selectedZone, setSelectedZone] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [gridSize, setGridSize] = useState({ rows: 10, cols: 12 });
  
  // Load heatmap data
  useEffect(() => {
    loadHeatmapData();
  }, []);
  
  const loadHeatmapData = async () => {
    setRefreshing(true);
    try {
      const data = await api.fleet.getFleetHeatmap();
      updateFleetHeatmap(data);
      
      // Auto-adjust grid size based on machine positions
      if (data.machines.length > 0) {
        const maxX = Math.max(...data.machines.map(m => m.x));
        const maxY = Math.max(...data.machines.map(m => m.y));
        setGridSize({
          rows: Math.max(10, maxY + 2),
          cols: Math.max(12, maxX + 2),
        });
      }
    } catch (error) {
      console.error('Failed to load heatmap data:', error);
    } finally {
      setRefreshing(false);
    }
  };
  
  // Filter machines based on current filters
  const filteredMachines = useMemo(() => {
    if (!fleetHeatmap?.machines) return [];
    
    return fleetHeatmap.machines.filter(machine => {
      // Risk level filter
      if (!filters.riskLevels.includes(machine.risk_level)) return false;
      
      // Status filter
      if (!filters.statuses.includes(machine.status)) return false;
      
      // Zone filter
      if (filters.zones.length > 0 && !filters.zones.includes(machine.zone)) return false;
      
      // Offline filter
      if (!filters.showOffline && machine.status === 'offline') return false;
      
      // Maintenance due filter (simplified - would need API support)
      if (!filters.showMaintenanceDue && machine.alert_count === 0) return false;
      
      return true;
    });
  }, [fleetHeatmap?.machines, filters]);
  
  // Generate zone statistics
  const zoneStats = useMemo(() => {
    if (!fleetHeatmap?.zones) return {};
    
    const stats: Record<string, {
      total: number;
      critical: number;
      avgHealth: number;
      color: string;
    }> = {};
    
    fleetHeatmap.zones.forEach((zone, index) => {
      const zoneMachines = filteredMachines.filter(m => m.zone === zone.id);
      stats[zone.id] = {
        total: zoneMachines.length,
        critical: zoneMachines.filter(m => m.risk_level === 'CRITICAL').length,
        avgHealth: zoneMachines.length > 0 
          ? zoneMachines.reduce((sum, m) => sum + m.health_score, 0) / zoneMachines.length
          : 0,
        color: ZONE_COLORS[index % ZONE_COLORS.length],
      };
    });
    
    return stats;
  }, [filteredMachines, fleetHeatmap?.zones]);
  
  // Handle machine click
  const handleMachineClick = (machine: HeatmapMachine) => {
    selectMachine(machine.engine_id);
    setView('machine-detail');
  };
  
  // Handle filter changes
  const updateFilter = <K extends keyof HeatmapFilters>(
    key: K,
    value: HeatmapFilters[K]
  ) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };
  
  // Generate grid cells
  const generateGrid = () => {
    const cells = [];
    for (let row = 0; row < gridSize.rows; row++) {
      for (let col = 0; col < gridSize.cols; col++) {
        const machine = filteredMachines.find(m => m.x === col && m.y === row);
        cells.push(
          <GridCell
            key={`${row}-${col}`}
            x={col}
            y={row}
            machine={machine}
            isSelected={machine?.engine_id === selectedMachine}
            onMachineClick={handleMachineClick}
            onMachineHover={setHoveredMachine}
            zoomLevel={zoomLevel}
          />
        );
      }
    }
    return cells;
  };
  
  // Calculate grid dimensions based on zoom
  const cellSize = 48 * zoomLevel;
  const gridWidth = gridSize.cols * cellSize;
  const gridHeight = gridSize.rows * cellSize;
  
  if (isLoading && !fleetHeatmap) {
    return (
      <div className="fleet-heatmap-loading">
        <div className="spinner large" />
        <p>Loading fleet heatmap...</p>
      </div>
    );
  }
  
  return (
    <div className={`fleet-heatmap ${className}`}>
      {/* Header Controls */}
      <div className="heatmap-header">
        <div className="heatmap-title">
          <MapPin size={20} />
          <h2>Fleet Heatmap</h2>
          <span className="machine-count">
            {filteredMachines.length} / {fleetHeatmap?.machines.length || 0} machines
          </span>
        </div>
        
        <div className="heatmap-controls">
          <button
            className="btn btn-icon"
            onClick={() => setShowFilters(!showFilters)}
            title="Filters"
          >
            <Filter size={16} />
          </button>
          
          <div className="zoom-controls">
            <button
              className="btn btn-icon"
              onClick={() => setZoomLevel(Math.max(0.5, zoomLevel - 0.25))}
              disabled={zoomLevel <= 0.5}
              title="Zoom out"
            >
              <ZoomOut size={16} />
            </button>
            <span className="zoom-level">{Math.round(zoomLevel * 100)}%</span>
            <button
              className="btn btn-icon"
              onClick={() => setZoomLevel(Math.min(3, zoomLevel + 0.25))}
              disabled={zoomLevel >= 3}
              title="Zoom in"
            >
              <ZoomIn size={16} />
            </button>
          </div>
          
          <button
            className="btn btn-primary"
            onClick={loadHeatmapData}
            disabled={refreshing}
            title="Refresh data"
          >
            <RefreshCw size={16} className={refreshing ? 'spinning' : ''} />
            Refresh
          </button>
        </div>
      </div>
      
      {/* Filter Panel */}
      <AnimatePresence>
        {showFilters && (
          <motion.div
            className="heatmap-filters"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <FilterPanel
              filters={filters}
              zones={fleetHeatmap?.zones || []}
              onFilterChange={updateFilter}
            />
          </motion.div>
        )}
      </AnimatePresence>
      
      <div className="heatmap-content">
        {/* Zone Legend */}
        {fleetHeatmap?.zones && fleetHeatmap.zones.length > 0 && (
          <div className="zone-legend">
            <h4>Zones</h4>
            <div className="zone-list">
              {fleetHeatmap.zones.map((zone, index) => {
                const stats = zoneStats[zone.id];
                return (
                  <div
                    key={zone.id}
                    className={`zone-item ${selectedZone === zone.id ? 'selected' : ''}`}
                    onClick={() => setSelectedZone(selectedZone === zone.id ? null : zone.id)}
                  >
                    <div
                      className="zone-color"
                      style={{ backgroundColor: ZONE_COLORS[index % ZONE_COLORS.length] }}
                    />
                    <div className="zone-info">
                      <div className="zone-name">{zone.name}</div>
                      <div className="zone-stats">
                        {stats?.total || 0} machines
                        {stats?.critical > 0 && (
                          <span className="critical-count">
                            • {stats.critical} critical
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
        
        {/* Main Grid */}
        <div className="heatmap-grid-container">
          <div
            className="heatmap-grid"
            style={{
              width: gridWidth,
              height: gridHeight,
              gridTemplateColumns: `repeat(${gridSize.cols}, ${cellSize}px)`,
              gridTemplateRows: `repeat(${gridSize.rows}, ${cellSize}px)`,
            }}
          >
            {generateGrid()}
          </div>
        </div>
        
        {/* Status Legend */}
        <div className="status-legend">
          <h4>Risk Levels</h4>
          <div className="legend-items">
            {Object.entries(RISK_COLORS).map(([level, color]) => (
              <div key={level} className="legend-item">
                <div className="legend-dot" style={{ backgroundColor: color }} />
                <span>{level}</span>
              </div>
            ))}
          </div>
          
          <h4>Status</h4>
          <div className="legend-items">
            {Object.entries(STATUS_COLORS).map(([status, color]) => (
              <div key={status} className="legend-item">
                <div className="legend-dot status" style={{ borderColor: color }} />
                <span>{status}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* Machine Tooltip */}
      <AnimatePresence>
        {hoveredMachine && (
          <MachineTooltip machine={hoveredMachine} />
        )}
      </AnimatePresence>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// GRID CELL COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

interface GridCellProps {
  x: number;
  y: number;
  machine?: HeatmapMachine;
  isSelected: boolean;
  onMachineClick: (machine: HeatmapMachine) => void;
  onMachineHover: (machine: HeatmapMachine | null) => void;
  zoomLevel: number;
}

const GridCell: React.FC<GridCellProps> = ({
  x,
  y,
  machine,
  isSelected,
  onMachineClick,
  onMachineHover,
  zoomLevel,
}) => {
  if (!machine) {
    return <div className="grid-cell empty" />;
  }
  
  const riskColor = RISK_COLORS[machine.risk_level];
  const statusColor = STATUS_COLORS[machine.status];
  const showDetails = zoomLevel >= 1;
  
  return (
    <motion.div
      className={`grid-cell machine ${isSelected ? 'selected' : ''} ${machine.status}`}
      style={{
        backgroundColor: riskColor,
        borderColor: statusColor,
        borderWidth: machine.status === 'offline' ? '2px' : '1px',
        borderStyle: machine.status === 'maintenance' ? 'dashed' : 'solid',
      }}
      onClick={() => onMachineClick(machine)}
      onMouseEnter={() => onMachineHover(machine)}
      onMouseLeave={() => onMachineHover(null)}
      whileHover={{ scale: 1.05, zIndex: 10 }}
      whileTap={{ scale: 0.95 }}
      transition={{ duration: 0.1 }}
    >
      {/* Machine ID */}
      <div className="machine-id">
        {showDetails ? `ENG-${machine.engine_id.toString().padStart(3, '0')}` : machine.engine_id}
      </div>
      
      {/* Health Score */}
      {showDetails && (
        <div className="health-score">
          {machine.health_score.toFixed(0)}
        </div>
      )}
      
      {/* Alert Indicators */}
      {machine.alert_count > 0 && (
        <div className="alert-indicator">
          <AlertTriangle size={zoomLevel >= 1.5 ? 12 : 8} />
          {zoomLevel >= 1.5 && <span>{machine.alert_count}</span>}
        </div>
      )}
      
      {/* Maintenance Due Indicator */}
      {machine.last_maintenance && (
        <div className="maintenance-indicator">
          <Settings size={zoomLevel >= 1.5 ? 10 : 6} />
        </div>
      )}
    </motion.div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// FILTER PANEL COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

interface FilterPanelProps {
  filters: HeatmapFilters;
  zones: FleetZone[];
  onFilterChange: <K extends keyof HeatmapFilters>(key: K, value: HeatmapFilters[K]) => void;
}

const FilterPanel: React.FC<FilterPanelProps> = ({
  filters,
  zones,
  onFilterChange,
}) => {
  const toggleRiskLevel = (level: RiskLevel) => {
    const newLevels = filters.riskLevels.includes(level)
      ? filters.riskLevels.filter(l => l !== level)
      : [...filters.riskLevels, level];
    onFilterChange('riskLevels', newLevels);
  };
  
  const toggleStatus = (status: MachineStatus) => {
    const newStatuses = filters.statuses.includes(status)
      ? filters.statuses.filter(s => s !== status)
      : [...filters.statuses, status];
    onFilterChange('statuses', newStatuses);
  };
  
  const toggleZone = (zoneId: string) => {
    const newZones = filters.zones.includes(zoneId)
      ? filters.zones.filter(z => z !== zoneId)
      : [...filters.zones, zoneId];
    onFilterChange('zones', newZones);
  };
  
  return (
    <div className="filter-panel">
      <div className="filter-group">
        <label>Risk Levels:</label>
        <div className="filter-options">
          {(Object.keys(RISK_COLORS) as RiskLevel[]).map(level => (
            <button
              key={level}
              className={`filter-chip ${filters.riskLevels.includes(level) ? 'active' : ''}`}
              onClick={() => toggleRiskLevel(level)}
              style={{
                borderColor: RISK_COLORS[level],
                backgroundColor: filters.riskLevels.includes(level) ? RISK_COLORS[level] : 'transparent',
              }}
            >
              {level}
            </button>
          ))}
        </div>
      </div>
      
      <div className="filter-group">
        <label>Status:</label>
        <div className="filter-options">
          {(Object.keys(STATUS_COLORS) as MachineStatus[]).map(status => (
            <button
              key={status}
              className={`filter-chip ${filters.statuses.includes(status) ? 'active' : ''}`}
              onClick={() => toggleStatus(status)}
            >
              {status}
            </button>
          ))}
        </div>
      </div>
      
      {zones.length > 0 && (
        <div className="filter-group">
          <label>Zones:</label>
          <div className="filter-options">
            {zones.map(zone => (
              <button
                key={zone.id}
                className={`filter-chip ${filters.zones.includes(zone.id) ? 'active' : ''}`}
                onClick={() => toggleZone(zone.id)}
              >
                {zone.name}
              </button>
            ))}
          </div>
        </div>
      )}
      
      <div className="filter-group">
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={filters.showOffline}
            onChange={(e) => onFilterChange('showOffline', e.target.checked)}
          />
          Show offline machines
        </label>
        
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={filters.showMaintenanceDue}
            onChange={(e) => onFilterChange('showMaintenanceDue', e.target.checked)}
          />
          Show maintenance due
        </label>
      </div>
      
      <div className="filter-actions">
        <button
          className="btn btn-secondary"
          onClick={() => onFilterChange('riskLevels', ['CRITICAL', 'HIGH', 'ATTENTION', 'NORMAL'])}
        >
          Reset Filters
        </button>
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// MACHINE TOOLTIP COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

interface MachineTooltipProps {
  machine: HeatmapMachine;
}

const MachineTooltip: React.FC<MachineTooltipProps> = ({ machine }) => {
  return (
    <motion.div
      className="machine-tooltip"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ duration: 0.1 }}
    >
      <div className="tooltip-header">
        <h4>ENG-{machine.engine_id.toString().padStart(3, '0')}</h4>
        <span className={`risk-badge risk-${machine.risk_level}`}>
          {machine.risk_level}
        </span>
      </div>
      
      <div className="tooltip-content">
        <div className="tooltip-row">
          <span>Health Score:</span>
          <span className="value">{machine.health_score.toFixed(1)}</span>
        </div>
        <div className="tooltip-row">
          <span>Status:</span>
          <span className={`status-badge status-${machine.status}`}>
            {machine.status}
          </span>
        </div>
        <div className="tooltip-row">
          <span>Zone:</span>
          <span className="value">{machine.zone}</span>
        </div>
        {machine.alert_count > 0 && (
          <div className="tooltip-row">
            <span>Active Alerts:</span>
            <span className="value alert">{machine.alert_count}</span>
          </div>
        )}
        {machine.last_maintenance && (
          <div className="tooltip-row">
            <span>Last Maintenance:</span>
            <span className="value">{machine.last_maintenance}</span>
          </div>
        )}
      </div>
      
      <div className="tooltip-footer">
        <Info size={12} />
        <span>Click to view details</span>
      </div>
    </motion.div>
  );
};

export default FleetHeatmap;