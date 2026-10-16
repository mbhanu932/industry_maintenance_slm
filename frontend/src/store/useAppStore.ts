/**
 * Global Application State Management
 * Production-ready Zustand store with TypeScript support
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import {
  ApplicationState,
  ViewType,
  MachineContext,
  FleetSummary,
  FleetHeatmapData,
  PriorityTask,
  Notification,
  User,
  DiagnosisResult,
  SensorData,
  HistoryData,
} from '../types';

interface AppStore extends ApplicationState {
  // ═══════════════════════════════════════════════════════════════════════════
  // DATA STATE
  // ═══════════════════════════════════════════════════════════════════════════
  
  // Machine data
  machineIds: number[];
  machineContexts: Record<number, MachineContext>;
  sensorDataCache: Record<number, SensorData>;
  historyDataCache: Record<number, HistoryData>;
  
  // Fleet data
  fleetSummary: FleetSummary | null;
  fleetHeatmap: FleetHeatmapData | null;
  priorityTasks: PriorityTask[];
  
  // AI & Diagnosis
  diagnosisResults: Record<number, DiagnosisResult[]>;
  activeDiagnosis: DiagnosisResult | null;
  isAnalyzing: boolean;
  
  // ═══════════════════════════════════════════════════════════════════════════
  // UI STATE ACTIONS
  // ═══════════════════════════════════════════════════════════════════════════
  
  setView: (view: ViewType) => void;
  selectMachine: (machineId: number | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setAutoRefresh: (enabled: boolean) => void;
  setRefreshInterval: (interval: number) => void;
  updateLastRefresh: () => void;
  
  // ═══════════════════════════════════════════════════════════════════════════
  // DATA ACTIONS
  // ═══════════════════════════════════════════════════════════════════════════
  
  // Machine data actions
  setMachineIds: (ids: number[]) => void;
  updateMachineContext: (context: MachineContext) => void;
  updateSensorData: (data: SensorData) => void;
  updateHistoryData: (data: HistoryData) => void;
  
  // Fleet data actions
  updateFleetSummary: (summary: FleetSummary) => void;
  updateFleetHeatmap: (heatmap: FleetHeatmapData) => void;
  updatePriorityTasks: (tasks: PriorityTask[]) => void;
  
  // AI & Diagnosis actions
  setActiveDiagnosis: (result: DiagnosisResult | null) => void;
  addDiagnosisResult: (result: DiagnosisResult) => void;
  setAnalyzing: (analyzing: boolean) => void;
  clearDiagnosisHistory: (machineId: number) => void;
  
  // Notification actions
  addNotification: (notification: Omit<Notification, 'id'>) => void;
  removeNotification: (id: string) => void;
  markNotificationRead: (id: string) => void;
  clearAllNotifications: () => void;
  
  // User actions
  setUser: (user: User | null) => void;
  
  // ═══════════════════════════════════════════════════════════════════════════
  // COMPUTED GETTERS
  // ═══════════════════════════════════════════════════════════════════════════
  
  // Get machine data with caching
  getMachineContext: (id: number) => MachineContext | null;
  getSensorData: (id: number) => SensorData | null;
  getHistoryData: (id: number) => HistoryData | null;
  
  // Get fleet insights
  getCriticalMachines: () => number[];
  getHighRiskMachines: () => number[];
  getUnreadNotifications: () => Notification[];
  getHighPriorityTasks: () => PriorityTask[];
  
  // Get statistics
  getFleetHealthMetrics: () => {
    avgHealth: number;
    criticalCount: number;
    maintenanceDue: number;
    totalAlerts: number;
  };
}

export const useAppStore = create<AppStore>()(
  subscribeWithSelector((set, get) => ({
    // ═══════════════════════════════════════════════════════════════════════════
    // INITIAL STATE
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Application state
    selectedMachine: null,
    currentView: 'dashboard' as ViewType,
    isLoading: false,
    error: null,
    lastRefresh: new Date(),
    autoRefresh: true,
    refreshInterval: 30000, // 30 seconds
    notifications: [],
    user: null,
    
    // Data state
    machineIds: [],
    machineContexts: {},
    sensorDataCache: {},
    historyDataCache: {},
    fleetSummary: null,
    fleetHeatmap: null,
    priorityTasks: [],
    diagnosisResults: {},
    activeDiagnosis: null,
    isAnalyzing: false,
    
    // ═══════════════════════════════════════════════════════════════════════════
    // UI ACTIONS
    // ═══════════════════════════════════════════════════════════════════════════
    
    setView: (view) => set({ currentView: view }),
    
    selectMachine: (machineId) => set({ selectedMachine: machineId }),
    
    setLoading: (loading) => set({ isLoading: loading }),
    
    setError: (error) => set({ error }),
    
    setAutoRefresh: (enabled) => set({ autoRefresh: enabled }),
    
    setRefreshInterval: (interval) => set({ refreshInterval: interval }),
    
    updateLastRefresh: () => set({ lastRefresh: new Date() }),
    
    // ═══════════════════════════════════════════════════════════════════════════
    // DATA ACTIONS
    // ═══════════════════════════════════════════════════════════════════════════
    
    setMachineIds: (ids) => set({ machineIds: ids }),
    
    updateMachineContext: (context) => set((state) => ({
      machineContexts: {
        ...state.machineContexts,
        [context.engine_id]: context,
      },
    })),
    
    updateSensorData: (data) => set((state) => ({
      sensorDataCache: {
        ...state.sensorDataCache,
        [data.engine_id]: data,
      },
    })),
    
    updateHistoryData: (data) => set((state) => ({
      historyDataCache: {
        ...state.historyDataCache,
        [data.engine_id]: data,
      },
    })),
    
    updateFleetSummary: (summary) => set({ fleetSummary: summary }),
    
    updateFleetHeatmap: (heatmap) => set({ fleetHeatmap: heatmap }),
    
    updatePriorityTasks: (tasks) => set({ priorityTasks: tasks }),
    
    setActiveDiagnosis: (result) => set({ activeDiagnosis: result }),
    
    addDiagnosisResult: (result) => set((state) => {
      const machineResults = state.diagnosisResults[result.engine_id] || [];
      return {
        diagnosisResults: {
          ...state.diagnosisResults,
          [result.engine_id]: [result, ...machineResults.slice(0, 9)], // Keep last 10
        },
      };
    }),
    
    setAnalyzing: (analyzing) => set({ isAnalyzing: analyzing }),
    
    clearDiagnosisHistory: (machineId) => set((state) => ({
      diagnosisResults: {
        ...state.diagnosisResults,
        [machineId]: [],
      },
    })),
    
    // ═══════════════════════════════════════════════════════════════════════════
    // NOTIFICATION ACTIONS
    // ═══════════════════════════════════════════════════════════════════════════
    
    addNotification: (notification) => set((state) => ({
      notifications: [
        {
          ...notification,
          id: `notif_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        },
        ...state.notifications.slice(0, 99), // Keep last 100
      ],
    })),
    
    removeNotification: (id) => set((state) => ({
      notifications: state.notifications.filter(n => n.id !== id),
    })),
    
    markNotificationRead: (id) => set((state) => ({
      notifications: state.notifications.map(n =>
        n.id === id ? { ...n, read: true } : n
      ),
    })),
    
    clearAllNotifications: () => set({ notifications: [] }),
    
    setUser: (user) => set({ user }),
    
    // ═══════════════════════════════════════════════════════════════════════════
    // COMPUTED GETTERS
    // ═══════════════════════════════════════════════════════════════════════════
    
    getMachineContext: (id) => {
      const state = get();
      return state.machineContexts[id] || null;
    },
    
    getSensorData: (id) => {
      const state = get();
      return state.sensorDataCache[id] || null;
    },
    
    getHistoryData: (id) => {
      const state = get();
      return state.historyDataCache[id] || null;
    },
    
    getCriticalMachines: () => {
      const state = get();
      return Object.values(state.machineContexts)
        .filter(ctx => ctx.risk_level === 'CRITICAL')
        .map(ctx => ctx.engine_id);
    },
    
    getHighRiskMachines: () => {
      const state = get();
      return Object.values(state.machineContexts)
        .filter(ctx => ['CRITICAL', 'HIGH'].includes(ctx.risk_level))
        .map(ctx => ctx.engine_id);
    },
    
    getUnreadNotifications: () => {
      const state = get();
      return state.notifications.filter(n => !n.read);
    },
    
    getHighPriorityTasks: () => {
      const state = get();
      return state.priorityTasks
        .filter(task => ['critical', 'high'].includes(task.priority))
        .sort((a, b) => {
          const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
          return priorityOrder[a.priority] - priorityOrder[b.priority];
        });
    },
    
    getFleetHealthMetrics: () => {
      const state = get();
      const contexts = Object.values(state.machineContexts);
      
      if (contexts.length === 0) {
        return {
          avgHealth: 0,
          criticalCount: 0,
          maintenanceDue: 0,
          totalAlerts: 0,
        };
      }
      
      const avgHealth = contexts.reduce((sum, ctx) => sum + ctx.health_score, 0) / contexts.length;
      const criticalCount = contexts.filter(ctx => ctx.risk_level === 'CRITICAL').length;
      const maintenanceDue = state.priorityTasks.filter(task => 
        task.task_type === 'maintenance' && task.status === 'pending'
      ).length;
      const totalAlerts = state.notifications.filter(n => !n.read && n.type === 'error').length;
      
      return {
        avgHealth: Math.round(avgHealth),
        criticalCount,
        maintenanceDue,
        totalAlerts,
      };
    },
  }))
);

// ═══════════════════════════════════════════════════════════════════════════════
// STORE SELECTORS (for better performance with specific subscriptions)
// ═══════════════════════════════════════════════════════════════════════════════

export const useSelectedMachine = () => useAppStore(state => state.selectedMachine);
export const useCurrentView = () => useAppStore(state => state.currentView);
export const useIsLoading = () => useAppStore(state => state.isLoading);
export const useError = () => useAppStore(state => state.error);
export const useFleetSummary = () => useAppStore(state => state.fleetSummary);
export const useFleetHeatmap = () => useAppStore(state => state.fleetHeatmap);
export const usePriorityTasks = () => useAppStore(state => state.priorityTasks);
export const useActiveDiagnosis = () => useAppStore(state => state.activeDiagnosis);
export const useIsAnalyzing = () => useAppStore(state => state.isAnalyzing);
export const useNotifications = () => useAppStore(state => state.notifications);
export const useUnreadNotifications = () => useAppStore(state => state.getUnreadNotifications());
export const useUser = () => useAppStore(state => state.user);
export const useFleetMetrics = () => useAppStore(state => state.getFleetHealthMetrics());

// Machine-specific selectors
export const useMachineContext = (id: number | null) => 
  useAppStore(state => id ? state.getMachineContext(id) : null);

export const useSensorData = (id: number | null) => 
  useAppStore(state => id ? state.getSensorData(id) : null);

export const useHistoryData = (id: number | null) => 
  useAppStore(state => id ? state.getHistoryData(id) : null);

// ═══════════════════════════════════════════════════════════════════════════════
// STORE PERSISTENCE (for critical settings)
// ═══════════════════════════════════════════════════════════════════════════════

// Subscribe to auto-refresh changes and persist to localStorage
useAppStore.subscribe(
  (state) => state.autoRefresh,
  (autoRefresh) => {
    localStorage.setItem('copilot_auto_refresh', JSON.stringify(autoRefresh));
  }
);

useAppStore.subscribe(
  (state) => state.refreshInterval,
  (refreshInterval) => {
    localStorage.setItem('copilot_refresh_interval', JSON.stringify(refreshInterval));
  }
);

useAppStore.subscribe(
  (state) => state.currentView,
  (currentView) => {
    localStorage.setItem('copilot_current_view', JSON.stringify(currentView));
  }
);

// Initialize from localStorage on startup
const initializeFromStorage = () => {
  try {
    const savedAutoRefresh = localStorage.getItem('copilot_auto_refresh');
    if (savedAutoRefresh) {
      useAppStore.getState().setAutoRefresh(JSON.parse(savedAutoRefresh));
    }
    
    const savedInterval = localStorage.getItem('copilot_refresh_interval');
    if (savedInterval) {
      useAppStore.getState().setRefreshInterval(JSON.parse(savedInterval));
    }
    
    const savedView = localStorage.getItem('copilot_current_view');
    if (savedView) {
      useAppStore.getState().setView(JSON.parse(savedView));
    }
  } catch (error) {
    console.warn('Failed to initialize from localStorage:', error);
  }
};

// Call on module load
if (typeof window !== 'undefined') {
  initializeFromStorage();
}