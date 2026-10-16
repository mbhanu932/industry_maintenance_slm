/**
 * Type definitions for Industrial Maintenance Copilot Frontend
 * Production-ready type system with comprehensive interfaces
 */

// ═══════════════════════════════════════════════════════════════════════════════
// CORE DATA TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface MachineContext {
  engine_id: number;
  cycle: number;
  predicted_rul: number;
  health_score: number;
  risk_level: RiskLevel;
  anomaly_score: number;
  anomaly_status: string;
  failure_probability_24h: number;
  failure_probability_48h: number;
  last_updated: string;
  location?: string;
  model?: string;
  serial_number?: string;
}

export interface Sensor {
  key: SensorType;
  label: string;
  unit: string;
  icon: string;
  value: number;
  normal_min: number;
  normal_max: number;
  critical_min?: number;
  critical_max?: number;
  delta_pct: number;
  status: SensorStatus;
  trend_direction: 'up' | 'down' | 'stable';
  last_calibration?: string;
}

export interface SensorData {
  engine_id: number;
  sensors: Sensor[];
  trend: TrendDataPoint[];
  timestamp: string;
}

export interface TrendDataPoint {
  hour: string;
  timestamp: string;
  temperature: number;
  pressure: number;
  vibration: number;
  voltage: number;
  rotation: number;
  current: number;
  [key: string]: number | string;
}

export interface MaintenanceEvent {
  id: string;
  engine_id: number;
  event_type: 'maintenance' | 'repair' | 'inspection' | 'replacement';
  description: string;
  technician: string;
  date: string;
  duration_hours: number;
  cost?: number;
  parts_used?: string[];
  completion_status: 'completed' | 'pending' | 'cancelled';
}

export interface ErrorEvent {
  id: string;
  engine_id: number;
  error_code: string;
  description: string;
  severity: ErrorSeverity;
  first_occurrence: string;
  last_occurrence: string;
  count: number;
  resolved: boolean;
  resolution_notes?: string;
}

export interface HistoryData {
  engine_id: number;
  machine_age_years: number;
  last_service_days_ago: number;
  total_maintenance_events: number;
  total_errors: number;
  error_log: ErrorEvent[];
  maintenance_log: MaintenanceEvent[];
  performance_trends: {
    health_score_trend: number[];
    rul_trend: number[];
    anomaly_trend: number[];
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// AI & DIAGNOSIS TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface DiagnosisResult {
  engine_id: number;
  diagnosis_id: string;
  timestamp: string;
  severity: DiagnosisSeverity;
  diagnosis: string;
  confidence_score: number;
  evidence: Evidence[];
  recommended_actions: RecommendedAction[];
  knowledge_evidence: KnowledgeEvidence[];
  reasoning_trace: ReasoningStep[];
  estimated_cost?: number;
  estimated_downtime?: number;
  urgency_level: UrgencyLevel;
  sop_compliance: boolean;
  engineer_review_required: boolean;
}

export interface Evidence {
  id: string;
  type: 'sensor' | 'historical' | 'pattern' | 'correlation';
  description: string;
  weight: number;
  supporting_data: any;
  confidence: number;
}

export interface RecommendedAction {
  id: string;
  action: string;
  priority: ActionPriority;
  estimated_effort_hours: number;
  required_skills: string[];
  parts_needed?: string[];
  safety_considerations: string[];
  completion_deadline?: string;
}

export interface KnowledgeEvidence {
  text: string;
  score: number;
  source: string;
  relevance_score: number;
  document_id?: string;
}

export interface ReasoningStep {
  step: number;
  description: string;
  input_data: any;
  output: any;
  confidence: number;
  processing_time_ms: number;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FLEET & MONITORING TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface FleetSummary {
  total_machines: number;
  critical: number;
  high: number;
  attention: number;
  normal: number;
  offline: number;
  avg_health_score: number;
  avg_rul: number;
  total_active_alerts: number;
  maintenance_due: number;
  last_updated: string;
}

export interface FleetHeatmapData {
  machines: HeatmapMachine[];
  zones: FleetZone[];
  alerts: FleetAlert[];
  timestamp: string;
}

export interface HeatmapMachine {
  engine_id: number;
  x: number;  // Grid position
  y: number;  // Grid position
  zone: string;
  risk_level: RiskLevel;
  health_score: number;
  status: MachineStatus;
  alert_count: number;
  last_maintenance: string;
}

export interface FleetZone {
  id: string;
  name: string;
  machine_count: number;
  avg_health_score: number;
  critical_machines: number;
  color: string;
}

export interface FleetAlert {
  id: string;
  engine_id: number;
  type: AlertType;
  severity: AlertSeverity;
  message: string;
  created_at: string;
  acknowledged: boolean;
  acknowledged_by?: string;
  resolved: boolean;
  resolved_at?: string;
}

export interface PriorityTask {
  id: string;
  engine_id: number;
  task_type: TaskType;
  title: string;
  description: string;
  priority: TaskPriority;
  estimated_effort: number;
  deadline?: string;
  assigned_to?: string;
  dependencies: string[];
  completion_percentage: number;
  created_at: string;
  updated_at: string;
  status: TaskStatus;
}

// ═══════════════════════════════════════════════════════════════════════════════
// AUDIT & COMPLIANCE TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface AuditEvent {
  id: string;
  timestamp: string;
  event_type: AuditEventType;
  user_id: string;
  machine_id?: number;
  action: string;
  details: Record<string, any>;
  ip_address: string;
  user_agent: string;
  session_id: string;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
}

export interface ComplianceReport {
  report_id: string;
  generated_at: string;
  period_start: string;
  period_end: string;
  total_events: number;
  critical_events: number;
  sop_violations: number;
  engineer_approvals: number;
  compliance_score: number;
  recommendations: string[];
}

// ═══════════════════════════════════════════════════════════════════════════════
// ENUM TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export type RiskLevel = 'CRITICAL' | 'HIGH' | 'ATTENTION' | 'NORMAL' | 'LOW';
export type SensorStatus = 'normal' | 'warning' | 'critical' | 'offline';
export type SensorType = 'temperature' | 'pressure' | 'vibration' | 'voltage' | 'rotation' | 'current';
export type DiagnosisSeverity = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
export type UrgencyLevel = 'immediate' | 'urgent' | 'scheduled' | 'routine';
export type ActionPriority = 'critical' | 'high' | 'medium' | 'low';
export type ErrorSeverity = 'critical' | 'major' | 'minor' | 'warning';
export type MachineStatus = 'online' | 'offline' | 'maintenance' | 'error';
export type AlertType = 'sensor' | 'maintenance' | 'error' | 'performance' | 'safety';
export type AlertSeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';
export type TaskType = 'maintenance' | 'inspection' | 'repair' | 'replacement' | 'calibration';
export type TaskPriority = 'critical' | 'high' | 'medium' | 'low';
export type TaskStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled' | 'blocked';
export type AuditEventType = 'login' | 'logout' | 'diagnosis' | 'maintenance' | 'configuration' | 'data_access';

// ═══════════════════════════════════════════════════════════════════════════════
// UI & APPLICATION TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface ApplicationState {
  selectedMachine: number | null;
  currentView: ViewType;
  isLoading: boolean;
  error: string | null;
  lastRefresh: Date;
  autoRefresh: boolean;
  refreshInterval: number;
  notifications: Notification[];
  user: User | null;
}

export type ViewType = 'dashboard' | 'fleet-heatmap' | 'priority-board' | 'machine-detail' | 'audit' | 'settings';

export interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  permissions: Permission[];
  last_login: string;
}

export type UserRole = 'operator' | 'technician' | 'engineer' | 'supervisor' | 'admin';
export type Permission = 'read' | 'write' | 'approve' | 'admin' | 'audit';

export interface Notification {
  id: string;
  type: NotificationType;
  message: string;
  timestamp: string;
  read: boolean;
  action_required: boolean;
  related_machine?: number;
}

export type NotificationType = 'info' | 'warning' | 'error' | 'success' | 'maintenance';

// ═══════════════════════════════════════════════════════════════════════════════
// API RESPONSE TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  timestamp: string;
  request_id: string;
}

export interface ApiError {
  error: string;
  code: number;
  details?: Record<string, any>;
  timestamp: string;
  request_id: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface AppConfig {
  api_base_url: string;
  refresh_intervals: {
    dashboard: number;
    fleet: number;
    machine_detail: number;
  };
  thresholds: {
    health_score: {
      critical: number;
      warning: number;
    };
    anomaly_score: {
      critical: number;
      warning: number;
    };
  };
  features: {
    auto_refresh: boolean;
    notifications: boolean;
    audit_logging: boolean;
    neural_model: boolean;
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
export type RequiredOnly<T, K extends keyof T> = Pick<T, K>;
export type WithTimestamp<T> = T & { timestamp: string };
export type WithId<T> = T & { id: string };

// Type guards
export const isValidRiskLevel = (value: string): value is RiskLevel => {
  return ['CRITICAL', 'HIGH', 'ATTENTION', 'NORMAL', 'LOW'].includes(value);
};

export const isValidSensorStatus = (value: string): value is SensorStatus => {
  return ['normal', 'warning', 'critical', 'offline'].includes(value);
};

export const isValidTaskPriority = (value: string): value is TaskPriority => {
  return ['critical', 'high', 'medium', 'low'].includes(value);
};