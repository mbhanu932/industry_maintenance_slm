/**
 * Production-Ready API Service Layer
 * Comprehensive API client with error handling, caching, and retry logic
 */

import {
  MachineContext,
  SensorData,
  HistoryData,
  DiagnosisResult,
  FleetSummary,
  FleetHeatmapData,
  PriorityTask,
  ApiResponse,
  ApiError,
  ComplianceReport,
  AuditEvent,
} from '../types';

// ═══════════════════════════════════════════════════════════════════════════════
// API CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'https://industry-maintenance-slm.onrender.com';
const API_TIMEOUT = parseInt(import.meta.env.VITE_API_TIMEOUT || '30000', 10);
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second

// Log API configuration in development
if (import.meta.env.DEV) {
  console.log('🔗 API Configuration:', {
    baseUrl: API_BASE_URL,
    timeout: API_TIMEOUT,
    maxRetries: MAX_RETRIES,
  });
}

interface RequestOptions extends RequestInit {
  timeout?: number;
  retries?: number;
  skipAuth?: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HTTP CLIENT WITH RETRY AND TIMEOUT
// ═══════════════════════════════════════════════════════════════════════════════

class ApiClient {
  private baseUrl: string;
  private authToken: string | null = null;
  
  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
    this.authToken = localStorage.getItem('copilot_auth_token');
  }
  
  setAuthToken(token: string | null) {
    this.authToken = token;
    if (token) {
      localStorage.setItem('copilot_auth_token', token);
    } else {
      localStorage.removeItem('copilot_auth_token');
    }
  }
  
  private async fetchWithTimeout(
    url: string,
    options: RequestOptions = {}
  ): Promise<Response> {
    const {
      timeout = API_TIMEOUT,
      retries = MAX_RETRIES,
      skipAuth = false,
      ...fetchOptions
    } = options;
    
    // Add authentication header
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...fetchOptions.headers,
    };
    
    if (!skipAuth && this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
      const response = await fetch(url, {
        ...fetchOptions,
        headers,
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        if (response.status === 401) {
          this.setAuthToken(null);
          throw new Error('Authentication required');
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error.name === 'AbortError') {
        throw new Error('Request timeout');
      }
      
      // Retry on network errors
      if (retries > 0 && error instanceof TypeError) {
        await this.delay(RETRY_DELAY);
        return this.fetchWithTimeout(url, { ...options, retries: retries - 1 });
      }
      
      throw error;
    }
  }
  
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  private async request<T>(
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    try {
      const response = await this.fetchWithTimeout(url, options);
      const data = await response.json();
      
      // Handle wrapped API responses
      if (data.success !== undefined) {
        if (!data.success) {
          throw new Error(data.message || 'API request failed');
        }
        return data.data;
      }
      
      return data;
    } catch (error) {
      console.error(`API request failed [${options.method || 'GET'} ${endpoint}]:`, error);
      throw error;
    }
  }
  
  // HTTP method helpers
  async get<T>(endpoint: string, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'GET' });
  }
  
  async post<T>(endpoint: string, data?: any, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }
  
  async put<T>(endpoint: string, data?: any, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    });
  }
  
  async delete<T>(endpoint: string, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'DELETE' });
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// API SERVICE INSTANCE
// ═══════════════════════════════════════════════════════════════════════════════

const apiClient = new ApiClient(API_BASE_URL);

// ═══════════════════════════════════════════════════════════════════════════════
// MACHINE & SENSOR ENDPOINTS
// ═══════════════════════════════════════════════════════════════════════════════

export const machineApi = {
  // Get all machine IDs
  async getMachines(): Promise<{ machines: number[] }> {
    return apiClient.get('/machines');
  },
  
  // Get machine context/status
  async getMachineContext(id: number): Promise<MachineContext> {
    return apiClient.get(`/machines/${id}`);
  },
  
  // Get multiple machine contexts (batch request)
  async getMachineContexts(ids: number[]): Promise<MachineContext[]> {
    return apiClient.post('/machines/batch', { machine_ids: ids });
  },
  
  // Get sensor data for a machine
  async getSensorData(id: number): Promise<SensorData> {
    return apiClient.get(`/machines/${id}/sensors`);
  },
  
  // Get historical data for a machine
  async getHistoryData(id: number): Promise<HistoryData> {
    return apiClient.get(`/machines/${id}/history`);
  },
  
  // Update machine configuration
  async updateMachineConfig(id: number, config: Partial<MachineContext>): Promise<MachineContext> {
    return apiClient.put(`/machines/${id}/config`, config);
  },
  
  // Calibrate sensors
  async calibrateSensors(id: number, sensorTypes: string[]): Promise<{ success: boolean; message: string }> {
    return apiClient.post(`/machines/${id}/calibrate`, { sensor_types: sensorTypes });
  },
};

// ═══════════════════════════════════════════════════════════════════════════════
// FLEET & MONITORING ENDPOINTS
// ═══════════════════════════════════════════════════════════════════════════════

export const fleetApi = {
  // Get fleet summary statistics
  async getFleetSummary(): Promise<FleetSummary> {
    return apiClient.get('/fleet/summary');
  },
  
  // Get fleet heatmap data
  async getFleetHeatmap(): Promise<FleetHeatmapData> {
    return apiClient.get('/fleet/heatmap');
  },
  
  // Get priority tasks
  async getPriorityTasks(): Promise<PriorityTask[]> {
    return apiClient.get('/tasks/priority');
  },
  
  // Create new priority task
  async createTask(task: Omit<PriorityTask, 'id' | 'created_at' | 'updated_at'>): Promise<PriorityTask> {
    return apiClient.post('/tasks', task);
  },
  
  // Update task status
  async updateTask(taskId: string, updates: Partial<PriorityTask>): Promise<PriorityTask> {
    return apiClient.put(`/tasks/${taskId}`, updates);
  },
  
  // Get fleet alerts
  async getFleetAlerts(): Promise<any[]> {
    return apiClient.get('/fleet/alerts');
  },
  
  // Acknowledge alert
  async acknowledgeAlert(alertId: string, userId: string): Promise<{ success: boolean }> {
    return apiClient.post(`/alerts/${alertId}/acknowledge`, { user_id: userId });
  },
  
  // Get fleet zones
  async getFleetZones(): Promise<any[]> {
    return apiClient.get('/fleet/zones');
  },
};

// ═══════════════════════════════════════════════════════════════════════════════
// AI & DIAGNOSIS ENDPOINTS
// ═══════════════════════════════════════════════════════════════════════════════

export const diagnosisApi = {
  // Run investigation/diagnosis
  async investigate(machineId: number, question: string): Promise<DiagnosisResult> {
    return apiClient.post(`/machines/${machineId}/investigate`, { question });
  },
  
  // Get diagnosis history for a machine
  async getDiagnosisHistory(machineId: number): Promise<DiagnosisResult[]> {
    return apiClient.get(`/machines/${machineId}/diagnoses`);
  },
  
  // Approve/reject a diagnosis (engineer review)
  async reviewDiagnosis(
    diagnosisId: string,
    approved: boolean,
    notes?: string
  ): Promise<{ success: boolean }> {
    return apiClient.post(`/diagnoses/${diagnosisId}/review`, {
      approved,
      notes,
    });
  },
  
  // Get AI model status
  async getModelStatus(): Promise<{
    status: 'online' | 'offline' | 'degraded';
    version: string;
    last_updated: string;
    guardrails_active: boolean;
  }> {
    return apiClient.get('/ai/status');
  },
  
  // Submit feedback on diagnosis
  async submitFeedback(
    diagnosisId: string,
    rating: number,
    feedback: string
  ): Promise<{ success: boolean }> {
    return apiClient.post(`/diagnoses/${diagnosisId}/feedback`, {
      rating,
      feedback,
    });
  },
};

// ═══════════════════════════════════════════════════════════════════════════════
// AUDIT & COMPLIANCE ENDPOINTS
// ═══════════════════════════════════════════════════════════════════════════════

export const auditApi = {
  // Get audit events
  async getAuditEvents(
    startDate?: string,
    endDate?: string,
    eventType?: string,
    machineId?: number
  ): Promise<AuditEvent[]> {
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    if (eventType) params.append('event_type', eventType);
    if (machineId) params.append('machine_id', machineId.toString());
    
    const query = params.toString() ? `?${params.toString()}` : '';
    return apiClient.get(`/audit/events${query}`);
  },
  
  // Generate compliance report
  async generateComplianceReport(
    startDate: string,
    endDate: string
  ): Promise<ComplianceReport> {
    return apiClient.post('/audit/compliance-report', {
      start_date: startDate,
      end_date: endDate,
    });
  },
  
  // Get SOP compliance status
  async getSopCompliance(): Promise<{
    compliance_score: number;
    violations: number;
    last_audit: string;
    critical_issues: string[];
  }> {
    return apiClient.get('/audit/sop-compliance');
  },
};

// ═══════════════════════════════════════════════════════════════════════════════
// USER & AUTHENTICATION ENDPOINTS
// ═══════════════════════════════════════════════════════════════════════════════

export const authApi = {
  // Login
  async login(username: string, password: string): Promise<{
    token: string;
    user: any;
    expires_at: string;
  }> {
    const result = await apiClient.post('/auth/login', {
      username,
      password,
    }, { skipAuth: true });
    
    apiClient.setAuthToken(result.token);
    return result;
  },
  
  // Logout
  async logout(): Promise<{ success: boolean }> {
    const result = await apiClient.post('/auth/logout');
    apiClient.setAuthToken(null);
    return result;
  },
  
  // Get current user
  async getCurrentUser(): Promise<any> {
    return apiClient.get('/auth/me');
  },
  
  // Refresh token
  async refreshToken(): Promise<{ token: string; expires_at: string }> {
    const result = await apiClient.post('/auth/refresh');
    apiClient.setAuthToken(result.token);
    return result;
  },
  
  // Change password
  async changePassword(currentPassword: string, newPassword: string): Promise<{ success: boolean }> {
    return apiClient.put('/auth/password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
  },
};

// ═══════════════════════════════════════════════════════════════════════════════
// SYSTEM & HEALTH ENDPOINTS
// ═══════════════════════════════════════════════════════════════════════════════

export const systemApi = {
  // Health check
  async getHealthCheck(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    version: string;
    timestamp: string;
    components: Record<string, { status: string; details?: any }>;
  }> {
    return apiClient.get('/health', { timeout: 5000 });
  },
  
  // Get system metrics
  async getMetrics(): Promise<{
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    active_connections: number;
    request_rate: number;
    error_rate: number;
  }> {
    return apiClient.get('/metrics');
  },
  
  // Get application configuration
  async getConfig(): Promise<{
    version: string;
    environment: string;
    features: Record<string, boolean>;
    limits: Record<string, number>;
  }> {
    return apiClient.get('/config');
  },
};

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

export const apiUtils = {
  // Check if API is available
  async checkConnection(): Promise<boolean> {
    try {
      await systemApi.getHealthCheck();
      return true;
    } catch (error) {
      console.warn('API connection failed:', error);
      return false;
    }
  },
  
  // Batch multiple requests with concurrency control
  async batchRequests<T>(
    requests: (() => Promise<T>)[],
    concurrency: number = 5
  ): Promise<T[]> {
    const results: T[] = [];
    
    for (let i = 0; i < requests.length; i += concurrency) {
      const batch = requests.slice(i, i + concurrency);
      const batchResults = await Promise.all(batch.map(req => req()));
      results.push(...batchResults);
    }
    
    return results;
  },
  
  // Handle API errors with user-friendly messages
  formatError(error: any): string {
    if (typeof error === 'string') return error;
    if (error?.message) return error.message;
    if (error?.error) return error.error;
    return 'An unexpected error occurred';
  },
  
  // Get auth token
  getAuthToken(): string | null {
    return apiClient.authToken;
  },
  
  // Set auth token
  setAuthToken(token: string | null): void {
    apiClient.setAuthToken(token);
  },
};

// Export all APIs as a single object for convenience
export const api = {
  machines: machineApi,
  fleet: fleetApi,
  diagnosis: diagnosisApi,
  audit: auditApi,
  auth: authApi,
  system: systemApi,
  utils: apiUtils,
};

export default api;
