import React, { useEffect, useState, useCallback, useRef } from "react";
import { createRoot } from "react-dom/client";
import {
  LineChart, Line, AreaChart, Area, RadialBarChart, RadialBar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar
} from "recharts";
import {
  Activity, Thermometer, Gauge, Zap, RotateCw, AlertTriangle,
  CheckCircle, XCircle, Clock, Wrench, ChevronRight, Bot,
  BarChart2, History, Cpu, Search, RefreshCw, Send
} from "lucide-react";
import "./index.css";

const API = import.meta.env.VITE_API_BASE_URL || "https://industry-maintenance-slm.onrender.com";

// ── Types ──────────────────────────────────────────────────────────────────────
interface MachineCtx {
  engine_id: number;
  cycle: number;
  predicted_rul: number;
  health_score: number;
  risk_level: string;
  anomaly_score: number;
  anomaly_status: string;
  failure_probability_24h: number;
  failure_probability_48h: number;
}

interface Sensor {
  key: string;
  label: string;
  unit: string;
  icon: string;
  value: number;
  normal_min: number;
  normal_max: number;
  delta_pct: number;
  status: "normal" | "warning" | "critical";
}

interface SensorData {
  engine_id: number;
  sensors: Sensor[];
  trend: Record<string, number | string>[];
}

interface HistoryData {
  engine_id: number;
  machine_age_years: number;
  last_service_days_ago: number;
  total_maintenance_events: number;
  total_errors: number;
  error_log: { code: string; description: string; days_ago: number; count: number }[];
  maintenance_log: { action: string; days_ago: number; technician: string }[];
}

interface DiagnosisResult {
  engine_id: number;
  severity: string;
  diagnosis: string;
  evidence: string[];
  recommended_actions: string[];
  knowledge_evidence: { text: string; score: number }[];
}

interface FleetSummary {
  total_machines: number;
  critical: number;
  high: number;
  attention: number;
  normal: number;
  avg_health_score: number;
  avg_rul: number;
}

interface Toast {
  id: string;
  type: "success" | "error" | "info";
  message: string;
  duration?: number;
}

type TabType = "sensors" | "trends" | "history" | "copilot";
type RiskFilterType = "All" | "Critical" | "High" | "Attention" | "Normal";
type SortByType = "engine_id" | "lowest_rul" | "lowest_health";

// ── Icon map ───────────────────────────────────────────────────────────────────
const SENSOR_ICONS: Record<string, React.ReactNode> = {
  thermometer: <Thermometer size={18} />,
  gauge: <Gauge size={18} />,
  activity: <Activity size={18} />,
  zap: <Zap size={18} />,
  "rotate-cw": <RotateCw size={18} />,
};

// ── Toast System ──────────────────────────────────────────────────────────────
function ToastContainer({ toasts, onRemove }: { toasts: Toast[]; onRemove: (id: string) => void }) {
  return (
    <div className="toast-container">
      {toasts.map(t => (
        <div
          key={t.id}
          className={`toast ${t.type}`}
          onAnimationEnd={() => {
            const timeout = (t.duration || 3000) - 300;
            setTimeout(() => onRemove(t.id), timeout);
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            {t.type === "success" ? <CheckCircle size={16} color="var(--green)" />
              : t.type === "error" ? <AlertTriangle size={16} color="var(--red)" />
              : <Bot size={16} color="var(--accent)" />}
            <span>{t.message}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Utility ────────────────────────────────────────────────────────────────────
function timeStr() {
  return new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function riskColor(r: string) {
  if (r === "CRITICAL") return "var(--red)";
  if (r === "HIGH") return "var(--yellow)";
  if (r === "ATTENTION") return "#f97316";
  return "var(--green)";
}

function riskDotClass(r: string) {
  if (r === "CRITICAL") return "risk-dot-critical";
  if (r === "HIGH") return "risk-dot-high";
  if (r === "ATTENTION") return "risk-dot-attention";
  return "risk-dot-normal";
}

// ── Health Ring ────────────────────────────────────────────────────────────────
function HealthRing({ score }: { score: number }) {
  const color = score >= 70 ? "var(--green)" : score >= 40 ? "var(--yellow)" : "var(--red)";
  const data = [{ value: score }, { value: Math.max(0, 100 - score) }];
  return (
    <div className="health-ring-wrap">
      <ResponsiveContainer width={120} height={120}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={40}
            outerRadius={55}
            startAngle={90}
            endAngle={-270}
            dataKey="value"
            strokeWidth={0}
          >
            <Cell fill={color} />
            <Cell fill="var(--bg-surface)" />
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div style={{ marginTop: -68, textAlign: "center", pointerEvents: "none" }}>
        <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 20, fontWeight: 700, color }}>
          {score.toFixed(0)}
        </div>
        <div style={{ fontSize: 10, color: "var(--text-muted)" }}>/ 100</div>
      </div>
      <div style={{ marginTop: 52, fontSize: 11, color: "var(--text-muted)" }}>Health Score</div>
    </div>
  );
}

// ── Sensor Card ────────────────────────────────────────────────────────────────
function SensorCard({ s }: { s: Sensor }) {
  const range = s.normal_max - s.normal_min || 1;
  const pct = Math.min(100, Math.max(0, ((s.value - s.normal_min * 0.8) / (range * 1.5)) * 100));
  const icon = SENSOR_ICONS[s.icon] ?? <Activity size={18} />;
  const deltaUp = s.delta_pct > 1;
  const deltaDown = s.delta_pct < -1;

  return (
    <div className={`sensor-card status-${s.status}`}>
      <div className="sensor-top">
        <div className={`sensor-icon-wrap ${s.status}`}>{icon}</div>
        <span className={`sensor-status-pill ${s.status}`}>{s.status}</span>
      </div>
      <div className="sensor-name">{s.label}</div>
      <div className="sensor-value">
        {s.value.toLocaleString(undefined, { maximumFractionDigits: 1 })}
        <span className="unit">{s.unit}</span>
      </div>
      <div className="sensor-bar-wrap">
        <div className="sensor-bar-label">
          <span>{s.normal_min} {s.unit}</span>
          <span>Normal range</span>
          <span>{s.normal_max} {s.unit}</span>
        </div>
        <div className="sensor-bar-track">
          <div className={`sensor-bar-fill ${s.status}`} style={{ width: `${pct}%` }} />
        </div>
      </div>
      <div className={`sensor-delta ${deltaUp ? "up" : deltaDown ? "down" : "flat"}`}>
        {deltaUp ? "▲" : deltaDown ? "▼" : "—"}
        {" "}{Math.abs(s.delta_pct).toFixed(1)}% from nominal
      </div>
    </div>
  );
}

// ── Trend Chart ────────────────────────────────────────────────────────────────
function TrendChart({
  trend,
  keys,
  colors,
  title
}: {
  trend: Record<string, number | string>[];
  keys: string[];
  colors: string[];
  title: string;
}) {
  return (
    <div className="card" style={{ padding: "16px" }}>
      <div className="card-header">
        <div className="card-icon"><BarChart2 size={16} /></div>
        <span className="card-title">{title}</span>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={trend} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
          <defs>
            {keys.map((k, i) => (
              <linearGradient key={k} id={`grad-${k}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={colors[i]} stopOpacity={0.25} />
                <stop offset="95%" stopColor={colors[i]} stopOpacity={0} />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="hour"
            tick={{ fill: "var(--text-muted)", fontSize: 10 }}
            tickLine={false}
            interval={3}
          />
          <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} tickLine={false} />
          <Tooltip
            contentStyle={{
              background: "var(--bg-card)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12
            }}
            labelStyle={{ color: "var(--text-secondary)" }}
            itemStyle={{ color: "var(--text-primary)" }}
          />
          <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
          {keys.map((k, i) => (
            <Area
              key={k}
              type="monotone"
              dataKey={k}
              stroke={colors[i]}
              fill={`url(#grad-${k})`}
              strokeWidth={1.5}
              dot={false}
              name={k}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Failure Probability Bar ────────────────────────────────────────────────────
function FailureBar({ label, value }: { label: string; value: number }) {
  const pct = Math.min(100, value * 100);
  const color = pct > 60 ? "var(--red)" : pct > 30 ? "var(--yellow)" : "var(--green)";
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 5 }}>
        <span style={{ color: "var(--text-secondary)" }}>{label}</span>
        <span style={{ fontFamily: "'JetBrains Mono',monospace", fontWeight: 600, color }}>{pct.toFixed(1)}%</span>
      </div>
      <div style={{ height: 6, background: "var(--bg-input)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 3, transition: "width 0.8s ease" }} />
      </div>
    </div>
  );
}

// ── Diagnosis Panel ────────────────────────────────────────────────────────────
function DiagnosisPanel({
  result,
  loading,
  machineId,
  onApprove
}: {
  result: DiagnosisResult | null;
  loading: boolean;
  machineId?: number;
  onApprove?: (result: DiagnosisResult) => void;
}) {
  const [checkedActions, setCheckedActions] = useState<Set<number>>(new Set());
  const [approving, setApproving] = useState(false);

  const toggleAction = (index: number) => {
    const newChecked = new Set(checkedActions);
    if (newChecked.has(index)) {
      newChecked.delete(index);
    } else {
      newChecked.add(index);
    }
    setCheckedActions(newChecked);
  };

  const handleApprove = async () => {
    if (!machineId || !result) return;
    setApproving(true);
    try {
      const response = await fetch(`${API}/maintenance/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          engine_id: machineId,
          diagnosis: result.diagnosis,
          recommended_actions: result.recommended_actions,
          checked_actions: Array.from(checkedActions),
          timestamp: new Date().toISOString(),
        }),
      });
      if (response.ok) {
        if (onApprove) onApprove(result);
        setCheckedActions(new Set());
      }
    } catch (e) {
      console.error("Approval failed:", e);
    } finally {
      setApproving(false);
    }
  };

  if (loading) {
    return (
      <div className="diagnosis-card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 40, gap: 14 }}>
        <div className="spinner" style={{ width: 32, height: 32 }} />
        <div style={{ color: "var(--text-muted)", fontSize: 13 }}>AI is analysing the machine…</div>
        <div className="thinking-dots"><span>·</span><span>·</span><span>·</span></div>
      </div>
    );
  }
  if (!result) {
    return (
      <div className="diagnosis-card empty-state" style={{ minHeight: 200 }}>
        <Bot size={40} style={{ opacity: 0.3 }} />
        <span>Ask the AI Copilot a question to get started</span>
      </div>
    );
  }

  const sevColor = result.severity === "CRITICAL" ? "var(--red)"
    : result.severity === "HIGH" ? "var(--yellow)"
    : result.severity === "MEDIUM" ? "#f97316"
    : "var(--green)";
  const sevBg = result.severity === "CRITICAL" ? "rgba(239,68,68,0.15)"
    : result.severity === "HIGH" ? "rgba(245,158,11,0.15)"
    : result.severity === "MEDIUM" ? "rgba(249,115,22,0.15)"
    : "rgba(34,197,94,0.1)";
  const sevBorder = result.severity === "CRITICAL" ? "rgba(239,68,68,0.3)"
    : result.severity === "HIGH" ? "rgba(245,158,11,0.3)"
    : result.severity === "MEDIUM" ? "rgba(249,115,22,0.3)"
    : "rgba(34,197,94,0.25)";

  return (
    <div className="diagnosis-card">
      <div style={{
        background: sevBg,
        border: `1.5px solid ${sevBorder}`,
        borderRadius: "var(--r-lg) var(--r-lg) 0 0",
        padding: "16px 18px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 10, height: 10, borderRadius: "50%",
            background: sevColor,
            boxShadow: `0 0 8px ${sevColor}`,
          }} />
          <div>
            <div style={{ fontWeight: 700, color: sevColor, fontSize: 13, textTransform: "uppercase", letterSpacing: "0.07em" }}>
              {result.severity} SEVERITY
            </div>
            <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 2 }}>
              {result.severity === "CRITICAL" ? "Immediate intervention required"
                : result.severity === "HIGH" ? "Schedule urgent maintenance"
                : result.severity === "MEDIUM" ? "Plan maintenance within 7 days"
                : "Continue monitoring"}
            </div>
          </div>
        </div>
        <Wrench size={20} color={sevColor} opacity={0.6} />
      </div>

      <div className="diagnosis-body">
        <div className="diagnosis-section">
          <div className="diagnosis-section-title">Diagnosis Summary</div>
          <div className="diagnosis-text" style={{ background: "var(--bg-surface)", padding: "12px 14px", borderRadius: "var(--r-sm)", borderLeft: `3px solid ${sevColor}` }}>
            {result.diagnosis}
          </div>
        </div>

        {result.evidence && result.evidence.length > 0 && (
          <div className="diagnosis-section">
            <div className="diagnosis-section-title">Evidence Breakdown</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {result.evidence.map((e, i) => (
                <div key={i} style={{
                  background: "var(--bg-surface)",
                  border: "1px solid var(--border-bright)",
                  borderRadius: "var(--r-md)",
                  padding: "8px 12px",
                  fontSize: 12,
                  color: "var(--text-secondary)",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}>
                  <Activity size={13} color="var(--accent)" />
                  {e}
                </div>
              ))}
            </div>
          </div>
        )}

        {result.recommended_actions && result.recommended_actions.length > 0 && (
          <div className="diagnosis-section">
            <div className="diagnosis-section-title">Action Checklist</div>
            <div style={{ background: "var(--bg-surface)", borderRadius: "var(--r-md)", border: "1px solid var(--border)", overflow: "hidden" }}>
              {result.recommended_actions.map((a, i) => (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 10,
                    padding: "12px 14px",
                    borderBottom: i < result.recommended_actions.length - 1 ? "1px solid var(--border)" : "none",
                    cursor: "pointer",
                    transition: "background 0.2s",
                    background: checkedActions.has(i) ? "rgba(34,197,94,0.08)" : "transparent",
                  }}
                  onClick={() => toggleAction(i)}
                >
                  <input
                    type="checkbox"
                    checked={checkedActions.has(i)}
                    onChange={() => toggleAction(i)}
                    style={{
                      width: 18, height: 18, marginTop: 2,
                      cursor: "pointer", accentColor: "var(--green)",
                    }}
                  />
                  <span style={{
                    flex: 1, fontSize: 13, color: "var(--text-primary)",
                    textDecoration: checkedActions.has(i) ? "line-through" : "none",
                    opacity: checkedActions.has(i) ? 0.6 : 1,
                  }}>
                    {a}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {result.knowledge_evidence && result.knowledge_evidence.length > 0 && (
          <div className="diagnosis-section">
            <div className="diagnosis-section-title">Related SOP Knowledge</div>
            {result.knowledge_evidence.slice(0, 2).map((k, i) => (
              <div key={i} className="rag-chunk">
                <div className="rag-score">
                  <Search size={10} /> Relevance: {(k.score * 100).toFixed(0)}%
                </div>
                <div>{k.text.substring(0, 200)}{k.text.length > 200 ? "…" : ""}</div>
              </div>
            ))}
          </div>
        )}

        <div style={{ marginTop: 16, paddingTop: 14, borderTop: "1px solid var(--border)" }}>
          <button
            onClick={handleApprove}
            disabled={approving || result.recommended_actions.length === 0}
            style={{
              width: "100%",
              padding: "12px 16px",
              borderRadius: "var(--r-md)",
              border: "none",
              background: sevColor,
              color: "#fff",
              fontWeight: 600,
              fontSize: 13,
              cursor: approving ? "not-allowed" : "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 8,
              opacity: approving ? 0.6 : 1,
              transition: "all 0.2s",
            }}
          >
            {approving ? (
              <>
                <div className="spinner" style={{ width: 14, height: 14 }} />
                Approving…
              </>
            ) : (
              <>
                <CheckCircle size={16} />
                Approve & Log Maintenance
              </>
            )}
          </button>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 6, textAlign: "center" }}>
            Review actions above, then approve to log in the maintenance system
          </div>
        </div>
      </div>
    </div>
  );
}

// ── History Tab ────────────────────────────────────────────────────────────────
function HistoryTab({ hist }: { hist: HistoryData | null }) {
  if (!hist) return <div className="empty-state"><History size={40} /><span>No history loaded</span></div>;

  const errorByCode = hist.error_log.reduce<Record<string, number>>((acc, e) => {
    acc[e.code] = (acc[e.code] || 0) + e.count;
    return acc;
  }, {});
  const errorChartData = Object.entries(errorByCode).map(([code, count]) => ({ code, count }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
        {[
          { label: "Machine Age", val: `${hist.machine_age_years} yrs`, icon: <Cpu size={18} /> },
          { label: "Last Service", val: `${hist.last_service_days_ago}d ago`, icon: <Wrench size={18} /> },
          { label: "Maintenance Events", val: hist.total_maintenance_events, icon: <CheckCircle size={18} /> },
          { label: "Total Errors", val: hist.total_errors, icon: <AlertTriangle size={18} /> },
        ].map(({ label, val, icon }) => (
          <div key={label} className="card" style={{ padding: "14px 18px", display: "flex", gap: 12, alignItems: "center" }}>
            <div className="card-icon">{icon}</div>
            <div>
              <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.07em" }}>{label}</div>
              <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 18, fontWeight: 700 }}>{val}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="history-grid">
        <div className="card">
          <div className="card-header">
            <div className="card-icon" style={{ background: "rgba(239,68,68,0.1)", color: "var(--red)" }}>
              <AlertTriangle size={16} />
            </div>
            <span className="card-title">Error Log</span>
          </div>
          {hist.error_log.length === 0 ? (
            <div style={{ color: "var(--text-muted)", fontSize: 13 }}>No errors recorded</div>
          ) : (
            hist.error_log.map((e, i) => (
              <div key={i} className="timeline-item">
                <div className="timeline-dot error" />
                <div>
                  <div className="timeline-text">
                    <span style={{ fontFamily: "'JetBrains Mono',monospace", color: "var(--red)", fontWeight: 600 }}>{e.code}</span>
                    {" — "}{e.description}
                  </div>
                  <div className="timeline-meta">{e.days_ago}d ago · occurred {e.count}×</div>
                </div>
              </div>
            ))
          )}
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-icon"><Wrench size={16} /></div>
            <span className="card-title">Maintenance Log</span>
          </div>
          {hist.maintenance_log.length === 0 ? (
            <div style={{ color: "var(--text-muted)", fontSize: 13 }}>No maintenance recorded</div>
          ) : (
            hist.maintenance_log.map((m, i) => (
              <div key={i} className="timeline-item">
                <div className="timeline-dot" />
                <div>
                  <div className="timeline-text">{m.action}</div>
                  <div className="timeline-meta">{m.days_ago}d ago · {m.technician}</div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {errorChartData.length > 0 && (
        <div className="card" style={{ padding: "16px" }}>
          <div className="card-header">
            <div className="card-icon" style={{ background: "rgba(239,68,68,0.1)", color: "var(--red)" }}>
              <BarChart2 size={16} />
            </div>
            <span className="card-title">Error Frequency</span>
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={errorChartData} margin={{ top: 4, right: 8, bottom: 4, left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="code" tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} />
              <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} />
              <Tooltip
                contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
              />
              <Bar dataKey="count" fill="var(--red)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ── Fleet Summary Bar ──────────────────────────────────────────────────────────
function FleetSummaryPanel({ fleet }: { fleet: FleetSummary | null }) {
  if (!fleet) return null;
  return (
    <div className="fleet-summary">
      <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.07em" }}>
        Fleet Overview
      </div>
      {[
        { label: "Total", val: fleet.total_machines, color: "var(--text-primary)" },
        { label: "Critical", val: fleet.critical, color: "var(--red)" },
        { label: "High Risk", val: fleet.high, color: "var(--yellow)" },
        { label: "Attention", val: fleet.attention, color: "#f97316" },
        { label: "Normal", val: fleet.normal, color: "var(--green)" },
      ].map(({ label, val, color }) => (
        <div key={label} className="fleet-row">
          <span className="fleet-label">{label}</span>
          <span className="fleet-val" style={{ color }}>{val}</span>
        </div>
      ))}
      <div style={{ borderTop: "1px solid var(--border)", marginTop: 8, paddingTop: 8 }}>
        <div className="fleet-row">
          <span className="fleet-label">Avg Health</span>
          <span className="fleet-val" style={{ color: fleet.avg_health_score >= 60 ? "var(--green)" : "var(--yellow)" }}>
            {fleet.avg_health_score}/100
          </span>
        </div>
        <div className="fleet-row">
          <span className="fleet-label">Avg RUL</span>
          <span className="fleet-val">{fleet.avg_rul} cyc</span>
        </div>
      </div>
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────────
const QUICK_QUESTIONS = [
  "Should this turbofan engine be taken offline for immediate inspection?",
  "What are the LOTO (lockout-tagout) and safety protocols before maintenance?",
  "Which sensor channels (HPC temp, pressure, fan speed) show strongest degradation?",
  "Is there high-pressure compressor (HPC) thermal runaway or exhaust gas overtemp?",
  "Is there core rotational instability or bearing assembly degradation?",
  "Is HPC outlet pressure dropping relative to fuel flow baseline?",
  "What is the remaining useful life (RUL) and is degradation accelerating?",
  "What is the calculated failure risk for the next 24 and 48 operational cycles?",
  "What anomalous drift occurred over the last 24 engine cycles?",
  "How do current sensor readings compare to the baseline at cycle 1?",
  "What bore-scope and non-destructive inspection scope is required per SOP §4?",
  "What maintenance actions are required and what is the urgency?",
  "Why is this turbofan engine at its current risk level?",
  "How does this engine compare to similar engines in the fleet by health and RUL?",
  "What is the overall fleet risk status and which engines need priority?",
  "Is this a one-time anomaly or progressive degradation requiring intervention?",
  "What is the full diagnostic verdict and recommended next actions?",
];

const AUTO_REFRESH_MS = 30_000;

export default function App() {
  const [machineIds, setMachineIds] = useState<number[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [ctx, setCtx] = useState<MachineCtx | null>(null);
  const [sensorData, setSensorData] = useState<SensorData | null>(null);
  const [histData, setHistData] = useState<HistoryData | null>(null);
  const [fleet, setFleet] = useState<FleetSummary | null>(null);
  const [tab, setTab] = useState<TabType>("sensors");
  const [question, setQuestion] = useState(QUICK_QUESTIONS[0]);
  const [diagnosis, setDiagnosis] = useState<DiagnosisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [riskMap, setRiskMap] = useState<Record<number, string>>({});
  const [machineDetailsMap, setMachineDetailsMap] = useState<Record<number, MachineCtx>>({});

  const [riskFilter, setRiskFilter] = useState<RiskFilterType>("All");
  const [sortBy, setSortBy] = useState<SortByType>("engine_id");
  const [search, setSearch] = useState("");
  const [clock, setClock] = useState(timeStr());
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [refreshing, setRefreshing] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const selectedIdRef = useRef<number | null>(null);

  const showToast = (message: string, type: "success" | "error" | "info" = "info", duration: number = 3000) => {
    const id = Math.random().toString(36);
    const newToast: Toast = { id, message, type, duration };
    setToasts(prev => [...prev, newToast]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, duration);
  };

  useEffect(() => {
    const timer = setInterval(() => setClock(timeStr()), 1000);
    return () => clearInterval(timer);
  }, []);

  const selectMachine = useCallback(async (id: number) => {
    selectedIdRef.current = id;
    setSelectedId(id);
    setCtx(null);
    setSensorData(null);
    setHistData(null);
    setDiagnosis(null);

    const [ctxR, sensorR, histR] = await Promise.all([
      fetch(`${API}/machines/${id}`).then(r => r.ok ? r.json() : null),
      fetch(`${API}/machines/${id}/sensors`).then(r => r.ok ? r.json() : null),
      fetch(`${API}/machines/${id}/history`).then(r => r.ok ? r.json() : null),
    ]);

    if (selectedIdRef.current !== id) return;
    setCtx(ctxR);
    setSensorData(sensorR);
    setHistData(histR);

    if (ctxR) {
      setRiskMap(prev => ({ ...prev, [id]: ctxR.risk_level }));
      setMachineDetailsMap(prev => ({ ...prev, [id]: ctxR }));
    }
  }, []);

  useEffect(() => {
    fetch(`${API}/fleet/summary`).then(r => r.json()).then(setFleet).catch(() => {});
    fetch(`${API}/machines`).then(r => r.json()).then(d => {
      const ids: number[] = d.machines || [];
      setMachineIds(ids);
      if (ids.length > 0) selectMachine(ids[0]);

      ids.slice(0, 50).forEach(id => {
        fetch(`${API}/machines/${id}`)
          .then(r => r.ok ? r.json() : null)
          .then(data => {
            if (data) {
              setRiskMap(prev => ({ ...prev, [id]: data.risk_level }));
              setMachineDetailsMap(prev => ({ ...prev, [id]: data }));
            }
          })
          .catch(() => {});
      });
    });
  }, [selectMachine]);

  useEffect(() => {
    const t = setInterval(async () => {
      const id = selectedIdRef.current;
      if (!id) return;
      setRefreshing(true);
      try {
        const [ctxR, sensorR] = await Promise.all([
          fetch(`${API}/machines/${id}`).then(r => r.ok ? r.json() : null),
          fetch(`${API}/machines/${id}/sensors`).then(r => r.ok ? r.json() : null),
        ]);
        if (selectedIdRef.current !== id) return;
        if (ctxR) {
          setCtx(ctxR);
          setRiskMap(prev => ({ ...prev, [id]: ctxR.risk_level }));
          setMachineDetailsMap(prev => ({ ...prev, [id]: ctxR }));
        }
        if (sensorR) setSensorData(sensorR);
        fetch(`${API}/fleet/summary`).then(r => r.json()).then(setFleet).catch(() => {});
        setLastRefresh(new Date());
      } finally {
        setRefreshing(false);
      }
    }, AUTO_REFRESH_MS);
    return () => clearInterval(t);
  }, []);

  const investigate = async () => {
    if (!selectedId || !question.trim()) return;
    setLoading(true);
    setDiagnosis(null);
    try {
      const r = await fetch(`${API}/machines/${selectedId}/investigate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      setDiagnosis(await r.json());
      setTab("copilot");
    } finally {
      setLoading(false);
    }
  };

  const handleManualRefresh = () => {
    if (selectedId) selectMachine(selectedId);
    setLastRefresh(new Date());
  };

  const handleSortChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSortBy(e.target.value as SortByType);
  };

  const riskFilters: RiskFilterType[] = ["All", "Critical", "High", "Attention", "Normal"];

  const filteredIds = machineIds.filter(id => {
    if (search !== "" && !String(id).includes(search)) return false;

    const risk = riskMap[id] ?? "NORMAL";
    if (riskFilter === "All") return true;
    if (riskFilter === "Critical" && risk !== "CRITICAL") return false;
    if (riskFilter === "High" && risk !== "HIGH") return false;
    if (riskFilter === "Attention" && risk !== "ATTENTION") return false;
    if (riskFilter === "Normal" && risk !== "NORMAL") return false;

    return true;
  });

  if (sortBy === "lowest_rul") {
    filteredIds.sort((a, b) => {
      const rulA = machineDetailsMap[a]?.predicted_rul ?? (a === selectedId ? (ctx?.predicted_rul ?? 999) : 999);
      const rulB = machineDetailsMap[b]?.predicted_rul ?? (b === selectedId ? (ctx?.predicted_rul ?? 999) : 999);
      return rulA - rulB;
    });
  } else if (sortBy === "lowest_health") {
    filteredIds.sort((a, b) => {
      const healthA = machineDetailsMap[a]?.health_score ?? (a === selectedId ? (ctx?.health_score ?? 100) : 100);
      const healthB = machineDetailsMap[b]?.health_score ?? (b === selectedId ? (ctx?.health_score ?? 100) : 100);
      return healthA - healthB;
    });
  } else {
    filteredIds.sort((a, b) => a - b);
  }

  const criticalSensors = sensorData?.sensors.filter(s => s.status === "critical") ?? [];

  return (
    <div className="app-layout">
      <ToastContainer toasts={toasts} onRemove={(id) => setToasts(prev => prev.filter(t => t.id !== id))} />

      {/* ── Top Bar ── */}
      <header className="topbar">
        <div className="topbar-logo">
          <Cpu size={20} />
          Industrial Maintenance Copilot
          <span className="topbar-badge">SLM v2</span>
        </div>
        <div className="topbar-spacer" />
        {refreshing && (
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--accent)" }}>
            <div className="spinner" style={{ width: 12, height: 12 }} /> Auto-refreshing…
          </div>
        )}
        {!refreshing && (
          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
            Last updated: {lastRefresh.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
          </div>
        )}
        {criticalSensors.length > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--red)", background: "var(--red-bg)", padding: "4px 12px", borderRadius: 20, border: "1px solid rgba(239,68,68,0.3)" }}>
            <AlertTriangle size={13} /> {criticalSensors.length} critical sensor{criticalSensors.length > 1 ? "s" : ""}
          </div>
        )}
        <div className="status-dot" />
        <span className="topbar-time">{clock}</span>
      </header>

      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <FleetSummaryPanel fleet={fleet} />

        <div className="sidebar-section-label" style={{ marginTop: 16 }}>Risk Filter</div>
        <div style={{ display: "flex", gap: 6, padding: "0 8px", flexWrap: "wrap" }}>
          {riskFilters.map(level => (
            <button
              key={level}
              onClick={() => setRiskFilter(level)}
              style={{
                padding: "6px 12px",
                fontSize: 11,
                fontWeight: 600,
                borderRadius: 16,
                border: `1.5px solid ${riskFilter === level ? riskColor(level === "All" ? "NORMAL" : level.toUpperCase()) : "var(--border)"}`,
                background: riskFilter === level ? `${riskColor(level === "All" ? "NORMAL" : level.toUpperCase())}12` : "var(--bg-input)",
                color: riskFilter === level ? riskColor(level === "All" ? "NORMAL" : level.toUpperCase()) : "var(--text-secondary)",
                cursor: "pointer",
                transition: "all 0.2s ease",
              }}
            >
              {level}
            </button>
          ))}
        </div>

        <div className="sidebar-section-label" style={{ marginTop: 12 }}>Sort By</div>
        <select
          value={sortBy}
          onChange={handleSortChange}
          style={{
            margin: "0 8px 12px",
            padding: "8px 12px",
            borderRadius: 8,
            border: "1px solid var(--border)",
            background: "var(--bg-input)",
            color: "var(--text-primary)",
            fontSize: 12,
            cursor: "pointer",
            width: "calc(100% - 16px)",
          }}
        >
          <option value="engine_id">Engine ID</option>
          <option value="lowest_rul">Lowest RUL</option>
          <option value="lowest_health">Lowest Health</option>
        </select>

        <div className="sidebar-section-label">Machines ({filteredIds.length})</div>
        <input
          className="machine-search"
          placeholder="Search engine ID…"
          value={search}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearch(e.target.value)}
        />
        <div style={{ overflowY: "auto", flex: 1 }}>
          {filteredIds.map(id => {
            const isSelected = id === selectedId;
            const risk = riskMap[id] ?? "NORMAL";
            const riskDot = riskDotClass(risk);
            const mData = machineDetailsMap[id] || (isSelected ? ctx : null);
            const healthScore = mData?.health_score ?? 50;
            const rul = mData?.predicted_rul ?? 0;
            return (
              <div
                key={id}
                className={`machine-item ${isSelected ? "active" : ""}`}
                onClick={() => selectMachine(id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  padding: "8px 12px",
                  cursor: "pointer",
                  borderRadius: 6,
                  marginBottom: 4,
                  background: isSelected ? "rgba(14,165,233,0.1)" : "transparent",
                  border: isSelected ? "1px solid var(--accent)" : "transparent",
                  transition: "all 0.15s ease",
                }}
              >
                <Cpu size={13} color={isSelected ? "var(--accent)" : "var(--text-muted)"} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 12, fontFamily: "'JetBrains Mono',monospace", fontWeight: 600, color: "var(--text-primary)" }}>
                    ENG-{String(id).padStart(3, "0")}
                  </div>
                  <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
                    RUL: {rul.toFixed(0)} • H: {healthScore.toFixed(0)}
                  </div>
                </div>
                <span className={`machine-risk-dot ${riskDot}`} />
                {isSelected && <ChevronRight size={12} color="var(--accent)" />}
              </div>
            );
          })}
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="main-content">
        {!ctx ? (
          <div className="empty-state" style={{ minHeight: "60vh" }}>
            <Cpu size={48} />
            <span>Select a machine to begin</span>
          </div>
        ) : (
          <>
            {/* Machine Header */}
            <div className="machine-header">
              <div>
                <div className="machine-title">ENG-{String(ctx.engine_id).padStart(3, "0")}</div>
                <div className="machine-subtitle">Operational Cycle: {ctx.cycle}</div>
              </div>
              <div style={{ flex: 1 }} />
              <span className={`risk-badge risk-${ctx.risk_level}`}>{ctx.risk_level}</span>

              {/* RUL Urgency Progress Gauge */}
              <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 180 }}>
                <div className="kpi-label" style={{ marginBottom: 2 }}>RUL Status</div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4, color: "var(--text-muted)" }}>
                      <span>{ctx.predicted_rul.toFixed(0)} cycles left</span>
                      <span style={{ fontFamily: "'JetBrains Mono',monospace", fontWeight: 600 }}>
                        {ctx.predicted_rul <= 25 ? "🔴 CRITICAL" : ctx.predicted_rul <= 50 ? "🟠 HIGH" : ctx.predicted_rul <= 75 ? "🟡 ATTENTION" : "🟢 NORMAL"}
                      </span>
                    </div>
                    <div style={{ height: 8, background: "var(--bg-input)", borderRadius: 4, overflow: "hidden", border: "1px solid var(--border)" }}>
                      <div
                        style={{
                          height: "100%",
                          width: `${Math.max(5, Math.min(100, (ctx.predicted_rul / 125) * 100))}%`,
                          background: ctx.predicted_rul <= 25 ? "var(--red)" : ctx.predicted_rul <= 50 ? "var(--yellow)" : ctx.predicted_rul <= 75 ? "#f97316" : "var(--green)",
                          borderRadius: 4,
                          transition: "width 0.6s ease, background 0.6s ease"
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="kpi">
                <div className="kpi-label">Health Score</div>
                <div className="kpi-value" style={{ color: ctx.health_score >= 70 ? "var(--green)" : ctx.health_score >= 40 ? "var(--yellow)" : "var(--red)" }}>
                  {ctx.health_score.toFixed(1)}
                </div>
                <div className="kpi-sub">/ 100</div>
              </div>

              <div className="kpi">
                <div className="kpi-label">Anomaly Score</div>
                <div className="kpi-value" style={{ color: ctx.anomaly_score > 0.5 ? "var(--red)" : ctx.anomaly_score > 0.25 ? "var(--yellow)" : "var(--green)" }}>
                  {ctx.anomaly_score.toFixed(3)}
                </div>
                <div className="kpi-sub">{ctx.anomaly_status}</div>
              </div>

              {/* 24h & 48h Failure Risk Indicators */}
              <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 160 }}>
                <FailureBar label="Failure P(24h)" value={ctx.failure_probability_24h} />
                <FailureBar label="Failure P(48h)" value={ctx.failure_probability_48h} />
              </div>
            </div>

            {/* Tabs */}
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <div className="tabs">
                {[
                  { id: "sensors" as TabType, label: "Sensors", icon: <Activity size={14} /> },
                  { id: "trends" as TabType,  label: "24h Trends", icon: <BarChart2 size={14} /> },
                  { id: "history" as TabType, label: "History", icon: <History size={14} /> },
                  { id: "copilot" as TabType, label: "AI Copilot", icon: <Bot size={14} /> },
                ].map(t => (
                  <button
                    key={t.id}
                    className={`tab ${tab === t.id ? "active" : ""}`}
                    onClick={() => setTab(t.id)}
                  >
                    {t.icon}{t.label}
                  </button>
                ))}
              </div>
              <button
                className="btn btn-primary"
                style={{ marginLeft: "auto" }}
                onClick={handleManualRefresh}
                disabled={refreshing}
              >
                <RefreshCw size={14} style={{ animation: refreshing ? "spin 0.7s linear infinite" : "none" }} /> Refresh
              </button>
            </div>

            {/* ── Sensors Tab ── */}
            {tab === "sensors" && sensorData && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div className="sensor-grid">
                  {sensorData.sensors.map(s => <SensorCard key={s.key} s={s} />)}
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "140px 1fr", gap: 16 }}>
                  <div className="card" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <HealthRing score={ctx.health_score} />
                  </div>
                  <div className="card" style={{ padding: "16px" }}>
                    <div className="card-header">
                      <div className="card-icon"><Activity size={16} /></div>
                      <span className="card-title">Sensor Status Summary</span>
                    </div>
                    <ResponsiveContainer width="100%" height={120}>
                      <BarChart
                        data={sensorData.sensors.map(s => ({
                          name: s.label.slice(0, 5),
                          value: s.value,
                          normal_min: s.normal_min,
                          normal_max: s.normal_max,
                        }))}
                        margin={{ top: 4, right: 8, bottom: 0, left: -16 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 10 }} tickLine={false} />
                        <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} tickLine={false} />
                        <Tooltip
                          contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
                        />
                        <Bar
                          dataKey="value"
                          radius={[4, 4, 0, 0]}
                          fill="var(--accent)"
                          label={false}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                    <div style={{ marginTop: 12, display: "flex", gap: 20, flexWrap: "wrap" }}>
                      {[
                        { label: "Normal", count: sensorData.sensors.filter(s => s.status === "normal").length, color: "var(--green)" },
                        { label: "Warning", count: sensorData.sensors.filter(s => s.status === "warning").length, color: "var(--yellow)" },
                        { label: "Critical", count: criticalSensors.length, color: "var(--red)" },
                      ].map(({ label, count, color }) => (
                        <div key={label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                          <div style={{ width: 10, height: 10, borderRadius: 2, background: color }} />
                          <span style={{ color: "var(--text-secondary)" }}>{label}:</span>
                          <span style={{ fontWeight: 700, color }}>{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ── Trends Tab ── */}
            {tab === "trends" && sensorData && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div className="chart-row">
                  <TrendChart
                    trend={sensorData.trend}
                    keys={["temperature", "pressure"]}
                    colors={["#ef4444", "#3b82f6"]}
                    title="Temperature & Pressure (24h)"
                  />
                  <TrendChart
                    trend={sensorData.trend}
                    keys={["vibration", "current"]}
                    colors={["#f59e0b", "#8b5cf6"]}
                    title="Vibration & Current (24h)"
                  />
                </div>
                <div className="chart-row">
                  <TrendChart
                    trend={sensorData.trend}
                    keys={["voltage"]}
                    colors={["#22c55e"]}
                    title="Voltage (24h)"
                  />
                  <TrendChart
                    trend={sensorData.trend}
                    keys={["rotation"]}
                    colors={["#06b6d4"]}
                    title="Rotation Speed (24h)"
                  />
                </div>
              </div>
            )}

            {/* ── History Tab ── */}
            {tab === "history" && <HistoryTab hist={histData} />}

            {/* ── Copilot Tab ── */}
            {tab === "copilot" && (
              <div className="copilot-wrap">
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div className="card">
                    <div className="card-header">
                      <div className="card-icon"><Bot size={16} /></div>
                      <span className="card-title">Ask the AI Copilot</span>
                    </div>
                    <div className="question-chips">
                      {QUICK_QUESTIONS.map(q => (
                        <button key={q} className="chip" onClick={() => setQuestion(q)}>{q}</button>
                      ))}
                    </div>
                    <div className="copilot-input-row">
                      <textarea
                        className="copilot-textarea"
                        value={question}
                        onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setQuestion(e.target.value)}
                        placeholder="Ask about this machine…"
                        onKeyDown={(e: React.KeyboardEvent<HTMLTextAreaElement>) => {
                          if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            investigate();
                          }
                        }}
                      />
                    </div>
                    <div style={{ marginTop: 8 }}>
                      <button className="btn btn-primary" onClick={investigate} disabled={loading}>
                        {loading ? <><div className="spinner" />Analysing…</> : <><Send size={14} />Investigate</>}
                      </button>
                    </div>
                  </div>

                  <div className="card">
                    <div className="card-header">
                      <div className="card-icon"><Cpu size={16} /></div>
                      <span className="card-title">Machine Context</span>
                    </div>
                    {[
                      { k: "Engine ID", v: `ENG-${String(ctx.engine_id).padStart(3, "0")}` },
                      { k: "Risk Level", v: ctx.risk_level, color: riskColor(ctx.risk_level) },
                      { k: "Health Score", v: `${ctx.health_score.toFixed(1)} / 100` },
                      { k: "Predicted RUL", v: `${ctx.predicted_rul.toFixed(1)} cycles` },
                      { k: "Anomaly Status", v: ctx.anomaly_status },
                      { k: "Failure P(24h)", v: `${(ctx.failure_probability_24h * 100).toFixed(1)}%` },
                      ...(histData ? [
                        { k: "Machine Age", v: `${histData.machine_age_years} years` },
                        { k: "Last Service", v: `${histData.last_service_days_ago} days ago` },
                        { k: "Total Errors", v: String(histData.total_errors) },
                      ] : []),
                    ].map(({ k, v, color }) => (
                      <div key={k} className="stat-row">
                        <span className="stat-key">{k}</span>
                        <span className="stat-val" style={color ? { color } : {}}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <DiagnosisPanel
                  result={diagnosis}
                  loading={loading}
                  machineId={selectedId ?? undefined}
                  onApprove={(result) => showToast(`✓ Maintenance plan approved for Engine ${result.engine_id}`, "success")}
                />
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);







// import React, { useEffect, useState, useCallback, useRef } from "react";
// import { createRoot } from "react-dom/client";
// import {
//   LineChart, Line, AreaChart, Area, RadialBarChart, RadialBar,
//   XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
//   PieChart, Pie, Cell, BarChart, Bar
// } from "recharts";
// import {
//   Activity, Thermometer, Gauge, Zap, RotateCw, AlertTriangle,
//   CheckCircle, XCircle, Clock, Wrench, ChevronRight, Bot,
//   BarChart2, History, Cpu, Search, RefreshCw, Send
// } from "lucide-react";
// import "./index.css";

// const API = "http://localhost:8000";

// // ── Types ──────────────────────────────────────────────────────────────────────
// interface MachineCtx {
//   engine_id: number; cycle: number; predicted_rul: number;
//   health_score: number; risk_level: string; anomaly_score: number;
//   anomaly_status: string; failure_probability_24h: number; failure_probability_48h: number;
// }
// interface Sensor {
//   key: string; label: string; unit: string; icon: string;
//   value: number; normal_min: number; normal_max: number;
//   delta_pct: number; status: "normal" | "warning" | "critical";
// }
// interface SensorData { engine_id: number; sensors: Sensor[]; trend: Record<string, number | string>[]; }
// interface HistoryData {
//   engine_id: number; machine_age_years: number; last_service_days_ago: number;
//   total_maintenance_events: number; total_errors: number;
//   error_log: { code: string; description: string; days_ago: number; count: number }[];
//   maintenance_log: { action: string; days_ago: number; technician: string }[];
// }
// interface DiagnosisResult {
//   engine_id: number; severity: string; diagnosis: string;
//   evidence: string[]; recommended_actions: string[];
//   knowledge_evidence: { text: string; score: number }[];
// }
// interface FleetSummary {
//   total_machines: number; critical: number; high: number;
//   attention: number; normal: number;
//   avg_health_score: number; avg_rul: number;
// }
// interface Toast {
//   id: string;
//   type: "success" | "error" | "info";
//   message: string;
//   duration?: number;
// }

// // ── Icon map ───────────────────────────────────────────────────────────────────
// const SENSOR_ICONS: Record<string, React.ReactNode> = {
//   thermometer: <Thermometer size={18} />,
//   gauge: <Gauge size={18} />,
//   activity: <Activity size={18} />,
//   zap: <Zap size={18} />,
//   "rotate-cw": <RotateCw size={18} />,
// };

// const STATUS_COLORS: Record<string, string> = {
//   normal: "var(--green)", warning: "var(--yellow)", critical: "var(--red)"
// };

// const TREND_COLORS = ["#0ea5e9","#22c55e","#f59e0b","#a78bfa","#ec4899","#06b6d4"];

// // ── Toast System ──────────────────────────────────────────────────────────────
// function ToastContainer({ toasts, onRemove }: { toasts: Toast[]; onRemove: (id: string) => void }) {
//   return (
//     <div className="toast-container">
//       {toasts.map(t => (
//         <div key={t.id} className={`toast ${t.type}`}
//           onAnimationEnd={() => { setTimeout(() => onRemove(t.id), t.duration || 3000 - 300); }}>
//           <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
//             {t.type === "success" ? <CheckCircle size={16} color="var(--green)" />
//               : t.type === "error" ? <AlertTriangle size={16} color="var(--red)" />
//               : <Bot size={16} color="var(--accent)" />}
//             <span>{t.message}</span>
//           </div>
//         </div>
//       ))}
//     </div>
//   );
// }

// // ── Skeleton Loaders ───────────────────────────────────────────────────────────
// function SkeletonSensorGrid() {
//   return (
//     <div className="skeleton-sensor-grid">
//       {[...Array(6)].map((_, i) => (
//         <div key={i} className="skeleton-sensor-card">
//           <div style={{ height: 24 }} />
//           <div style={{ height: 20, width: "70%" }} />
//           <div style={{ height: 12, width: "50%" }} />
//           <div style={{ height: 12 }} />
//           <div style={{ height: 8, width: "40%" }} />
//         </div>
//       ))}
//     </div>
//   );
// }

// function SkeletonCard() {
//   return (
//     <div className="skeleton-card">
//       <div className="skeleton skeleton-line" />
//       <div className="skeleton skeleton-line medium" />
//       <div className="skeleton skeleton-line short" />
//       <div className="skeleton skeleton-line medium" />
//     </div>
//   );
// }
// const SENSOR_KEYS = ["temperature","pressure","vibration","voltage","rotation","current"];

// // ── Utility ────────────────────────────────────────────────────────────────────
// function timeStr() {
//   return new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
// }

// function riskColor(r: string) {
//   if (r === "CRITICAL") return "var(--red)";
//   if (r === "HIGH")     return "var(--yellow)";
//   if (r === "ATTENTION") return "#f97316";
//   return "var(--green)";
// }

// function riskDotClass(r: string) {
//   if (r === "CRITICAL") return "risk-dot-critical";
//   if (r === "HIGH")     return "risk-dot-high";
//   if (r === "ATTENTION") return "risk-dot-attention";
//   return "risk-dot-normal";
// }

// // ── Health Ring ────────────────────────────────────────────────────────────────
// function HealthRing({ score }: { score: number }) {
//   const color = score >= 70 ? "var(--green)" : score >= 40 ? "var(--yellow)" : "var(--red)";
//   const data = [{ value: score }, { value: 100 - score }];
//   return (
//     <div className="health-ring-wrap">
//       <ResponsiveContainer width={120} height={120}>
//         <PieChart>
//           <Pie data={data} cx="50%" cy="50%" innerRadius={40} outerRadius={55}
//             startAngle={90} endAngle={-270} dataKey="value" strokeWidth={0}>
//             <Cell fill={color} />
//             <Cell fill="var(--bg-surface)" />
//           </Pie>
//         </PieChart>
//       </ResponsiveContainer>
//       <div style={{ marginTop: -68, textAlign: "center", pointerEvents: "none" }}>
//         <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 20, fontWeight: 700, color }}>
//           {score.toFixed(0)}
//         </div>
//         <div style={{ fontSize: 10, color: "var(--text-muted)" }}>/ 100</div>
//       </div>
//       <div style={{ marginTop: 52, fontSize: 11, color: "var(--text-muted)" }}>Health Score</div>
//     </div>
//   );
// }

// // ── Sensor Card ────────────────────────────────────────────────────────────────
// function SensorCard({ s }: { s: Sensor }) {
//   const range = s.normal_max - s.normal_min;
//   const pct = Math.min(100, Math.max(0, ((s.value - s.normal_min * 0.8) / (range * 1.5)) * 100));
//   const icon = SENSOR_ICONS[s.icon] ?? <Activity size={18} />;
//   const deltaUp = s.delta_pct > 1;
//   const deltaDown = s.delta_pct < -1;

//   return (
//     <div className={`sensor-card status-${s.status}`}>
//       <div className="sensor-top">
//         <div className={`sensor-icon-wrap ${s.status}`}>{icon}</div>
//         <span className={`sensor-status-pill ${s.status}`}>{s.status}</span>
//       </div>
//       <div className="sensor-name">{s.label}</div>
//       <div className="sensor-value">
//         {s.value.toLocaleString(undefined, { maximumFractionDigits: 1 })}
//         <span className="unit">{s.unit}</span>
//       </div>
//       <div className="sensor-bar-wrap">
//         <div className="sensor-bar-label">
//           <span>{s.normal_min} {s.unit}</span>
//           <span>Normal range</span>
//           <span>{s.normal_max} {s.unit}</span>
//         </div>
//         <div className="sensor-bar-track">
//           <div className={`sensor-bar-fill ${s.status}`} style={{ width: `${pct}%` }} />
//         </div>
//       </div>
//       <div className={`sensor-delta ${deltaUp ? "up" : deltaDown ? "down" : "flat"}`}>
//         {deltaUp ? "▲" : deltaDown ? "▼" : "—"}
//         {" "}{Math.abs(s.delta_pct).toFixed(1)}% from nominal
//       </div>
//     </div>
//   );
// }

// // ── Trend Chart ────────────────────────────────────────────────────────────────
// function TrendChart({ trend, keys, colors, title }: {
//   trend: Record<string, number | string>[];
//   keys: string[]; colors: string[]; title: string;
// }) {
//   return (
//     <div className="card" style={{ padding: "16px" }}>
//       <div className="card-header">
//         <div className="card-icon"><BarChart2 size={16} /></div>
//         <span className="card-title">{title}</span>
//       </div>
//       <ResponsiveContainer width="100%" height={200}>
//         <AreaChart data={trend} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
//           <defs>
//             {keys.map((k, i) => (
//               <linearGradient key={k} id={`grad-${k}`} x1="0" y1="0" x2="0" y2="1">
//                 <stop offset="5%" stopColor={colors[i]} stopOpacity={0.25} />
//                 <stop offset="95%" stopColor={colors[i]} stopOpacity={0} />
//               </linearGradient>
//             ))}
//           </defs>
//           <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
//           <XAxis dataKey="hour" tick={{ fill: "var(--text-muted)", fontSize: 10 }}
//             tickLine={false} interval={3} />
//           <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} tickLine={false} />
//           <Tooltip
//             contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
//             labelStyle={{ color: "var(--text-secondary)" }}
//             itemStyle={{ color: "var(--text-primary)" }}
//           />
//           <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
//           {keys.map((k, i) => (
//             <Area key={k} type="monotone" dataKey={k} stroke={colors[i]}
//               fill={`url(#grad-${k})`} strokeWidth={1.5} dot={false} name={k} />
//           ))}
//         </AreaChart>
//       </ResponsiveContainer>
//     </div>
//   );
// }

// // ── Failure Probability Bar ────────────────────────────────────────────────────
// function FailureBar({ label, value }: { label: string; value: number }) {
//   const pct = Math.min(100, value * 100);
//   const color = pct > 60 ? "var(--red)" : pct > 30 ? "var(--yellow)" : "var(--green)";
//   return (
//     <div style={{ marginBottom: 12 }}>
//       <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 5 }}>
//         <span style={{ color: "var(--text-secondary)" }}>{label}</span>
//         <span style={{ fontFamily: "'JetBrains Mono',monospace", fontWeight: 600, color }}>{pct.toFixed(1)}%</span>
//       </div>
//       <div style={{ height: 6, background: "var(--bg-input)", borderRadius: 3, overflow: "hidden" }}>
//         <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 3, transition: "width 0.8s ease" }} />
//       </div>
//     </div>
//   );
// }

// // ── Diagnosis Panel ────────────────────────────────────────────────────────────
// function DiagnosisPanel({ result, loading, machineId, onApprove }: { result: DiagnosisResult | null; loading: boolean; machineId?: number; onApprove?: (result: DiagnosisResult) => void }) {
//   const [checkedActions, setCheckedActions] = useState<Set<number>>(new Set());
//   const [approving, setApproving] = useState(false);

//   const toggleAction = (index: number) => {
//     const newChecked = new Set(checkedActions);
//     if (newChecked.has(index)) newChecked.delete(index);
//     else newChecked.add(index);
//     setCheckedActions(newChecked);
//   };

//   const handleApprove = async () => {
//     if (!machineId || !result) return;
//     setApproving(true);
//     try {
//       const response = await fetch(`${API}/maintenance/approve`, {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({
//           engine_id: machineId,
//           diagnosis: result.diagnosis,
//           recommended_actions: result.recommended_actions,
//           checked_actions: Array.from(checkedActions),
//           timestamp: new Date().toISOString(),
//         }),
//       });
//       if (response.ok) {
//         onApprove?.(result);
//         setCheckedActions(new Set());
//       }
//     } catch (e) {
//       console.error("Approval failed:", e);
//     } finally {
//       setApproving(false);
//     }
//   };

//   if (loading) {
//     return (
//       <div className="diagnosis-card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 40, gap: 14 }}>
//         <div className="spinner" style={{ width: 32, height: 32 }} />
//         <div style={{ color: "var(--text-muted)", fontSize: 13 }}>AI is analysing the machine…</div>
//         <div className="thinking-dots"><span>·</span><span>·</span><span>·</span></div>
//       </div>
//     );
//   }
//   if (!result) {
//     return (
//       <div className="diagnosis-card empty-state" style={{ minHeight: 200 }}>
//         <Bot size={40} style={{ opacity: 0.3 }} />
//         <span>Ask the AI Copilot a question to get started</span>
//       </div>
//     );
//   }

//   const sevColor = result.severity === "CRITICAL" ? "var(--red)"
//     : result.severity === "HIGH" ? "var(--yellow)"
//     : result.severity === "MEDIUM" ? "#f97316"
//     : "var(--green)";
//   const sevBg = result.severity === "CRITICAL" ? "rgba(239,68,68,0.15)"
//     : result.severity === "HIGH" ? "rgba(245,158,11,0.15)"
//     : result.severity === "MEDIUM" ? "rgba(249,115,22,0.15)"
//     : "rgba(34,197,94,0.1)";
//   const sevBorder = result.severity === "CRITICAL" ? "rgba(239,68,68,0.3)"
//     : result.severity === "HIGH" ? "rgba(245,158,11,0.3)"
//     : result.severity === "MEDIUM" ? "rgba(249,115,22,0.3)"
//     : "rgba(34,197,94,0.25)";

//   return (
//     <div className="diagnosis-card">
//       {/* ── Severity Banner ── */}
//       <div style={{
//         background: sevBg,
//         border: `1.5px solid ${sevBorder}`,
//         borderRadius: "var(--r-lg) var(--r-lg) 0 0",
//         padding: "16px 18px",
//         display: "flex",
//         alignItems: "center",
//         justifyContent: "space-between",
//       }}>
//         <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
//           <div style={{
//             width: 10, height: 10, borderRadius: "50%",
//             background: sevColor,
//             boxShadow: `0 0 8px ${sevColor}`,
//           }} />
//           <div>
//             <div style={{ fontWeight: 700, color: sevColor, fontSize: 13, textTransform: "uppercase", letterSpacing: "0.07em" }}>
//               {result.severity} SEVERITY
//             </div>
//             <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 2 }}>
//               {result.severity === "CRITICAL" ? "Immediate intervention required"
//                 : result.severity === "HIGH" ? "Schedule urgent maintenance"
//                 : result.severity === "MEDIUM" ? "Plan maintenance within 7 days"
//                 : "Continue monitoring"}
//             </div>
//           </div>
//         </div>
//         <Wrench size={20} color={sevColor} opacity={0.6} />
//       </div>

//       <div className="diagnosis-body">
//         {/* Diagnosis Summary */}
//         <div className="diagnosis-section">
//           <div className="diagnosis-section-title">Diagnosis Summary</div>
//           <div className="diagnosis-text" style={{ background: "var(--bg-surface)", padding: "12px 14px", borderRadius: "var(--r-sm)", borderLeft: `3px solid ${sevColor}` }}>
//             {result.diagnosis}
//           </div>
//         </div>

//         {/* Evidence Breakdown – Metric Badges */}
//         {result.evidence?.length > 0 && (
//           <div className="diagnosis-section">
//             <div className="diagnosis-section-title">Evidence Breakdown</div>
//             <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
//               {result.evidence.map((e, i) => (
//                 <div key={i} style={{
//                   background: "var(--bg-surface)",
//                   border: "1px solid var(--border-bright)",
//                   borderRadius: "var(--r-md)",
//                   padding: "8px 12px",
//                   fontSize: 12,
//                   color: "var(--text-secondary)",
//                   display: "flex",
//                   alignItems: "center",
//                   gap: 6,
//                 }}>
//                   <Activity size={13} color="var(--accent)" />
//                   {e}
//                 </div>
//               ))}
//             </div>
//           </div>
//         )}

//         {/* Recommended Actions – Interactive Checklist */}
//         {result.recommended_actions?.length > 0 && (
//           <div className="diagnosis-section">
//             <div className="diagnosis-section-title">Action Checklist</div>
//             <div style={{ background: "var(--bg-surface)", borderRadius: "var(--r-md)", border: "1px solid var(--border)", overflow: "hidden" }}>
//               {result.recommended_actions.map((a, i) => (
//                 <div key={i} style={{
//                   display: "flex",
//                   alignItems: "flex-start",
//                   gap: 10,
//                   padding: "12px 14px",
//                   borderBottom: i < result.recommended_actions.length - 1 ? "1px solid var(--border)" : "none",
//                   cursor: "pointer",
//                   transition: "background 0.2s",
//                   background: checkedActions.has(i) ? "rgba(34,197,94,0.08)" : "transparent",
//                 }}
//                 onClick={() => toggleAction(i)}>
//                   <input
//                     type="checkbox"
//                     checked={checkedActions.has(i)}
//                     onChange={() => toggleAction(i)}
//                     style={{
//                       width: 18, height: 18, marginTop: 2,
//                       cursor: "pointer", accentColor: "var(--green)",
//                     }}
//                   />
//                   <span style={{
//                     flex: 1, fontSize: 13, color: "var(--text-primary)",
//                     textDecoration: checkedActions.has(i) ? "line-through" : "none",
//                     opacity: checkedActions.has(i) ? 0.6 : 1,
//                   }}>
//                     {a}
//                   </span>
//                 </div>
//               ))}
//             </div>
//           </div>
//         )}

//         {/* Knowledge Base Evidence */}
//         {result.knowledge_evidence?.length > 0 && (
//           <div className="diagnosis-section">
//             <div className="diagnosis-section-title">Related SOP Knowledge</div>
//             {result.knowledge_evidence.slice(0, 2).map((k, i) => (
//               <div key={i} className="rag-chunk">
//                 <div className="rag-score">
//                   <Search size={10} /> Relevance: {(k.score * 100).toFixed(0)}%
//                 </div>
//                 <div>{k.text.substring(0, 200)}{k.text.length > 200 ? "…" : ""}</div>
//               </div>
//             ))}
//           </div>
//         )}

//         {/* Approval Workflow Button */}
//         <div style={{ marginTop: 16, paddingTop: 14, borderTop: "1px solid var(--border)" }}>
//           <button
//             onClick={handleApprove}
//             disabled={approving || result.recommended_actions.length === 0}
//             style={{
//               width: "100%",
//               padding: "12px 16px",
//               borderRadius: "var(--r-md)",
//               border: "none",
//               background: sevColor,
//               color: "#fff",
//               fontWeight: 600,
//               fontSize: 13,
//               cursor: approving ? "not-allowed" : "pointer",
//               display: "flex",
//               alignItems: "center",
//               justifyContent: "center",
//               gap: 8,
//               opacity: approving ? 0.6 : 1,
//               transition: "all 0.2s",
//             }}
//           >
//             {approving ? (
//               <>
//                 <div className="spinner" style={{ width: 14, height: 14 }} />
//                 Approving…
//               </>
//             ) : (
//               <>
//                 <CheckCircle size={16} />
//                 Approve & Log Maintenance
//               </>
//             )}
//           </button>
//           <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 6, textAlign: "center" }}>
//             Review actions above, then approve to log in the maintenance system
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// }

// // ── History Tab ────────────────────────────────────────────────────────────────
// function HistoryTab({ hist }: { hist: HistoryData | null }) {
//   if (!hist) return <div className="empty-state"><History size={40} /><span>No history loaded</span></div>;

//   const errorByCode = hist.error_log.reduce<Record<string, number>>((acc, e) => {
//     acc[e.code] = (acc[e.code] || 0) + e.count; return acc;
//   }, {});
//   const errorChartData = Object.entries(errorByCode).map(([code, count]) => ({ code, count }));

//   return (
//     <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
//       {/* Stats row */}
//       <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
//         {[
//           { label: "Machine Age", val: `${hist.machine_age_years} yrs`, icon: <Cpu size={18} /> },
//           { label: "Last Service", val: `${hist.last_service_days_ago}d ago`, icon: <Wrench size={18} /> },
//           { label: "Maintenance Events", val: hist.total_maintenance_events, icon: <CheckCircle size={18} /> },
//           { label: "Total Errors", val: hist.total_errors, icon: <AlertTriangle size={18} /> },
//         ].map(({ label, val, icon }) => (
//           <div key={label} className="card" style={{ padding: "14px 18px", display: "flex", gap: 12, alignItems: "center" }}>
//             <div className="card-icon">{icon}</div>
//             <div>
//               <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.07em" }}>{label}</div>
//               <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 18, fontWeight: 700 }}>{val}</div>
//             </div>
//           </div>
//         ))}
//       </div>

//       <div className="history-grid">
//         {/* Error log */}
//         <div className="card">
//           <div className="card-header">
//             <div className="card-icon" style={{ background: "rgba(239,68,68,0.1)", color: "var(--red)" }}>
//               <AlertTriangle size={16} />
//             </div>
//             <span className="card-title">Error Log</span>
//           </div>
//           {hist.error_log.length === 0
//             ? <div style={{ color: "var(--text-muted)", fontSize: 13 }}>No errors recorded</div>
//             : hist.error_log.map((e, i) => (
//               <div key={i} className="timeline-item">
//                 <div className="timeline-dot error" />
//                 <div>
//                   <div className="timeline-text">
//                     <span style={{ fontFamily: "'JetBrains Mono',monospace", color: "var(--red)", fontWeight: 600 }}>{e.code}</span>
//                     {" — "}{e.description}
//                   </div>
//                   <div className="timeline-meta">{e.days_ago}d ago · occurred {e.count}×</div>
//                 </div>
//               </div>
//             ))}
//         </div>

//         {/* Maintenance log */}
//         <div className="card">
//           <div className="card-header">
//             <div className="card-icon"><Wrench size={16} /></div>
//             <span className="card-title">Maintenance Log</span>
//           </div>
//           {hist.maintenance_log.length === 0
//             ? <div style={{ color: "var(--text-muted)", fontSize: 13 }}>No maintenance recorded</div>
//             : hist.maintenance_log.map((m, i) => (
//               <div key={i} className="timeline-item">
//                 <div className="timeline-dot" />
//                 <div>
//                   <div className="timeline-text">{m.action}</div>
//                   <div className="timeline-meta">{m.days_ago}d ago · {m.technician}</div>
//                 </div>
//               </div>
//             ))}
//         </div>
//       </div>

//       {/* Error frequency bar chart */}
//       {errorChartData.length > 0 && (
//         <div className="card" style={{ padding: "16px" }}>
//           <div className="card-header">
//             <div className="card-icon" style={{ background: "rgba(239,68,68,0.1)", color: "var(--red)" }}>
//               <BarChart2 size={16} />
//             </div>
//             <span className="card-title">Error Frequency</span>
//           </div>
//           <ResponsiveContainer width="100%" height={160}>
//             <BarChart data={errorChartData} margin={{ top: 4, right: 8, bottom: 4, left: -16 }}>
//               <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
//               <XAxis dataKey="code" tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} />
//               <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} />
//               <Tooltip
//                 contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
//               />
//               <Bar dataKey="count" fill="var(--red)" radius={[4, 4, 0, 0]} />
//             </BarChart>
//           </ResponsiveContainer>
//         </div>
//       )}
//     </div>
//   );
// }

// // ── Fleet Summary Bar ──────────────────────────────────────────────────────────
// function FleetSummaryPanel({ fleet }: { fleet: FleetSummary | null }) {
//   if (!fleet) return null;
//   return (
//     <div className="fleet-summary">
//       <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.07em" }}>
//         Fleet Overview
//       </div>
//       {[
//         { label: "Total", val: fleet.total_machines, color: "var(--text-primary)" },
//         { label: "Critical", val: fleet.critical, color: "var(--red)" },
//         { label: "High Risk", val: fleet.high, color: "var(--yellow)" },
//         { label: "Attention", val: fleet.attention, color: "#f97316" },
//         { label: "Normal", val: fleet.normal, color: "var(--green)" },
//       ].map(({ label, val, color }) => (
//         <div key={label} className="fleet-row">
//           <span className="fleet-label">{label}</span>
//           <span className="fleet-val" style={{ color }}>{val}</span>
//         </div>
//       ))}
//       <div style={{ borderTop: "1px solid var(--border)", marginTop: 8, paddingTop: 8 }}>
//         <div className="fleet-row">
//           <span className="fleet-label">Avg Health</span>
//           <span className="fleet-val" style={{ color: fleet.avg_health_score >= 60 ? "var(--green)" : "var(--yellow)" }}>
//             {fleet.avg_health_score}/100
//           </span>
//         </div>
//         <div className="fleet-row">
//           <span className="fleet-label">Avg RUL</span>
//           <span className="fleet-val">{fleet.avg_rul} cyc</span>
//         </div>
//       </div>
//     </div>
//   );
// }

// // ── Main App ───────────────────────────────────────────────────────────────────
// const QUICK_QUESTIONS = [
//   // ── Shutdown & Safety Decisions (Tier 1: Critical)
//   "Should this turbofan engine be taken offline for immediate inspection?",
//   "What are the LOTO (lockout-tagout) and safety protocols before maintenance?",
  
//   // ── Sensor Attribution & Diagnostics (Tier 2: Technical Analysis)
//   "Which sensor channels (HPC temp, pressure, fan speed) show strongest degradation?",
//   "Is there high-pressure compressor (HPC) thermal runaway or exhaust gas overtemp?",
//   "Is there core rotational instability or bearing assembly degradation?",
//   "Is HPC outlet pressure dropping relative to fuel flow baseline?",
  
//   // ── RUL & Failure Risk (Tier 3: Prognostics)
//   "What is the remaining useful life (RUL) and is degradation accelerating?",
//   "What is the calculated failure risk for the next 24 and 48 operational cycles?",
  
//   // ── Trend Analysis & Anomalies (Tier 4: Recent Behavior)
//   "What anomalous drift occurred over the last 24 engine cycles?",
//   "How do current sensor readings compare to the baseline at cycle 1?",
  
//   // ── Inspection & Maintenance Planning (Tier 5: Action Items)
//   "What bore-scope and non-destructive inspection scope is required per SOP §4?",
//   "What maintenance actions are required and what is the urgency?",
  
//   // ── Context & Status (Tier 6: Fleet Awareness)
//   "Why is this turbofan engine at its current risk level?",
//   "How does this engine compare to similar engines in the fleet by health and RUL?",
//   "What is the overall fleet risk status and which engines need priority?",
  
//   // ── Deep Diagnostics (Tier 7: Root Cause)
//   "Is this a one-time anomaly or progressive degradation requiring intervention?",
//   "What is the full diagnostic verdict and recommended next actions?",
// ];

// const AUTO_REFRESH_MS = 30_000; // 30 seconds

// // Health check on startup
// async function checkApiConnection() {
//   try {
//     const response = await fetch(`${API}/health`);
//     if (response.ok) {
//       console.log('✓ Backend API is running at', API);
//       return true;
//     }
//   } catch (error) {
//     console.error('✗ Cannot reach backend API at', API);
//     console.error('Error:', error);
//   }
//   return false;
// }

// export default function App() {
//   const [machineIds, setMachineIds] = useState<number[]>([]);
//   const [selectedId, setSelectedId] = useState<number | null>(null);
//   const [ctx, setCtx] = useState<MachineCtx | null>(null);
//   const [sensorData, setSensorData] = useState<SensorData | null>(null);
//   const [histData, setHistData] = useState<HistoryData | null>(null);
//   const [fleet, setFleet] = useState<FleetSummary | null>(null);
//   // riskMap: engine_id → risk_level, loaded lazily for all machines
//   const [tab, setTab] = useState<"sensors" | "trends" | "history" | "copilot">("sensors");
//   const [question, setQuestion] = useState(QUICK_QUESTIONS[0]);
//   const [diagnosis, setDiagnosis] = useState<DiagnosisResult | null>(null);
//   const [loading, setLoading] = useState(false);
//   const [riskMap, setRiskMap] = useState<Record<number, string>>({});
  
//   // Fleet filtering and sorting
//   const [riskFilter, setRiskFilter] = useState<"All" | "Critical" | "High" | "Attention" | "Normal">("All");
//   const [sortBy, setSortBy] = useState<"engine_id" | "lowest_rul" | "lowest_health">("engine_id");
//   const [search, setSearch] = useState("");
//   const [clock, setClock] = useState(timeStr());
//   const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
//   const [refreshing, setRefreshing] = useState(false);
//   const [apiConnected, setApiConnected] = useState(true);
//   const [toasts, setToasts] = useState<Toast[]>([]);
//   const selectedIdRef = useRef<number | null>(null);

//   // Toast helper
//   const showToast = (message: string, type: "success" | "error" | "info" = "info", duration: number = 3000) => {
//     const id = Math.random().toString(36);
//     const newToast: Toast = { id, message, type, duration };
//     setToasts(prev => [...prev, newToast]);
//     setTimeout(() => {
//       setToasts(prev => prev.filter(t => t.id !== id));
//     }, duration);
//   };

//   // Check API connection on mount
//   useEffect(() => {
//     checkApiConnection().then(setApiConnected);
//   }, []);

//   // Load fleet + machine list once on mount
//   useEffect(() => {
//     fetch(`${API}/fleet/summary`).then(r => r.json()).then(setFleet).catch(() => {});
//     fetch(`${API}/machines`).then(r => r.json()).then(d => {
//       const ids: number[] = d.machines;
//       setMachineIds(ids);
//       if (ids.length > 0) selectMachine(ids[0]);
//       // Load risk level for first 50 machines in background for sidebar dots
//       ids.slice(0, 50).forEach(id => {
//         fetch(`${API}/machines/${id}`)
//           .then(r => r.ok ? r.json() : null)
//           .then(data => {
//             if (data) setRiskMap(prev => ({ ...prev, [id]: data.risk_level }));
//           })
//           .catch(() => {});
//       });
//     });
//   }, []);

//   const selectMachine = useCallback(async (id: number) => {
//     selectedIdRef.current = id;
//     setSelectedId(id);
//     setCtx(null); setSensorData(null); setHistData(null); setDiagnosis(null);
//     const [ctxR, sensorR, histR] = await Promise.all([
//       fetch(`${API}/machines/${id}`).then(r => r.ok ? r.json() : null),
//       fetch(`${API}/machines/${id}/sensors`).then(r => r.ok ? r.json() : null),
//       fetch(`${API}/machines/${id}/history`).then(r => r.ok ? r.json() : null),
//     ]);
//     if (selectedIdRef.current !== id) return; // stale response guard
//     setCtx(ctxR); setSensorData(sensorR); setHistData(histR);
//     if (ctxR) setRiskMap(prev => ({ ...prev, [id]: ctxR.risk_level }));
//   }, []);

//   // Auto-refresh current machine data every 30s
//   useEffect(() => {
//     const t = setInterval(async () => {
//       const id = selectedIdRef.current;
//       if (!id) return;
//       setRefreshing(true);
//       try {
//         const [ctxR, sensorR] = await Promise.all([
//           fetch(`${API}/machines/${id}`).then(r => r.ok ? r.json() : null),
//           fetch(`${API}/machines/${id}/sensors`).then(r => r.ok ? r.json() : null),
//         ]);
//         if (selectedIdRef.current !== id) return;
//         if (ctxR) { setCtx(ctxR); setRiskMap(prev => ({ ...prev, [id]: ctxR.risk_level })); }
//         if (sensorR) setSensorData(sensorR);
//         // Refresh fleet summary too
//         fetch(`${API}/fleet/summary`).then(r => r.json()).then(setFleet).catch(() => {});
//         setLastRefresh(new Date());
//       } finally {
//         setRefreshing(false);
//       }
//     }, AUTO_REFRESH_MS);
//     return () => clearInterval(t);
//   }, []);

//   const investigate = async () => {
//     if (!selectedId || !question.trim()) return;
//     setLoading(true); setDiagnosis(null);
//     try {
//       const r = await fetch(`${API}/machines/${selectedId}/investigate`, {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ question }),
//       });
//       setDiagnosis(await r.json());
//       setTab("copilot");
//     } finally { setLoading(false); }
//   };

//   const handleManualRefresh = () => {
//     if (selectedId) selectMachine(selectedId);
//     setLastRefresh(new Date());
//   };

//   let filteredIds = machineIds.filter(id => {
//     // Text search
//     if (search !== "" && !String(id).includes(search)) return false;
    
//     // Risk filter
//     const risk = riskMap[id] ?? "NORMAL";
//     if (riskFilter === "All") return true;
//     if (riskFilter === "Critical" && risk !== "CRITICAL") return false;
//     if (riskFilter === "High" && risk !== "HIGH") return false;
//     if (riskFilter === "Attention" && risk !== "ATTENTION") return false;
//     if (riskFilter === "Normal" && risk !== "NORMAL") return false;
    
//     return true;
//   });

//   // Apply sorting
//   if (sortBy === "lowest_rul") {
//     filteredIds.sort((a, b) => {
//       const rulA = machineCtx[a]?.predicted_rul ?? 999;
//       const rulB = machineCtx[b]?.predicted_rul ?? 999;
//       return rulA - rulB;
//     });
//   } else if (sortBy === "lowest_health") {
//     filteredIds.sort((a, b) => {
//       const healthA = machineCtx[a]?.health_score ?? 100;
//       const healthB = machineCtx[b]?.health_score ?? 100;
//       return healthA - healthB;
//     });
//   } else {
//     filteredIds.sort((a, b) => a - b);
//   }

//   // Sensor summary for top bar
//   const criticalSensors = sensorData?.sensors.filter(s => s.status === "critical") ?? [];
//   const warningSensors = sensorData?.sensors.filter(s => s.status === "warning") ?? [];

//   return (
//     <div className="app-layout">
//       {/* ── Toast System ── */}
//       <ToastContainer toasts={toasts} onRemove={(id) => setToasts(prev => prev.filter(t => t.id !== id))} />
//       {/* ── Top Bar ── */}
//       <header className="topbar">
//         <div className="topbar-logo">
//           <Cpu size={20} />
//           Industrial Maintenance Copilot
//           <span className="topbar-badge">SLM v2</span>
//         </div>
//         <div className="topbar-spacer" />
//         {refreshing && (
//           <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--accent)" }}>
//             <div className="spinner" style={{ width: 12, height: 12 }} /> Auto-refreshing…
//           </div>
//         )}
//         {!refreshing && (
//           <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
//             Last updated: {lastRefresh.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
//           </div>
//         )}
//         {criticalSensors.length > 0 && (
//           <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--red)", background: "var(--red-bg)", padding: "4px 12px", borderRadius: 20, border: "1px solid rgba(239,68,68,0.3)" }}>
//             <AlertTriangle size={13} /> {criticalSensors.length} critical sensor{criticalSensors.length > 1 ? "s" : ""}
//           </div>
//         )}
//         <div className="status-dot" />
//         <span className="topbar-time">{clock}</span>
//       </header>

//       {/* ── Sidebar ── */}
//       <aside className="sidebar">
//         <FleetSummaryPanel fleet={fleet} />
        
//         <div className="sidebar-section-label" style={{ marginTop: 16 }}>Risk Filter</div>
//         <div style={{ display: "flex", gap: 6, padding: "0 8px", flexWrap: "wrap" }}>
//           {(["All", "Critical", "High", "Attention", "Normal"] as const).map(level => (
//             <button
//               key={level}
//               onClick={() => setRiskFilter(level)}
//               style={{
//                 padding: "6px 12px",
//                 fontSize: 11,
//                 fontWeight: 600,
//                 borderRadius: 16,
//                 border: `1.5px solid ${riskFilter === level ? riskColor(level === "All" ? "NORMAL" : level.toUpperCase()) : "var(--border)"}`,
//                 background: riskFilter === level ? `${riskColor(level === "All" ? "NORMAL" : level.toUpperCase())}12` : "var(--bg-input)",
//                 color: riskFilter === level ? riskColor(level === "All" ? "NORMAL" : level.toUpperCase()) : "var(--text-secondary)",
//                 cursor: "pointer",
//                 transition: "all 0.2s ease",
//               }}
//             >
//               {level}
//             </button>
//           ))}
//         </div>
        
//         <div className="sidebar-section-label" style={{ marginTop: 12 }}>Sort By</div>
//         <select
//           value={sortBy}
//           onChange={e => setSortBy(e.target.value as any)}
//           style={{
//             margin: "0 8px 12px",
//             padding: "8px 12px",
//             borderRadius: 8,
//             border: "1px solid var(--border)",
//             background: "var(--bg-input)",
//             color: "var(--text-primary)",
//             fontSize: 12,
//             cursor: "pointer",
//             width: "calc(100% - 16px)",
//           }}
//         >
//           <option value="engine_id">Engine ID</option>
//           <option value="lowest_rul">Lowest RUL</option>
//           <option value="lowest_health">Lowest Health</option>
//         </select>
        
//         <div className="sidebar-section-label">Machines ({filteredIds.length})</div>
//         <input className="machine-search" placeholder="Search engine ID…"
//           value={search} onChange={e => setSearch(e.target.value)} />
//         <div style={{ overflowY: "auto", flex: 1 }}>
//           {filteredIds.map(id => {
//             const isSelected = id === selectedId;
//             const risk = riskMap[id] ?? "NORMAL";
//             const riskDot = riskDotClass(risk);
//             const healthScore = machineCtx[id]?.health_score ?? 50;
//             const rul = machineCtx[id]?.predicted_rul ?? 0;
//             return (
//               <div key={id} className={`machine-item ${isSelected ? "active" : ""}`}
//                 onClick={() => selectMachine(id)}
//                 style={{
//                   display: "flex",
//                   alignItems: "center",
//                   gap: 8,
//                   padding: "8px 12px",
//                   cursor: "pointer",
//                   borderRadius: 6,
//                   marginBottom: 4,
//                   marginHorizontal: 8,
//                   background: isSelected ? "rgba(14,165,233,0.1)" : "transparent",
//                   border: isSelected ? "1px solid var(--accent)" : "transparent",
//                   transition: "all 0.15s ease",
//                 }}>
//                 <Cpu size={13} color={isSelected ? "var(--accent)" : "var(--text-muted)"} />
//                 <div style={{ flex: 1 }}>
//                   <div style={{ fontSize: 12, fontFamily: "'JetBrains Mono',monospace", fontWeight: 600, color: "var(--text-primary)" }}>
//                     ENG-{String(id).padStart(3, "0")}
//                   </div>
//                   <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
//                     RUL: {rul.toFixed(0)} • H: {healthScore.toFixed(0)}
//                   </div>
//                 </div>
//                 <span className={`machine-risk-dot ${riskDot}`} />
//                 {isSelected && <ChevronRight size={12} color="var(--accent)" />}
//               </div>
//             );
//           })}
//         </div>
//       </aside>

//       {/* ── Main ── */}
//       <main className="main-content">
//         {!ctx ? (
//           <div className="empty-state" style={{ minHeight: "60vh" }}>
//             <Cpu size={48} />
//             <span>Select a machine to begin</span>
//           </div>
//         ) : (
//           <>
//             {/* Machine Header */}
//             <div className="machine-header">
//               <div>
//                 <div className="machine-title">ENG-{String(ctx.engine_id).padStart(3, "0")}</div>
//                 <div className="machine-subtitle">Operational Cycle: {ctx.cycle}</div>
//               </div>
//               <div style={{ flex: 1 }} />
//               <span className={`risk-badge risk-${ctx.risk_level}`}>{ctx.risk_level}</span>
              
//               {/* RUL Urgency Progress Gauge */}
//               <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 180 }}>
//                 <div className="kpi-label" style={{ marginBottom: 2 }}>RUL Status</div>
//                 <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
//                   <div style={{ flex: 1 }}>
//                     <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4, color: "var(--text-muted)" }}>
//                       <span>{ctx.predicted_rul.toFixed(0)} cycles left</span>
//                       <span style={{ fontFamily: "'JetBrains Mono',monospace", fontWeight: 600 }}>
//                         {ctx.predicted_rul <= 25 ? "🔴 CRITICAL" : ctx.predicted_rul <= 50 ? "🟠 HIGH" : ctx.predicted_rul <= 75 ? "🟡 ATTENTION" : "🟢 NORMAL"}
//                       </span>
//                     </div>
//                     <div style={{ height: 8, background: "var(--bg-input)", borderRadius: 4, overflow: "hidden", border: "1px solid var(--border)" }}>
//                       <div style={{
//                         height: "100%",
//                         width: `${Math.max(5, Math.min(100, (ctx.predicted_rul / 125) * 100))}%`,
//                         background: ctx.predicted_rul <= 25 ? "var(--red)" : ctx.predicted_rul <= 50 ? "var(--yellow)" : ctx.predicted_rul <= 75 ? "#f97316" : "var(--green)",
//                         borderRadius: 4,
//                         transition: "width 0.6s ease, background 0.6s ease"
//                       }} />
//                     </div>
//                   </div>
//                 </div>
//               </div>
              
//               <div className="kpi">
//                 <div className="kpi-label">Health Score</div>
//                 <div className="kpi-value" style={{ color: ctx.health_score >= 70 ? "var(--green)" : ctx.health_score >= 40 ? "var(--yellow)" : "var(--red)" }}>
//                   {ctx.health_score.toFixed(1)}
//                 </div>
//                 <div className="kpi-sub">/ 100</div>
//               </div>
              
//               <div className="kpi">
//                 <div className="kpi-label">Anomaly Score</div>
//                 <div className="kpi-value" style={{ color: ctx.anomaly_score > 0.5 ? "var(--red)" : ctx.anomaly_score > 0.25 ? "var(--yellow)" : "var(--green)" }}>
//                   {ctx.anomaly_score.toFixed(3)}
//                 </div>
//                 <div className="kpi-sub">{ctx.anomaly_status}</div>
//               </div>
              
//               {/* 24h & 48h Failure Risk Indicators */}
//               <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 160 }}>
//                 <FailureBar label="Failure P(24h)" value={ctx.failure_probability_24h} />
//                 <FailureBar label="Failure P(48h)" value={ctx.failure_probability_48h} />
//               </div>
//             </div>

//             {/* Tabs */}
//             <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
//               <div className="tabs">
//                 {[
//                   { id: "sensors", label: "Sensors", icon: <Activity size={14} /> },
//                   { id: "trends",  label: "24h Trends", icon: <BarChart2 size={14} /> },
//                   { id: "history", label: "History", icon: <History size={14} /> },
//                   { id: "copilot", label: "AI Copilot", icon: <Bot size={14} /> },
//                 ].map(t => (
//                   <button key={t.id} className={`tab ${tab === t.id ? "active" : ""}`}
//                     onClick={() => setTab(t.id as any)}>
//                     {t.icon}{t.label}
//                   </button>
//                 ))}
//               </div>
//               <button className="btn btn-primary" style={{ marginLeft: "auto" }}
//                 onClick={handleManualRefresh} disabled={refreshing}>
//                 <RefreshCw size={14} style={{ animation: refreshing ? "spin 0.7s linear infinite" : "none" }} /> Refresh
//               </button>
//             </div>

//             {/* ── Sensors Tab ── */}
//             {tab === "sensors" && sensorData && (
//               <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
//                 <div className="sensor-grid">
//                   {sensorData.sensors.map(s => <SensorCard key={s.key} s={s} />)}
//                 </div>
//                 <div style={{ display: "grid", gridTemplateColumns: "140px 1fr", gap: 16 }}>
//                   <div className="card" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
//                     <HealthRing score={ctx.health_score} />
//                   </div>
//                   <div className="card" style={{ padding: "16px" }}>
//                     <div className="card-header">
//                       <div className="card-icon"><Activity size={16} /></div>
//                       <span className="card-title">Sensor Status Summary</span>
//                     </div>
//                     <ResponsiveContainer width="100%" height={120}>
//                       <BarChart
//                         data={sensorData.sensors.map(s => ({
//                           name: s.label.slice(0, 5),
//                           value: s.value,
//                           normal_min: s.normal_min,
//                           normal_max: s.normal_max,
//                         }))}
//                         margin={{ top: 4, right: 8, bottom: 0, left: -16 }}
//                       >
//                         <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
//                         <XAxis dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 10 }} tickLine={false} />
//                         <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} tickLine={false} />
//                         <Tooltip
//                           contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
//                         />
//                         <Bar dataKey="value" radius={[4, 4, 0, 0]}
//                           fill="var(--accent)"
//                           label={false}
//                         />
//                       </BarChart>
//                     </ResponsiveContainer>
//                     <div style={{ marginTop: 12, display: "flex", gap: 20, flexWrap: "wrap" }}>
//                       {[
//                         { label: "Normal", count: sensorData.sensors.filter(s => s.status === "normal").length, color: "var(--green)" },
//                         { label: "Warning", count: warningSensors.length, color: "var(--yellow)" },
//                         { label: "Critical", count: criticalSensors.length, color: "var(--red)" },
//                       ].map(({ label, count, color }) => (
//                         <div key={label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
//                           <div style={{ width: 10, height: 10, borderRadius: 2, background: color }} />
//                           <span style={{ color: "var(--text-secondary)" }}>{label}:</span>
//                           <span style={{ fontWeight: 700, color }}>{count}</span>
//                         </div>
//                       ))}
//                     </div>
//                   </div>
//                 </div>
//               </div>
//             )}

//             {/* ── Trends Tab ── */}
//             {tab === "trends" && sensorData && (
//               <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
//                 <div className="chart-row">
//                   <TrendChart trend={sensorData.trend}
//                     keys={["temperature", "pressure"]}
//                     colors={["#ef4444", "#3b82f6"]}
//                     title="Temperature & Pressure (24h)" />
//                   <TrendChart trend={sensorData.trend}
//                     keys={["vibration", "current"]}
//                     colors={["#f59e0b", "#8b5cf6"]}
//                     title="Vibration & Current (24h)" />
//                 </div>
//                 <div className="chart-row">
//                   <TrendChart trend={sensorData.trend}
//                     keys={["voltage"]}
//                     colors={["#22c55e"]}
//                     title="Voltage (24h)" />
//                   <TrendChart trend={sensorData.trend}
//                     keys={["rotation"]}
//                     colors={["#06b6d4"]}
//                     title="Rotation Speed (24h)" />
//                 </div>
//               </div>
//             )}

//             {/* ── History Tab ── */}
//             {tab === "history" && <HistoryTab hist={histData} />}

//             {/* ── Copilot Tab ── */}
//             {tab === "copilot" && (
//               <div className="copilot-wrap">
//                 <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
//                   <div className="card">
//                     <div className="card-header">
//                       <div className="card-icon"><Bot size={16} /></div>
//                       <span className="card-title">Ask the AI Copilot</span>
//                     </div>
//                     <div className="question-chips">
//                       {QUICK_QUESTIONS.map(q => (
//                         <button key={q} className="chip" onClick={() => setQuestion(q)}>{q}</button>
//                       ))}
//                     </div>
//                     <div className="copilot-input-row">
//                       <textarea className="copilot-textarea"
//                         value={question}
//                         onChange={e => setQuestion(e.target.value)}
//                         placeholder="Ask about this machine…"
//                         onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); investigate(); } }}
//                       />
//                     </div>
//                     <div style={{ marginTop: 8 }}>
//                       <button className="btn btn-primary" onClick={investigate} disabled={loading}>
//                         {loading ? <><div className="spinner" />Analysing…</> : <><Send size={14} />Investigate</>}
//                       </button>
//                     </div>
//                   </div>

//                   {/* Context summary card */}
//                   <div className="card">
//                     <div className="card-header">
//                       <div className="card-icon"><Cpu size={16} /></div>
//                       <span className="card-title">Machine Context</span>
//                     </div>
//                     {[
//                       { k: "Engine ID", v: `ENG-${String(ctx.engine_id).padStart(3,"0")}` },
//                       { k: "Risk Level", v: ctx.risk_level, color: riskColor(ctx.risk_level) },
//                       { k: "Health Score", v: `${ctx.health_score.toFixed(1)} / 100` },
//                       { k: "Predicted RUL", v: `${ctx.predicted_rul.toFixed(1)} cycles` },
//                       { k: "Anomaly Status", v: ctx.anomaly_status },
//                       { k: "Failure P(24h)", v: `${(ctx.failure_probability_24h * 100).toFixed(1)}%` },
//                       ...(histData ? [
//                         { k: "Machine Age", v: `${histData.machine_age_years} years` },
//                         { k: "Last Service", v: `${histData.last_service_days_ago} days ago` },
//                         { k: "Total Errors", v: String(histData.total_errors) },
//                       ] : []),
//                     ].map(({ k, v, color }) => (
//                       <div key={k} className="stat-row">
//                         <span className="stat-key">{k}</span>
//                         <span className="stat-val" style={color ? { color } : {}}>{v}</span>
//                       </div>
//                     ))}
//                   </div>
//                 </div>

//                 <DiagnosisPanel 
//                   result={diagnosis} 
//                   loading={loading} 
//                   machineId={selectedId ?? undefined}
//                   onApprove={(result) => showToast(`✓ Maintenance plan approved for Engine ${result.engine_id}`, "success")}
//                 />
//               </div>
//             )}
//           </>
//         )}
//       </main>
//     </div>
//   );
// }

// createRoot(document.getElementById("root")!).render(<App />);
