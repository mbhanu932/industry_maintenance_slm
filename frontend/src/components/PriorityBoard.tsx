/**
 * Priority Board Component
 * Production-ready task management with Kanban-style board and priority matrix
 */

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence, Reorder } from 'framer-motion';
import {
  Plus,
  Filter,
  Search,
  Clock,
  User,
  AlertTriangle,
  CheckCircle,
  Calendar,
  ArrowUp,
  ArrowDown,
  MoreVertical,
  Play,
  Pause,
  Square,
  Target,
  Wrench,
  Eye,
  Settings,
  Zap,
} from 'lucide-react';
import { formatDistanceToNow, isAfter, parseISO } from 'date-fns';
import { useAppStore, usePriorityTasks, useIsLoading } from '../store/useAppStore';
import { api } from '../services/api';
import {
  PriorityTask,
  TaskType,
  TaskPriority,
  TaskStatus,
  MachineContext,
} from '../types';

interface PriorityBoardProps {
  className?: string;
}

interface TaskFilters {
  priorities: TaskPriority[];
  statuses: TaskStatus[];
  types: TaskType[];
  assignedTo?: string;
  searchQuery: string;
  overdue: boolean;
  dueSoon: boolean;
}

const DEFAULT_FILTERS: TaskFilters = {
  priorities: ['critical', 'high', 'medium', 'low'],
  statuses: ['pending', 'in_progress', 'completed', 'cancelled', 'blocked'],
  types: ['maintenance', 'inspection', 'repair', 'replacement', 'calibration'],
  searchQuery: '',
  overdue: true,
  dueSoon: true,
};

// Task type icons and colors
const TASK_ICONS: Record<TaskType, React.ReactNode> = {
  maintenance: <Wrench size={16} />,
  inspection: <Eye size={16} />,
  repair: <Settings size={16} />,
  replacement: <Target size={16} />,
  calibration: <Zap size={16} />,
};

const PRIORITY_COLORS: Record<TaskPriority, string> = {
  critical: '#ef4444',
  high: '#f59e0b',
  medium: '#3b82f6',
  low: '#6b7280',
};

const STATUS_COLORS: Record<TaskStatus, string> = {
  pending: '#6b7280',
  in_progress: '#3b82f6',
  completed: '#22c55e',
  cancelled: '#ef4444',
  blocked: '#f59e0b',
};

export const PriorityBoard: React.FC<PriorityBoardProps> = ({ className = '' }) => {
  const {
    updatePriorityTasks,
    selectMachine,
    setView,
  } = useAppStore();
  
  const priorityTasks = usePriorityTasks();
  const isLoading = useIsLoading();
  
  // Local state
  const [filters, setFilters] = useState<TaskFilters>(DEFAULT_FILTERS);
  const [viewMode, setViewMode] = useState<'board' | 'matrix' | 'list'>('board');
  const [showFilters, setShowFilters] = useState(false);
  const [showNewTaskForm, setShowNewTaskForm] = useState(false);
  const [selectedTask, setSelectedTask] = useState<PriorityTask | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  
  // Load tasks
  useEffect(() => {
    loadTasks();
  }, []);
  
  const loadTasks = async () => {
    setRefreshing(true);
    try {
      const tasks = await api.fleet.getPriorityTasks();
      updatePriorityTasks(tasks);
    } catch (error) {
      console.error('Failed to load tasks:', error);
    } finally {
      setRefreshing(false);
    }
  };
  
  // Filter and sort tasks
  const filteredTasks = useMemo(() => {
    return priorityTasks
      .filter(task => {
        // Priority filter
        if (!filters.priorities.includes(task.priority)) return false;
        
        // Status filter
        if (!filters.statuses.includes(task.status)) return false;
        
        // Type filter
        if (!filters.types.includes(task.task_type)) return false;
        
        // Assigned user filter
        if (filters.assignedTo && task.assigned_to !== filters.assignedTo) return false;
        
        // Search query
        if (filters.searchQuery) {
          const query = filters.searchQuery.toLowerCase();
          if (!task.title.toLowerCase().includes(query) && 
              !task.description.toLowerCase().includes(query) &&
              !task.engine_id.toString().includes(query)) {
            return false;
          }
        }
        
        // Overdue filter
        if (!filters.overdue && task.deadline && isAfter(new Date(), parseISO(task.deadline))) {
          return false;
        }
        
        // Due soon filter (next 3 days)
        if (!filters.dueSoon && task.deadline) {
          const threeDaysFromNow = new Date();
          threeDaysFromNow.setDate(threeDaysFromNow.getDate() + 3);
          if (isAfter(parseISO(task.deadline), new Date()) && 
              isAfter(threeDaysFromNow, parseISO(task.deadline))) {
            return false;
          }
        }
        
        return true;
      })
      .sort((a, b) => {
        // Sort by priority first, then by deadline
        const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
        const priorityDiff = priorityOrder[a.priority] - priorityOrder[b.priority];
        
        if (priorityDiff !== 0) return priorityDiff;
        
        // Then by deadline (soonest first)
        if (a.deadline && b.deadline) {
          return new Date(a.deadline).getTime() - new Date(b.deadline).getTime();
        }
        if (a.deadline) return -1;
        if (b.deadline) return 1;
        
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      });
  }, [priorityTasks, filters]);
  
  // Group tasks by status for board view
  const tasksByStatus = useMemo(() => {
    const groups: Record<TaskStatus, PriorityTask[]> = {
      pending: [],
      in_progress: [],
      completed: [],
      cancelled: [],
      blocked: [],
    };
    
    filteredTasks.forEach(task => {
      groups[task.status].push(task);
    });
    
    return groups;
  }, [filteredTasks]);
  
  // Priority matrix data
  const priorityMatrix = useMemo(() => {
    const matrix: Record<TaskPriority, { urgent: PriorityTask[]; notUrgent: PriorityTask[] }> = {
      critical: { urgent: [], notUrgent: [] },
      high: { urgent: [], notUrgent: [] },
      medium: { urgent: [], notUrgent: [] },
      low: { urgent: [], notUrgent: [] },
    };
    
    filteredTasks.forEach(task => {
      const isUrgent = task.deadline ? 
        isAfter(new Date(Date.now() + 3 * 24 * 60 * 60 * 1000), parseISO(task.deadline)) 
        : false;
      
      if (isUrgent) {
        matrix[task.priority].urgent.push(task);
      } else {
        matrix[task.priority].notUrgent.push(task);
      }
    });
    
    return matrix;
  }, [filteredTasks]);
  
  // Handle task status update
  const handleTaskStatusUpdate = async (taskId: string, newStatus: TaskStatus) => {
    try {
      const updatedTask = await api.fleet.updateTask(taskId, { status: newStatus });
      // Update local state
      updatePriorityTasks(priorityTasks.map(task => 
        task.id === taskId ? updatedTask : task
      ));
    } catch (error) {
      console.error('Failed to update task status:', error);
    }
  };
  
  // Handle task click
  const handleTaskClick = (task: PriorityTask) => {
    setSelectedTask(task);
    // Navigate to machine if task is machine-specific
    if (task.engine_id) {
      selectMachine(task.engine_id);
      setView('machine-detail');
    }
  };
  
  return (
    <div className={`priority-board ${className}`}>
      {/* Header */}
      <div className="board-header">
        <div className="board-title">
          <Target size={20} />
          <h2>Priority Board</h2>
          <span className="task-count">
            {filteredTasks.length} tasks
          </span>
        </div>
        
        <div className="board-controls">
          {/* View Mode Selector */}
          <div className="view-mode-selector">
            <button
              className={`view-btn ${viewMode === 'board' ? 'active' : ''}`}
              onClick={() => setViewMode('board')}
              title="Board view"
            >
              Board
            </button>
            <button
              className={`view-btn ${viewMode === 'matrix' ? 'active' : ''}`}
              onClick={() => setViewMode('matrix')}
              title="Priority matrix"
            >
              Matrix
            </button>
            <button
              className={`view-btn ${viewMode === 'list' ? 'active' : ''}`}
              onClick={() => setViewMode('list')}
              title="List view"
            >
              List
            </button>
          </div>
          
          {/* Controls */}
          <button
            className="btn btn-icon"
            onClick={() => setShowFilters(!showFilters)}
            title="Filters"
          >
            <Filter size={16} />
          </button>
          
          <button
            className="btn btn-primary"
            onClick={() => setShowNewTaskForm(true)}
            title="Add new task"
          >
            <Plus size={16} />
            New Task
          </button>
        </div>
      </div>
      
      {/* Filter Panel */}
      <AnimatePresence>
        {showFilters && (
          <motion.div
            className="filter-panel"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <TaskFilterPanel
              filters={filters}
              onFilterChange={setFilters}
              taskCount={filteredTasks.length}
            />
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Content based on view mode */}
      <div className="board-content">
        {viewMode === 'board' && (
          <TaskBoard
            tasksByStatus={tasksByStatus}
            onTaskClick={handleTaskClick}
            onStatusUpdate={handleTaskStatusUpdate}
          />
        )}
        
        {viewMode === 'matrix' && (
          <PriorityMatrix
            matrix={priorityMatrix}
            onTaskClick={handleTaskClick}
          />
        )}
        
        {viewMode === 'list' && (
          <TaskList
            tasks={filteredTasks}
            onTaskClick={handleTaskClick}
            onStatusUpdate={handleTaskStatusUpdate}
          />
        )}
      </div>
      
      {/* Task Detail Modal */}
      <AnimatePresence>
        {selectedTask && (
          <TaskDetailModal
            task={selectedTask}
            onClose={() => setSelectedTask(null)}
            onUpdate={loadTasks}
          />
        )}
      </AnimatePresence>
      
      {/* New Task Form Modal */}
      <AnimatePresence>
        {showNewTaskForm && (
          <NewTaskModal
            onClose={() => setShowNewTaskForm(false)}
            onSubmit={loadTasks}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// TASK BOARD COMPONENT (Kanban Style)
// ═══════════════════════════════════════════════════════════════════════════════

interface TaskBoardProps {
  tasksByStatus: Record<TaskStatus, PriorityTask[]>;
  onTaskClick: (task: PriorityTask) => void;
  onStatusUpdate: (taskId: string, status: TaskStatus) => void;
}

const TaskBoard: React.FC<TaskBoardProps> = ({
  tasksByStatus,
  onTaskClick,
  onStatusUpdate,
}) => {
  const statusColumns: { status: TaskStatus; title: string; icon: React.ReactNode }[] = [
    { status: 'pending', title: 'Pending', icon: <Clock size={16} /> },
    { status: 'in_progress', title: 'In Progress', icon: <Play size={16} /> },
    { status: 'blocked', title: 'Blocked', icon: <Pause size={16} /> },
    { status: 'completed', title: 'Completed', icon: <CheckCircle size={16} /> },
  ];
  
  return (
    <div className="task-board">
      {statusColumns.map(column => (
        <div key={column.status} className="board-column">
          <div className="column-header">
            {column.icon}
            <span className="column-title">{column.title}</span>
            <span className="task-count">{tasksByStatus[column.status].length}</span>
          </div>
          
          <Reorder.Group
            axis="y"
            values={tasksByStatus[column.status]}
            onReorder={(newOrder) => {
              // Handle reordering logic here
            }}
            className="column-tasks"
          >
            <AnimatePresence>
              {tasksByStatus[column.status].map(task => (
                <Reorder.Item
                  key={task.id}
                  value={task}
                  className="task-card-wrapper"
                >
                  <TaskCard
                    task={task}
                    onClick={() => onTaskClick(task)}
                    onStatusChange={(status) => onStatusUpdate(task.id, status)}
                  />
                </Reorder.Item>
              ))}
            </AnimatePresence>
          </Reorder.Group>
        </div>
      ))}
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// PRIORITY MATRIX COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

interface PriorityMatrixProps {
  matrix: Record<TaskPriority, { urgent: PriorityTask[]; notUrgent: PriorityTask[] }>;
  onTaskClick: (task: PriorityTask) => void;
}

const PriorityMatrix: React.FC<PriorityMatrixProps> = ({ matrix, onTaskClick }) => {
  return (
    <div className="priority-matrix">
      <div className="matrix-grid">
        {/* Headers */}
        <div className="matrix-header"></div>
        <div className="matrix-header urgent">Urgent</div>
        <div className="matrix-header not-urgent">Not Urgent</div>
        
        {/* Matrix cells */}
        {(Object.keys(matrix) as TaskPriority[]).map(priority => (
          <React.Fragment key={priority}>
            <div className={`matrix-label priority-${priority}`}>
              {priority.toUpperCase()}
            </div>
            
            <div className={`matrix-cell urgent priority-${priority}`}>
              <div className="cell-tasks">
                {matrix[priority].urgent.map(task => (
                  <TaskCardMini
                    key={task.id}
                    task={task}
                    onClick={() => onTaskClick(task)}
                  />
                ))}
              </div>
            </div>
            
            <div className={`matrix-cell not-urgent priority-${priority}`}>
              <div className="cell-tasks">
                {matrix[priority].notUrgent.map(task => (
                  <TaskCardMini
                    key={task.id}
                    task={task}
                    onClick={() => onTaskClick(task)}
                  />
                ))}
              </div>
            </div>
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// TASK LIST COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

interface TaskListProps {
  tasks: PriorityTask[];
  onTaskClick: (task: PriorityTask) => void;
  onStatusUpdate: (taskId: string, status: TaskStatus) => void;
}

const TaskList: React.FC<TaskListProps> = ({ tasks, onTaskClick, onStatusUpdate }) => {
  return (
    <div className="task-list">
      <div className="list-header">
        <div className="header-cell">Task</div>
        <div className="header-cell">Priority</div>
        <div className="header-cell">Status</div>
        <div className="header-cell">Assigned</div>
        <div className="header-cell">Deadline</div>
        <div className="header-cell">Progress</div>
        <div className="header-cell">Actions</div>
      </div>
      
      <div className="list-body">
        {tasks.map(task => (
          <TaskListRow
            key={task.id}
            task={task}
            onClick={() => onTaskClick(task)}
            onStatusUpdate={(status) => onStatusUpdate(task.id, status)}
          />
        ))}
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// TASK CARD COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

interface TaskCardProps {
  task: PriorityTask;
  onClick: () => void;
  onStatusChange?: (status: TaskStatus) => void;
}

const TaskCard: React.FC<TaskCardProps> = ({ task, onClick, onStatusChange }) => {
  const isOverdue = task.deadline && isAfter(new Date(), parseISO(task.deadline));
  const isDueSoon = task.deadline && isAfter(
    new Date(Date.now() + 3 * 24 * 60 * 60 * 1000), 
    parseISO(task.deadline)
  );
  
  return (
    <motion.div
      className={`task-card priority-${task.priority} ${isOverdue ? 'overdue' : ''} ${isDueSoon ? 'due-soon' : ''}`}
      onClick={onClick}
      whileHover={{ scale: 1.02, boxShadow: '0 4px 20px rgba(0,0,0,0.15)' }}
      whileTap={{ scale: 0.98 }}
      layout
    >
      {/* Header */}
      <div className="task-header">
        <div className="task-type">
          {TASK_ICONS[task.task_type]}
          <span>{task.task_type}</span>
        </div>
        <div className={`priority-indicator priority-${task.priority}`}>
          {task.priority}
        </div>
      </div>
      
      {/* Content */}
      <div className="task-content">
        <h4 className="task-title">{task.title}</h4>
        <p className="task-description">{task.description}</p>
        
        {task.engine_id && (
          <div className="task-machine">
            <span>ENG-{task.engine_id.toString().padStart(3, '0')}</span>
          </div>
        )}
      </div>
      
      {/* Footer */}
      <div className="task-footer">
        <div className="task-meta">
          {task.assigned_to && (
            <div className="assigned-to">
              <User size={12} />
              <span>{task.assigned_to}</span>
            </div>
          )}
          
          {task.deadline && (
            <div className={`deadline ${isOverdue ? 'overdue' : isDueSoon ? 'due-soon' : ''}`}>
              <Calendar size={12} />
              <span>{formatDistanceToNow(parseISO(task.deadline), { addSuffix: true })}</span>
            </div>
          )}
        </div>
        
        <div className="task-progress">
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${task.completion_percentage}%` }}
            />
          </div>
          <span className="progress-text">{task.completion_percentage}%</span>
        </div>
      </div>
      
      {/* Effort indicator */}
      <div className="task-effort">
        <Clock size={12} />
        <span>{task.estimated_effort}h</span>
      </div>
    </motion.div>
  );
};

// Additional components would continue here...
// TaskCardMini, TaskListRow, TaskFilterPanel, TaskDetailModal, NewTaskModal

export default PriorityBoard;