import { useCallback, useEffect, useMemo, useState } from 'react';
import { MapView } from './components/MapContainer';
import { useWebSocket } from './hooks/useWebSocket';
import './App.css';

const STEP_DEFS = [
  {
    key: 'load_graph',
    label: 'Load Base Graph',
    description: 'Initialize baseline road network and map context.',
  },
  {
    key: 'environmental',
    label: 'FloodSentinel',
    description: 'Build dynamic weather and surge zone constraints.',
  },
  {
    key: 'infrastructure',
    label: 'GridGuardian',
    description: 'Evaluate power outages and facility availability.',
  },
  {
    key: 'temporal',
    label: 'RoutePilot',
    description: 'Predict route validity and TTL windows.',
  },
  {
    key: 'coordinator',
    label: 'Coordinator',
    description: 'Merge constraints and evaluate mission feasibility.',
  },
  {
    key: 'mission',
    label: 'MissionSolver',
    description: 'Produce final assignments and rescue routes.',
  },
];

const INITIAL_STEPS = Object.freeze(
  STEP_DEFS.reduce((acc, step) => {
    acc[step.key] = 'pending';
    return acc;
  }, {})
);

function formatClock(timestamp) {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function statusLabel(status) {
  if (status === 'connected') return 'Connected';
  if (status === 'connecting') return 'Connecting';
  if (status === 'disconnected') return 'Disconnected';
  return 'Error';
}

function resolveStepKey(msg) {
  const step = String(msg.step || '').toLowerCase();
  if (step === 'load_graph' || step === 'routing') {
    return step === 'load_graph' ? 'load_graph' : 'mission';
  }

  const agent = String(msg.agent || '').toLowerCase();
  if (agent.includes('flood') || agent.includes('environmental')) return 'environmental';
  if (agent.includes('grid') || agent.includes('infrastructure')) return 'infrastructure';
  if (agent.includes('route') || agent.includes('temporal')) return 'temporal';
  if (agent.includes('coord')) return 'coordinator';
  if (agent.includes('mission')) return 'mission';
  return null;
}

function createEvent(source, text, type = 'info') {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    source,
    text,
    type,
    timestamp: Date.now(),
  };
}

function withCap(items, cap = 500) {
  if (items.length <= cap) return items;
  return items.slice(items.length - cap);
}

export default function App() {
  const { status, messages, send, clearMessages } = useWebSocket();

  const [isRunning, setIsRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [stepStatus, setStepStatus] = useState(INITIAL_STEPS);
  const [constraints, setConstraints] = useState([]);
  const [markers, setMarkers] = useState([]);
  const [routes, setRoutes] = useState([]);
  const [events, setEvents] = useState([]);
  const [reasoning, setReasoning] = useState([]);
  const [coordinatorStats, setCoordinatorStats] = useState(null);
  const [missionSummary, setMissionSummary] = useState(null);
  const [workflowSummary, setWorkflowSummary] = useState(null);

  const resetWorkflowState = useCallback(
    (clearSocketMessages = true) => {
      setIsRunning(false);
      setStepStatus(INITIAL_STEPS);
      setConstraints([]);
      setMarkers([]);
      setRoutes([]);
      setEvents([]);
      setReasoning([]);
      setCoordinatorStats(null);
      setMissionSummary(null);
      setWorkflowSummary(null);
      if (clearSocketMessages) {
        clearMessages();
      }
    },
    [clearMessages]
  );

  const updateStep = useCallback((key, value) => {
    if (!key) return;
    setStepStatus((prev) => {
      if (prev[key] === value) return prev;
      return { ...prev, [key]: value };
    });
  }, []);

  const completeAllSteps = useCallback(() => {
    setStepStatus((prev) => {
      const next = { ...prev };
      STEP_DEFS.forEach((step) => {
        next[step.key] = 'complete';
      });
      return next;
    });
  }, []);

  const appendEvent = useCallback((source, text, type = 'info') => {
    setEvents((prev) => withCap([...prev, createEvent(source, text, type)]));
  }, []);

  const appendReasoning = useCallback((msg) => {
    setReasoning((prev) =>
      withCap([
        ...prev,
        {
          id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
          agent: msg.agent || 'Agent',
          content: msg.content || '',
          timestamp: Date.now(),
        },
      ], 120)
    );
  }, []);

  const processMessage = useCallback(
    (msg) => {
      const source = msg.agent || 'System';
      const messageText = msg.message || msg.reason || msg.type;

      switch (msg.type) {
        case 'connected': {
          appendEvent('System', 'WebSocket connected. Console ready.', 'success');
          break;
        }

        case 'workflow_start': {
          resetWorkflowState(false);
          setIsRunning(true);
          appendEvent('System', msg.message || 'Workflow started.', 'success');
          break;
        }

        case 'step': {
          const stepKey = resolveStepKey(msg);
          if (stepKey) updateStep(stepKey, 'active');
          appendEvent(source, messageText, 'info');
          break;
        }

        case 'step_complete': {
          const stepKey = resolveStepKey(msg);
          if (stepKey) updateStep(stepKey, 'complete');
          appendEvent(source, messageText, 'success');
          break;
        }

        case 'agent_start': {
          const stepKey = resolveStepKey(msg);
          if (stepKey) updateStep(stepKey, 'active');
          appendEvent(source, messageText, 'info');
          break;
        }

        case 'agent_complete': {
          const stepKey = resolveStepKey(msg);
          if (stepKey) updateStep(stepKey, 'complete');
          appendEvent(source, messageText, 'success');
          break;
        }

        case 'reasoning': {
          appendReasoning(msg);
          appendEvent(source, 'Reasoning trace received.', 'info');
          break;
        }

        case 'constraint': {
          if (msg.constraint) {
            setConstraints((prev) => [...prev, msg.constraint]);
          }
          break;
        }

        case 'marker': {
          if (msg.position) {
            setMarkers((prev) => [...prev, msg]);
          }
          break;
        }

        case 'route_complete': {
          const route = {
            route: msg.route || [],
            vehicle_name: msg.vehicle_name,
            total_distance_km: msg.total_distance_km,
            travel_time_min: msg.travel_time_min,
            target_names: msg.target_names || [],
            color: msg.color,
            popup: msg.popup,
          };
          setRoutes((prev) => [...prev, route]);
          appendEvent(
            source,
            `${msg.vehicle_name || 'Vehicle'} route added (${msg.total_distance_km || 0} km).`,
            'success'
          );
          break;
        }

        case 'coordinator_stats': {
          setCoordinatorStats(msg.stats || null);
          appendEvent('Coordinator', 'Constraint merge stats updated.', 'info');
          break;
        }

        case 'mission_summary': {
          setMissionSummary(msg);
          appendEvent('MissionSolver', 'Mission summary published.', 'success');
          break;
        }

        case 'workflow_complete': {
          completeAllSteps();
          setWorkflowSummary(msg);
          setIsRunning(false);
          appendEvent('System', msg.message || 'Workflow complete.', 'success');
          break;
        }

        case 'reset': {
          resetWorkflowState(false);
          appendEvent('System', 'Workflow reset completed.', 'warning');
          break;
        }

        default: {
          if (messageText) {
            appendEvent(source, messageText, 'info');
          }
          break;
        }
      }
    },
    [
      appendEvent,
      appendReasoning,
      completeAllSteps,
      resetWorkflowState,
      updateStep,
    ]
  );

  useEffect(() => {
    if (!messages.length) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    processMessage(messages[messages.length - 1]);
  }, [messages, processMessage]);

  const handleStart = useCallback(async () => {
    setIsRunning(true);
    const sent = send({ action: 'start' });
    if (!sent) {
      try {
        await fetch('/api/start', { method: 'POST' });
      } catch {
        appendEvent('System', 'Failed to start workflow over HTTP fallback.', 'error');
      }
    }
  }, [appendEvent, send]);

  const handleReset = useCallback(() => {
    send({ action: 'reset' });
    resetWorkflowState(false);
  }, [resetWorkflowState, send]);

  const handleSpeedChange = useCallback(
    (nextSpeed) => {
      setSpeed(nextSpeed);
      send({ action: 'set_speed', speed: nextSpeed });
    },
    [send]
  );

  const completedSteps = useMemo(
    () => Object.values(stepStatus).filter((state) => state === 'complete').length,
    [stepStatus]
  );

  const missionMetrics = useMemo(() => {
    if (missionSummary) {
      return {
        targets: `${missionSummary.targets_assigned || 0}/${missionSummary.total_targets || 0}`,
        rescued: missionSummary.population_rescued || 0,
        distance: missionSummary.total_distance_km || 0,
        deployed: missionSummary.vehicles_deployed || 0,
      };
    }

    if (workflowSummary?.mission_result) {
      return {
        targets: workflowSummary.mission_result.targets_assigned || 0,
        rescued: workflowSummary.mission_result.population_rescued || 0,
        distance: workflowSummary.mission_result.total_distance_km || 0,
        deployed: routes.length,
      };
    }

    return {
      targets: '0/0',
      rescued: 0,
      distance: 0,
      deployed: routes.length,
    };
  }, [missionSummary, routes.length, workflowSummary]);

  const recentEvents = useMemo(() => [...events].reverse(), [events]);
  const recentReasoning = useMemo(() => [...reasoning].reverse(), [reasoning]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <p className="eyebrow">Ryudo Operations</p>
          <h1>Dynamic Spatial Mission Console</h1>
        </div>

        <div className={`status-chip status-${status}`}>
          <span className="status-dot" />
          {statusLabel(status)}
        </div>

        <div className="control-strip">
          <button className="btn primary" onClick={handleStart} disabled={isRunning}>
            Run Workflow
          </button>
          <button className="btn ghost" onClick={handleReset}>
            Reset
          </button>
          <label className="speed-control" htmlFor="speed-range">
            Speed {speed.toFixed(1)}x
            <input
              id="speed-range"
              type="range"
              min="0.5"
              max="5"
              step="0.5"
              value={speed}
              onChange={(event) => handleSpeedChange(parseFloat(event.target.value))}
            />
          </label>
        </div>
      </header>

      <main className="workspace">
        <section className="map-pane">
          <MapView constraints={constraints} markers={markers} routes={routes} />

          <div className="overlay-stats">
            <article className="stat-card">
              <h3>Workflow</h3>
              <p>{completedSteps}/{STEP_DEFS.length} steps</p>
            </article>
            <article className="stat-card">
              <h3>Constraints</h3>
              <p>{constraints.length}</p>
            </article>
            <article className="stat-card">
              <h3>Markers</h3>
              <p>{markers.length}</p>
            </article>
            <article className="stat-card">
              <h3>Routes</h3>
              <p>{routes.length}</p>
            </article>
          </div>
        </section>

        <aside className="side-pane">
          <section className="panel">
            <div className="panel-header">
              <h2>Workflow Timeline</h2>
            </div>
            <ul className="timeline-list">
              {STEP_DEFS.map((step) => (
                <li key={step.key} className={`timeline-item ${stepStatus[step.key]}`}>
                  <div className="timeline-title-row">
                    <span className="timeline-title">{step.label}</span>
                    <span className="timeline-state">{stepStatus[step.key]}</span>
                  </div>
                  <p className="timeline-description">{step.description}</p>
                </li>
              ))}
            </ul>
          </section>

          <section className="panel">
            <div className="panel-header">
              <h2>Mission Snapshot</h2>
            </div>
            <div className="metric-grid">
              <article>
                <h3>Targets</h3>
                <p>{missionMetrics.targets}</p>
              </article>
              <article>
                <h3>Population</h3>
                <p>{missionMetrics.rescued}</p>
              </article>
              <article>
                <h3>Distance</h3>
                <p>{missionMetrics.distance} km</p>
              </article>
              <article>
                <h3>Vehicles</h3>
                <p>{missionMetrics.deployed}</p>
              </article>
            </div>

            {coordinatorStats && (
              <div className="coordinator-stats">
                <h3>Coordinator Stats</h3>
                <ul>
                  <li>Zones deleted: {coordinatorStats.zones_deleted ?? 0}</li>
                  <li>Edges removed: {coordinatorStats.edges_removed ?? 0}</li>
                  <li>Nodes disabled: {coordinatorStats.nodes_disabled ?? 0}</li>
                  <li>TTL applied: {coordinatorStats.ttl_applied ?? 0}</li>
                </ul>
              </div>
            )}
          </section>

          <section className="panel scrollable">
            <div className="panel-header">
              <h2>Live Event Feed</h2>
            </div>
            {recentEvents.length === 0 ? (
              <p className="empty">No events yet. Start the workflow to stream updates.</p>
            ) : (
              <ul className="event-list">
                {recentEvents.map((entry) => (
                  <li key={entry.id} className={`event-item ${entry.type}`}>
                    <div className="event-meta">
                      <span className="event-source">{entry.source}</span>
                      <span className="event-time">{formatClock(entry.timestamp)}</span>
                    </div>
                    <p>{entry.text}</p>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="panel scrollable">
            <div className="panel-header">
              <h2>Agent Reasoning</h2>
            </div>
            {recentReasoning.length === 0 ? (
              <p className="empty">No reasoning traces yet.</p>
            ) : (
              <div className="reasoning-list">
                {recentReasoning.map((trace) => (
                  <details key={trace.id} className="reasoning-item" open={false}>
                    <summary>
                      <span>{trace.agent}</span>
                      <time>{formatClock(trace.timestamp)}</time>
                    </summary>
                    <pre>{trace.content}</pre>
                  </details>
                ))}
              </div>
            )}
          </section>
        </aside>
      </main>
    </div>
  );
}
