import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { MapView } from './components/MapContainer';
import { Sidebar } from './components/Sidebar';
import { ConnectionStatus } from './components/ConnectionStatus';
import './App.css';

const STEPS = [
  { id: 'init', label: 'Initialization' },
  { id: 'flood', label: 'Environmental' },
  { id: 'infra', label: 'Infrastructure' },
  { id: 'temporal', label: 'Temporal' },
  { id: 'coord', label: 'Coordinator' },
  { id: 'mission', label: 'Mission Solver' },
];

function getStepIndex(agentName) {
  if (!agentName) return -1;
  const a = agentName.toLowerCase();
  if (a.includes('flood') || a.includes('environmental')) return 1;
  if (a.includes('grid')) return 2;
  if (a.includes('route') || a.includes('temporal')) return 3;
  if (a.includes('coord')) return 4;
  if (a.includes('mission')) return 5;
  return 0;
}

export default function App() {
  const { status, messages, send, clearMessages } = useWebSocket();

  // UI State
  const [activeStep, setActiveStep] = useState(-1);
  const [completedSteps, setCompletedSteps] = useState([]);
  const [logs, setLogs] = useState([]);
  const [reasoning, setReasoning] = useState([]);
  const [constraints, setConstraints] = useState([]);
  const [markers, setMarkers] = useState([]);
  const [routes, setRoutes] = useState([]);
  const [speed, setSpeed] = useState(1.0);
  const [isRunning, setIsRunning] = useState(false);

  // Process WebSocket messages
  useEffect(() => {
    if (messages.length === 0) return;

    const latestMessage = messages[messages.length - 1];
    processMessage(latestMessage);
  }, [messages]);

  const processMessage = useCallback((msg) => {
    switch (msg.type) {
      case 'workflow_start':
        resetState();
        setActiveStep(0);
        setIsRunning(true);
        addLog('System', 'Workflow Started');
        break;

      case 'agent_start':
        const stepIdx = getStepIndex(msg.agent);
        if (stepIdx !== -1) {
          // Complete all previous steps
          setCompletedSteps(prev => {
            const newCompleted = [...prev];
            for (let i = 0; i < stepIdx; i++) {
              if (!newCompleted.includes(i)) newCompleted.push(i);
            }
            return newCompleted;
          });
          setActiveStep(stepIdx);
        }
        addLog(msg.agent, msg.message);
        break;

      case 'agent_complete':
        setCompletedSteps(prev =>
          prev.includes(activeStep) ? prev : [...prev, activeStep]
        );
        break;

      case 'reasoning':
        setReasoning(prev => [...prev, { agent: msg.agent, content: msg.content }]);
        addLog(msg.agent, 'Reasoning generated');
        break;

      case 'constraint':
        setConstraints(prev => [...prev, msg.constraint]);
        break;

      case 'marker':
        setMarkers(prev => [...prev, msg]);
        break;

      case 'route_complete':
        setRoutes(prev => [...prev, { coords: msg.route, distance: msg.total_distance_km }]);
        addLog('MissionSolver', `Route computed: ${msg.total_distance_km}km`);
        break;

      case 'workflow_complete':
        setActiveStep(STEPS.length - 1);
        setCompletedSteps(prev => {
          const all = [];
          for (let i = 0; i < STEPS.length; i++) all.push(i);
          return all;
        });
        setIsRunning(false);
        addLog('System', 'Workflow Complete');
        break;

      case 'reset':
        resetState();
        addLog('System', 'Reset complete');
        break;
    }
  }, [activeStep]);

  const addLog = (agent, text) => {
    setLogs(prev => [...prev, { agent, text, timestamp: Date.now() }]);
  };

  const resetState = () => {
    setActiveStep(-1);
    setCompletedSteps([]);
    setLogs([]);
    setReasoning([]);
    setConstraints([]);
    setMarkers([]);
    setRoutes([]);
    setIsRunning(false);
    clearMessages();
  };

  const handleStart = () => {
    send({ action: 'start' });
    setIsRunning(true);
  };

  const handleReset = () => {
    send({ action: 'reset' });
    resetState();
  };

  const handleSpeedChange = (newSpeed) => {
    setSpeed(newSpeed);
    send({ action: 'set_speed', speed: newSpeed });
  };

  return (
    <div className="app">
      <div className="map-section">
        <MapView
          constraints={constraints}
          markers={markers}
          routes={routes}
        />
        <ConnectionStatus status={status} />
      </div>

      <Sidebar
        activeStep={activeStep}
        completedSteps={completedSteps}
        logs={logs}
        reasoning={reasoning}
        speed={speed}
        isRunning={isRunning}
        onStart={handleStart}
        onReset={handleReset}
        onSpeedChange={handleSpeedChange}
      />
    </div>
  );
}
