import { useState } from 'react';
import { motion as Motion, AnimatePresence } from 'framer-motion';
import { Play, RotateCcw, Zap, Clock, FileText, Brain } from 'lucide-react';
import { TimelineStep } from './TimelineStep';
import { LogEntry } from './LogEntry';
import { ReasoningPanel } from './ReasoningPanel';
import './Sidebar.css';

const STEPS = [
    { id: 'init', label: 'Initialization', desc: 'Loading localized map data', icon: 'map' },
    { id: 'flood', label: 'Environmental', desc: 'FloodSentinel analyzing cyclone data', icon: 'water' },
    { id: 'infra', label: 'Infrastructure', desc: 'GridGuardian checking power failures', icon: 'zap' },
    { id: 'temporal', label: 'Temporal', desc: 'RoutePilot predicting time constraints', icon: 'clock' },
    { id: 'coord', label: 'Coordinator', desc: 'Syncing constraints & refining graph', icon: 'network' },
    { id: 'mission', label: 'Mission Solver', desc: 'Calculating optimal rescue routes', icon: 'truck' },
];

const TABS = [
    { id: 'timeline', label: 'Timeline', icon: Clock },
    { id: 'logs', label: 'Live Logs', icon: FileText },
    { id: 'reasoning', label: 'Reasoning', icon: Brain },
];

export function Sidebar({
    activeStep,
    completedSteps,
    logs,
    reasoning,
    speed,
    isRunning,
    onStart,
    onReset,
    onSpeedChange,
}) {
    const [activeTab, setActiveTab] = useState('timeline');

    return (
        <aside className="sidebar">
            {/* Header */}
            <header className="sidebar-header">
                <Motion.h1
                    className="sidebar-title"
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    Ryudo
                </Motion.h1>
                <p className="sidebar-subtitle">Intelligent Disaster Response System</p>
            </header>

            {/* Controls */}
            <div className="controls">
                <Motion.button
                    className="btn btn-primary"
                    onClick={onStart}
                    disabled={isRunning}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                >
                    <Play size={16} />
                    Run
                </Motion.button>

                <Motion.button
                    className="btn btn-secondary"
                    onClick={onReset}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                >
                    <RotateCcw size={16} />
                </Motion.button>

                <div className="speed-control">
                    <div className="speed-slider-container">
                        <Zap size={12} className="speed-icon" />
                        <input
                            type="range"
                            min="0.5"
                            max="5"
                            step="0.5"
                            value={speed}
                            onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
                            className="speed-slider"
                        />
                        <span className="speed-value">{speed}x</span>
                    </div>
                </div>
            </div>

            {/* Tabs */}
            <nav className="tabs">
                {TABS.map((tab) => (
                    <button
                        key={tab.id}
                        className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        <tab.icon size={14} />
                        {tab.label}
                    </button>
                ))}
            </nav>

            {/* Tab Content */}
            <div className="content-area">
                <AnimatePresence mode="wait">
                    {activeTab === 'timeline' && (
                        <Motion.div
                            key="timeline"
                            className="tab-content"
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 10 }}
                            transition={{ duration: 0.2 }}
                        >
                            <div className="step-list">
                                {STEPS.map((step, idx) => (
                                    <TimelineStep
                                        key={step.id}
                                        step={step}
                                        index={idx}
                                        isActive={activeStep === idx}
                                        isCompleted={completedSteps.includes(idx)}
                                        isLast={idx === STEPS.length - 1}
                                    />
                                ))}
                            </div>
                        </Motion.div>
                    )}

                    {activeTab === 'logs' && (
                        <Motion.div
                            key="logs"
                            className="tab-content"
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 10 }}
                            transition={{ duration: 0.2 }}
                        >
                            <div className="log-container" id="logContainer">
                                {logs.length === 0 ? (
                                    <div className="log-empty">
                                        System ready. Waiting to start...
                                    </div>
                                ) : (
                                    logs.map((log, idx) => (
                                        <LogEntry key={idx} log={log} />
                                    ))
                                )}
                            </div>
                        </Motion.div>
                    )}

                    {activeTab === 'reasoning' && (
                        <Motion.div
                            key="reasoning"
                            className="tab-content"
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 10 }}
                            transition={{ duration: 0.2 }}
                        >
                            {reasoning.length === 0 ? (
                                <div className="reasoning-empty">
                                    <Brain size={32} />
                                    <p>Reasoning traces will appear here when agents are active.</p>
                                </div>
                            ) : (
                                reasoning.map((r, idx) => (
                                    <ReasoningPanel key={idx} agent={r.agent} content={r.content} />
                                ))
                            )}
                        </Motion.div>
                    )}
                </AnimatePresence>
            </div>
        </aside>
    );
}
