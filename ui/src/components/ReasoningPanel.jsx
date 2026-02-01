import { motion } from 'framer-motion';
import { Bot } from 'lucide-react';
import './ReasoningPanel.css';

function getAgentColor(agent) {
    const a = agent?.toLowerCase() || '';
    if (a.includes('flood') || a.includes('environmental')) return 'var(--agent-flood)';
    if (a.includes('grid')) return 'var(--agent-grid)';
    if (a.includes('route') || a.includes('temporal')) return 'var(--agent-route)';
    if (a.includes('mission')) return 'var(--agent-mission)';
    if (a.includes('coord')) return 'var(--agent-coord)';
    return 'var(--text-primary)';
}

export function ReasoningPanel({ agent, content }) {
    const color = getAgentColor(agent);

    return (
        <motion.div
            className="reasoning-block"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
        >
            <div className="reasoning-header" style={{ color }}>
                <Bot size={16} />
                <span>{agent}</span>
            </div>
            <div className="reasoning-content">
                {content}
            </div>
        </motion.div>
    );
}
