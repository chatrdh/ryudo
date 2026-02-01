import { motion } from 'framer-motion';
import clsx from 'clsx';
import './LogEntry.css';

function getAgentClass(agent) {
    const a = agent?.toLowerCase() || '';
    if (a.includes('flood') || a.includes('environmental')) return 'flood';
    if (a.includes('grid')) return 'grid';
    if (a.includes('route') || a.includes('temporal')) return 'temporal';
    if (a.includes('mission')) return 'mission';
    if (a.includes('coord')) return 'coord';
    return 'system';
}

export function LogEntry({ log }) {
    const agentClass = getAgentClass(log.agent);

    return (
        <motion.div
            className={clsx('log-entry', agentClass)}
            initial={{ opacity: 0, y: 10, height: 0 }}
            animate={{ opacity: 1, y: 0, height: 'auto' }}
            transition={{ duration: 0.2 }}
        >
            <span className="log-agent">[{log.agent}]</span>
            <span className="log-text">{log.text}</span>
        </motion.div>
    );
}
