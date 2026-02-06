import { motion as Motion } from 'framer-motion';
import { Link, Unlink, Satellite } from 'lucide-react';
import './ConnectionStatus.css';

export function ConnectionStatus({ status }) {
    const statusConfig = {
        connecting: { icon: Satellite, text: 'Connecting...', color: 'var(--text-secondary)' },
        connected: { icon: Link, text: 'Connected', color: 'var(--status-success)' },
        disconnected: { icon: Unlink, text: 'Disconnected', color: 'var(--status-error)' },
        error: { icon: Unlink, text: 'Error', color: 'var(--status-error)' },
    };

    const config = statusConfig[status] || statusConfig.connecting;

    return (
        <Motion.div
            className="connection-status"
            style={{ borderColor: config.color }}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: status === 'connected' ? 0.6 : 1, y: 0 }}
            transition={{ duration: 0.3 }}
        >
            <config.icon size={14} style={{ color: config.color }} />
            <span>{config.text}</span>
        </Motion.div>
    );
}
