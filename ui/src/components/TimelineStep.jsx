import { motion as Motion } from 'framer-motion';
import { Map, Droplets, Zap, Clock, Network, Truck, Check } from 'lucide-react';
import './TimelineStep.css';

const ICONS = {
    map: Map,
    water: Droplets,
    zap: Zap,
    clock: Clock,
    network: Network,
    truck: Truck,
};

export function TimelineStep({ step, index, isActive, isCompleted, isLast }) {
    const Icon = ICONS[step.icon] || Map;

    return (
        <Motion.div
            className={`step-item ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
        >
            {!isLast && <div className="step-line" />}

            <Motion.div
                className="step-icon"
                animate={isActive ? {
                    boxShadow: ['0 0 0px rgba(99, 102, 241, 0)', '0 0 20px rgba(99, 102, 241, 0.4)', '0 0 0px rgba(99, 102, 241, 0)'],
                } : {}}
                transition={{ repeat: isActive ? Infinity : 0, duration: 1.5 }}
            >
                {isCompleted ? <Check size={14} /> : <Icon size={14} />}
            </Motion.div>

            <div className="step-info">
                <div className="step-title">{step.label}</div>
                <div className="step-desc">{step.desc}</div>
            </div>
        </Motion.div>
    );
}
