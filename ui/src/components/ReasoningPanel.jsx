import { motion } from 'framer-motion';
import { Bot } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
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

// Clean up LaTeX for display (convert to readable format)
function cleanContent(content) {
    if (!content) return '';
    return content
        // Convert LaTeX fractions to readable format
        .replace(/\$([^$]+)\$/g, (match, inner) => {
            // Simple cleanup - remove backslashes and make readable
            return inner
                .replace(/\\text\{([^}]+)\}/g, '$1')
                .replace(/\\times/g, '×')
                .replace(/\\sqrt\{([^}]+)\}/g, '√($1)')
                .replace(/\\exp\(([^)]+)\)/g, 'exp($1)')
                .replace(/\^(\d+)/g, '^$1')
                .replace(/\_\{([^}]+)\}/g, '[$1]')
                .replace(/\\/g, '');
        })
        // Clean up double dollar LaTeX blocks
        .replace(/\$\$([^$]+)\$\$/g, (match, inner) => `\n${inner.replace(/\\/g, '')}\n`);
}

export function ReasoningPanel({ agent, content }) {
    const color = getAgentColor(agent);
    const cleanedContent = cleanContent(content);

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
                <ReactMarkdown>{cleanedContent}</ReactMarkdown>
            </div>
        </motion.div>
    );
}

