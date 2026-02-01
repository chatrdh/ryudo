import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Custom hook for WebSocket connection with automatic reconnection.
 */
export function useWebSocket() {
    const [status, setStatus] = useState('connecting');
    const [messages, setMessages] = useState([]);
    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);

    const connect = useCallback(() => {
        const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const wsUrl = `${proto}://${window.location.host}/ws`;

        try {
            wsRef.current = new WebSocket(wsUrl);

            wsRef.current.onopen = () => {
                setStatus('connected');
                console.log('[WS] Connected');
            };

            wsRef.current.onclose = () => {
                setStatus('disconnected');
                console.log('[WS] Disconnected, reconnecting...');
                reconnectTimeoutRef.current = setTimeout(connect, 2000);
            };

            wsRef.current.onerror = (error) => {
                console.error('[WS] Error:', error);
                setStatus('error');
            };

            wsRef.current.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    setMessages(prev => [...prev, data]);
                } catch (e) {
                    console.error('[WS] Parse error:', e);
                }
            };
        } catch (error) {
            console.error('[WS] Connection error:', error);
            setStatus('error');
        }
    }, []);

    useEffect(() => {
        connect();

        return () => {
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [connect]);

    const send = useCallback((data) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(data));
        }
    }, []);

    const clearMessages = useCallback(() => {
        setMessages([]);
    }, []);

    return { status, messages, send, clearMessages };
}
