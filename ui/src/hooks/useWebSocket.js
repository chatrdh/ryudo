import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Managed WebSocket hook with reconnection and graceful cleanup.
 */
export function useWebSocket() {
  const [status, setStatus] = useState('connecting');
  const [messages, setMessages] = useState([]);

  const wsRef = useRef(null);
  const connectRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const shouldReconnectRef = useRef(true);

  const scheduleReconnect = useCallback(() => {
    if (!shouldReconnectRef.current) {
      return;
    }

    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
    }

    reconnectTimerRef.current = setTimeout(() => {
      connectRef.current?.();
    }, 1800);
  }, []);

  const connect = useCallback(() => {
    if (!shouldReconnectRef.current) {
      return;
    }

    const customUrl = import.meta.env.VITE_WS_URL;
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = customUrl || `${protocol}://${window.location.host}/ws`;

    setStatus('connecting');

    try {
      const socket = new WebSocket(wsUrl);
      wsRef.current = socket;

      socket.onopen = () => {
        setStatus('connected');
      };

      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          setMessages((prev) => [...prev, payload]);
        } catch (error) {
          console.error('[WS] Failed to parse message:', error);
        }
      };

      socket.onerror = (error) => {
        console.error('[WS] Socket error:', error);
        setStatus('error');
      };

      socket.onclose = () => {
        wsRef.current = null;
        if (!shouldReconnectRef.current) {
          return;
        }
        setStatus('disconnected');
        scheduleReconnect();
      };
    } catch (error) {
      console.error('[WS] Connection failed:', error);
      setStatus('error');
      scheduleReconnect();
    }
  }, [scheduleReconnect]);

  useEffect(() => {
    shouldReconnectRef.current = true;
    connectRef.current = connect;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    connect();

    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  const send = useCallback((data) => {
    const socket = wsRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return false;
    }

    socket.send(JSON.stringify(data));
    return true;
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return {
    status,
    messages,
    send,
    clearMessages,
  };
}
