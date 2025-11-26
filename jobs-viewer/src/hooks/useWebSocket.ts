import { useEffect, useRef, useCallback, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { jobsKeys } from './useJobs'
import type { WSMessage } from '@/api/types'

interface UseWebSocketOptions {
  enabled?: boolean
  reconnectInterval?: number
  maxReconnectAttempts?: number
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    enabled = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
  } = options

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const queryClient = useQueryClient()
  const [connected, setConnected] = useState(false)

  const connect = useCallback(() => {
    if (!enabled) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws`

    try {
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        console.log('[WebSocket] Connected')
        setConnected(true)
        reconnectAttemptsRef.current = 0
      }

      ws.onmessage = (event) => {
        try {
          const message: WSMessage = JSON.parse(event.data)

          switch (message.type) {
            case 'CONNECTED':
              console.log('[WebSocket] Connection confirmed')
              break

            case 'JOB_STATUS_UPDATE':
              console.log('[WebSocket] Job update:', message.jobId)
              // Invalidate the specific job's queries
              if (message.jobId) {
                queryClient.invalidateQueries({ queryKey: jobsKeys.status(message.jobId) })
              }
              // Also invalidate the jobs list for updated totals
              queryClient.invalidateQueries({ queryKey: jobsKeys.lists() })
              break

            case 'PONG':
              // Heartbeat response
              break
          }
        } catch (err) {
          console.error('[WebSocket] Failed to parse message:', err)
        }
      }

      ws.onclose = (event) => {
        console.log('[WebSocket] Disconnected:', event.code, event.reason)
        setConnected(false)
        wsRef.current = null

        // Attempt to reconnect
        if (enabled && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++
          console.log(`[WebSocket] Reconnecting (attempt ${reconnectAttemptsRef.current})...`)
          setTimeout(connect, reconnectInterval)
        }
      }

      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error)
      }
    } catch (err) {
      console.error('[WebSocket] Failed to connect:', err)
    }
  }, [enabled, reconnectInterval, maxReconnectAttempts, queryClient])

  // Send a ping to keep connection alive
  const ping = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'PING' }))
    }
  }, [])

  // Subscribe to specific job updates
  const subscribeToJob = useCallback((jobId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'SUBSCRIBE_JOB', jobId }))
    }
  }, [])

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect()

    // Ping every 30 seconds to keep connection alive
    const pingInterval = setInterval(ping, 30000)

    return () => {
      clearInterval(pingInterval)
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect, ping])

  return {
    connected,
    subscribeToJob,
    reconnect: connect,
  }
}
