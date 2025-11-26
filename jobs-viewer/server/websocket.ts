import { WebSocketServer, WebSocket } from 'ws'
import { Server } from 'http'
import Redis from 'ioredis'

const REDIS_URL = process.env.REDIS_URL || 'redis://vision-redis:6379/0'

interface WSMessage {
  type: string
  jobId?: string
  data?: unknown
}

export function setupWebSocket(server: Server) {
  const wss = new WebSocketServer({ server, path: '/ws' })

  // Track connected clients
  const clients = new Set<WebSocket>()

  // Setup Redis subscriber for job updates
  let subscriber: Redis | null = null

  async function connectRedis() {
    try {
      subscriber = new Redis(REDIS_URL)

      subscriber.on('connect', () => {
        console.log('[WebSocket] Redis subscriber connected')
      })

      subscriber.on('error', (err) => {
        console.error('[WebSocket] Redis error:', err.message)
      })

      // Subscribe to vision-api job status updates
      await subscriber.psubscribe('vision:job:*:status')
      console.log('[WebSocket] Subscribed to vision:job:*:status')

      subscriber.on('pmessage', (_pattern, channel, message) => {
        try {
          // Extract job_id from channel: "vision:job:{job_id}:status"
          const parts = channel.split(':')
          const jobId = parts[2]
          const data = JSON.parse(message)

          const update: WSMessage = {
            type: 'JOB_STATUS_UPDATE',
            jobId,
            data
          }

          // Broadcast to all connected clients
          broadcast(JSON.stringify(update))
        } catch (err) {
          console.error('[WebSocket] Error parsing Redis message:', err)
        }
      })
    } catch (err) {
      console.error('[WebSocket] Failed to connect to Redis:', err)
      // Retry connection after delay
      setTimeout(connectRedis, 5000)
    }
  }

  function broadcast(message: string) {
    clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(message)
      }
    })
  }

  // Start Redis connection
  connectRedis()

  wss.on('connection', (ws) => {
    console.log('[WebSocket] Client connected')
    clients.add(ws)

    // Send initial connection confirmation
    ws.send(JSON.stringify({
      type: 'CONNECTED',
      data: { message: 'WebSocket connection established' }
    }))

    ws.on('message', (message) => {
      try {
        const msg: WSMessage = JSON.parse(message.toString())
        console.log('[WebSocket] Received:', msg.type)

        switch (msg.type) {
          case 'PING':
            ws.send(JSON.stringify({ type: 'PONG' }))
            break

          case 'SUBSCRIBE_JOB':
            // Could implement per-client job subscriptions here
            // For now, all clients receive all updates
            console.log(`[WebSocket] Client subscribed to job: ${msg.jobId}`)
            break

          default:
            console.log('[WebSocket] Unknown message type:', msg.type)
        }
      } catch (err) {
        console.error('[WebSocket] Invalid message:', err)
      }
    })

    ws.on('close', () => {
      console.log('[WebSocket] Client disconnected')
      clients.delete(ws)
    })

    ws.on('error', (err) => {
      console.error('[WebSocket] Client error:', err)
      clients.delete(ws)
    })
  })

  // Cleanup on server close
  wss.on('close', () => {
    if (subscriber) {
      subscriber.quit()
    }
  })

  console.log('[WebSocket] Server initialized on /ws')
}
