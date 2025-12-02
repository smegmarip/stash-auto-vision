import express from 'express'
import path from 'path'
import { fileURLToPath } from 'url'
import { createProxyMiddleware } from 'http-proxy-middleware'
import { createServer } from 'http'
import { setupWebSocket } from './websocket.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const app = express()
const PORT = process.env.PORT || 5020
const VISION_API_URL = process.env.VISION_API_URL || 'http://vision-api:5010'
const FRAME_SERVER_URL = process.env.FRAME_SERVER_URL || 'http://frame-server:5001'

// Logging middleware
app.use((req, _res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`)
  next()
})

// Proxy /api/vision/* to vision-api
app.use('/api/vision', createProxyMiddleware({
  target: VISION_API_URL,
  changeOrigin: true,
  pathRewrite: { '^/api/vision': '/vision' },
  onProxyReq: (proxyReq, req) => {
    console.log(`[Proxy] ${req.method} ${req.url} -> ${VISION_API_URL}`)
  },
  onError: (err, _req, res) => {
    console.error('[Proxy Error]', err.message)
    res.status(502).json({ error: 'Proxy error', message: err.message })
  }
}))

// Proxy /api/frames/* to frame-server
app.use('/api/frames', createProxyMiddleware({
  target: FRAME_SERVER_URL,
  changeOrigin: true,
  pathRewrite: { '^/api/frames': '' },
  onProxyReq: (proxyReq, req) => {
    console.log(`[Proxy] ${req.method} ${req.url} -> ${FRAME_SERVER_URL}`)
  },
  onError: (err, _req, res) => {
    console.error('[Proxy Error]', err.message)
    res.status(502).json({ error: 'Proxy error', message: err.message })
  }
}))

// Proxy external media to bypass CORS (images, videos, etc.)
app.get('/api/proxy', async (req, res) => {
  const url = req.query.url as string
  if (!url) {
    return res.status(400).json({ error: 'Missing url parameter' })
  }

  // Validate URL is external (security)
  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    return res.status(400).json({ error: 'Invalid URL' })
  }

  try {
    const response = await fetch(url)
    if (!response.ok) {
      return res.status(response.status).json({ error: 'Failed to fetch resource' })
    }

    // Forward content-type and set CORS headers
    const contentType = response.headers.get('content-type') || 'application/octet-stream'
    res.setHeader('Content-Type', contentType)
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Cache-Control', 'public, max-age=3600')

    // Stream the response
    const buffer = await response.arrayBuffer()
    res.send(Buffer.from(buffer))
  } catch (err) {
    console.error('[Proxy Error]', err)
    res.status(500).json({ error: 'Failed to proxy resource' })
  }
})

// Health check
app.get('/health', (_req, res) => {
  res.json({
    status: 'healthy',
    service: 'jobs-viewer',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  })
})

// Serve OpenAPI schema
app.get('/openapi.json', (_req, res) => {
  res.sendFile(path.join(__dirname, '../public/openapi.json'))
})

// Serve static files from the built React app
const staticPath = path.join(__dirname, '../dist/client')
app.use(express.static(staticPath))

// SPA fallback - serve index.html for all other routes
app.get('*', (req, res) => {
  const indexPath = path.join(staticPath, 'index.html')
  res.sendFile(indexPath, (err) => {
    if (err) {
      // In development, the static files might not exist
      res.status(404).json({ error: 'Not found', hint: 'Run npm run build first' })
    }
  })
})

const server = createServer(app)

// Setup WebSocket for real-time updates
setupWebSocket(server)

server.listen(PORT, () => {
  console.log(`
╔═══════════════════════════════════════════════════════════╗
║           Jobs Viewer - Stash Auto Vision                 ║
╠═══════════════════════════════════════════════════════════╣
║  Server:      http://localhost:${PORT}                       ║
║  Health:      http://localhost:${PORT}/health                ║
║  Vision API:  ${VISION_API_URL.padEnd(39)}║
║  Frame Server: ${FRAME_SERVER_URL.padEnd(38)}║
╚═══════════════════════════════════════════════════════════╝
`)
})

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down...')
  server.close(() => {
    console.log('Server closed')
    process.exit(0)
  })
})
