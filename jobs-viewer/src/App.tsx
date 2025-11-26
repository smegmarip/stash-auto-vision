import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MainLayout } from './components/layout/MainLayout'
import { JobsListPage } from './pages/JobsListPage'
import { JobDetailPage } from './pages/JobDetailPage'
import { useWebSocket } from './hooks/useWebSocket'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      refetchOnWindowFocus: false,
    },
  },
})

// WebSocket connection manager
function WebSocketManager() {
  useWebSocket({ enabled: true })
  return null
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <WebSocketManager />
      <BrowserRouter>
        <MainLayout>
          <Routes>
            <Route path="/" element={<Navigate to="/jobs" replace />} />
            <Route path="/jobs" element={<JobsListPage />} />
            <Route path="/jobs/:jobId" element={<JobDetailPage />} />
            <Route path="/jobs/:jobId/:service" element={<JobDetailPage />} />
          </Routes>
        </MainLayout>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
