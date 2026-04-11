import { useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { ArrowLeft, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useQueryClient } from '@tanstack/react-query'
import { RollupDetailView } from '@/components/rollup/RollupDetailView'
import { FacesDetailView } from '@/components/faces/FacesDetailView'
import { ScenesDetailView } from '@/components/scenes/ScenesDetailView'
import { SemanticsDetailView } from '@/components/semantics/SemanticsDetailView'

/** Map route service param to Redis module name */
function redisModule(service?: string): string {
  switch (service) {
    case 'faces': return 'faces'
    case 'scenes': return 'scenes'
    case 'semantics': return 'semantics'
    case 'objects': return 'objects'
    default: return 'vision'
  }
}

export function JobDetailPage() {
  const { jobId, service } = useParams<{ jobId: string; service?: string }>()
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const [clearing, setClearing] = useState(false)

  if (!jobId) {
    return <div>Job ID not found</div>
  }

  const handleClearJob = async () => {
    const module = redisModule(service)
    if (!window.confirm(`Clear cached data for this ${module} job?\n\nThis cannot be undone.`)) {
      return
    }
    setClearing(true)
    try {
      const res = await fetch('/api/admin/clear-cache', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ service: module, job_id: jobId }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      console.log('Job cache cleared:', data)
      await queryClient.invalidateQueries()
      navigate('/jobs')
    } catch (err) {
      console.error('Failed to clear job cache:', err)
      window.alert(`Failed to clear job cache: ${(err as Error).message}`)
    } finally {
      setClearing(false)
    }
  }

  const getTitle = () => {
    switch (service) {
      case 'faces':
        return 'Face Recognition Results'
      case 'scenes':
        return 'Scene Detection Results'
      case 'semantics':
        return 'Tag Classification Results'
      case 'objects':
        return 'Object Detection Results'
      default:
        return 'Job Details'
    }
  }

  const renderContent = () => {
    switch (service) {
      case 'faces':
        return <FacesDetailView jobId={jobId} />
      case 'scenes':
        return <ScenesDetailView jobId={jobId} />
      case 'semantics':
        return <SemanticsDetailView jobId={jobId} />
      case 'objects':
        return (
          <div className="text-center py-12 text-muted-foreground">
            Objects service not yet implemented
          </div>
        )
      default:
        return <RollupDetailView jobId={jobId} />
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link to={service ? `/jobs/${jobId}` : '/jobs'}>
          <Button variant="ghost" size="icon">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div className="flex-1">
          <h1 className="text-3xl font-bold tracking-tight">{getTitle()}</h1>
          <p className="text-muted-foreground font-mono text-sm">{jobId}</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={handleClearJob}
          disabled={clearing}
          className="text-destructive hover:text-destructive hover:bg-destructive/10"
        >
          <Trash2 className={`h-4 w-4 mr-2 ${clearing ? 'animate-pulse' : ''}`} />
          {clearing ? 'Removing...' : 'Delete Job'}
        </Button>
      </div>
      {renderContent()}
    </div>
  )
}
