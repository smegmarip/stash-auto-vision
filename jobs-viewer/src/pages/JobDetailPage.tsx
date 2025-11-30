import { useParams, Link } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { RollupDetailView } from '@/components/rollup/RollupDetailView'
import { FacesDetailView } from '@/components/faces/FacesDetailView'
import { ScenesDetailView } from '@/components/scenes/ScenesDetailView'
import { SemanticsDetailView } from '@/components/semantics/SemanticsDetailView'

export function JobDetailPage() {
  const { jobId, service } = useParams<{ jobId: string; service?: string }>()

  if (!jobId) {
    return <div>Job ID not found</div>
  }

  const getTitle = () => {
    switch (service) {
      case 'faces':
        return 'Face Recognition Results'
      case 'scenes':
        return 'Scene Detection Results'
      case 'semantics':
        return 'Semantic Analysis Results'
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
        <div>
          <h1 className="text-3xl font-bold tracking-tight">{getTitle()}</h1>
          <p className="text-muted-foreground font-mono text-sm">{jobId}</p>
        </div>
      </div>
      {renderContent()}
    </div>
  )
}
