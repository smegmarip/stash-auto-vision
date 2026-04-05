import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { X } from 'lucide-react'
import { useJobsStore } from '@/store/jobsStore'
import type { JobStatus, ServiceName } from '@/api/types'

export function JobFilters() {
  const { filters, setFilters, resetFilters } = useJobsStore()

  const hasActiveFilters = filters.status || filters.service

  return (
    <div className="flex flex-wrap items-center gap-3">
      {/* Status filter */}
      <Select
        value={filters.status || 'all'}
        onValueChange={(value) =>
          setFilters({ status: value === 'all' ? undefined : (value as JobStatus) })
        }
      >
        <SelectTrigger className="w-[140px]">
          <SelectValue placeholder="Status" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All statuses</SelectItem>
          <SelectItem value="completed">Completed</SelectItem>
          <SelectItem value="processing">Processing</SelectItem>
          <SelectItem value="queued">Queued</SelectItem>
          <SelectItem value="failed">Failed</SelectItem>
        </SelectContent>
      </Select>

      {/* Service filter */}
      <Select
        value={filters.service || 'all'}
        onValueChange={(value) =>
          setFilters({ service: value === 'all' ? undefined : (value as ServiceName) })
        }
      >
        <SelectTrigger className="w-[140px]">
          <SelectValue placeholder="Service" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All services</SelectItem>
          <SelectItem value="vision">Vision</SelectItem>
          <SelectItem value="faces">Faces</SelectItem>
          <SelectItem value="scenes">Scenes</SelectItem>
          <SelectItem value="semantics">Semantics</SelectItem>
          <SelectItem value="objects">Objects</SelectItem>
        </SelectContent>
      </Select>

      {/* Clear filters */}
      {hasActiveFilters && (
        <Button
          variant="ghost"
          size="sm"
          onClick={resetFilters}
          className="h-10"
        >
          <X className="h-4 w-4 mr-1" />
          Clear
        </Button>
      )}
    </div>
  )
}
