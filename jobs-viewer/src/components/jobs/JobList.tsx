import { useJobs } from '@/hooks/useJobs'
import { useJobsStore } from '@/store/jobsStore'
import { JobCard } from './JobCard'
import { JobFilters } from './JobFilters'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { ChevronLeft, ChevronRight, RefreshCw } from 'lucide-react'

export function JobList() {
  const { filters, viewMode, setViewMode, nextPage, prevPage } = useJobsStore()
  const { data, isLoading, isFetching, refetch } = useJobs(filters)

  const jobs = data?.jobs || []
  const total = data?.total || 0
  const limit = filters.limit || 20
  const offset = filters.offset || 0
  const currentPage = Math.floor(offset / limit) + 1
  const totalPages = Math.ceil(total / limit)
  const hasNextPage = offset + limit < total
  const hasPrevPage = offset > 0

  // Filter jobs by service for per-service view
  const facesJobs = jobs.filter((j) => j.service === 'faces')
  const scenesJobs = jobs.filter((j) => j.service === 'scenes')
  const semanticsJobs = jobs.filter((j) => j.service === 'semantics')

  return (
    <div className="space-y-4">
      {/* Filters and refresh */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <JobFilters />
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          disabled={isFetching}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Tabs */}
      <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as typeof viewMode)}>
        <TabsList>
          <TabsTrigger value="rollup">
            All Jobs ({total})
          </TabsTrigger>
          <TabsTrigger value="faces">
            Faces ({facesJobs.length})
          </TabsTrigger>
          <TabsTrigger value="scenes">
            Scenes ({scenesJobs.length})
          </TabsTrigger>
          <TabsTrigger value="semantics">
            Semantics ({semanticsJobs.length})
          </TabsTrigger>
        </TabsList>

        {/* Rollup view - all jobs */}
        <TabsContent value="rollup" className="mt-4">
          {isLoading ? (
            <JobListSkeleton />
          ) : jobs.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {jobs.map((job) => (
                <JobCard key={job.job_id} job={job} />
              ))}
            </div>
          )}
        </TabsContent>

        {/* Faces view */}
        <TabsContent value="faces" className="mt-4">
          {isLoading ? (
            <JobListSkeleton />
          ) : facesJobs.length === 0 ? (
            <EmptyState message="No faces jobs found" />
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {facesJobs.map((job) => (
                <JobCard key={job.job_id} job={job} />
              ))}
            </div>
          )}
        </TabsContent>

        {/* Scenes view */}
        <TabsContent value="scenes" className="mt-4">
          {isLoading ? (
            <JobListSkeleton />
          ) : scenesJobs.length === 0 ? (
            <EmptyState message="No scenes jobs found" />
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {scenesJobs.map((job) => (
                <JobCard key={job.job_id} job={job} />
              ))}
            </div>
          )}
        </TabsContent>

        {/* Semantics view */}
        <TabsContent value="semantics" className="mt-4">
          {isLoading ? (
            <JobListSkeleton />
          ) : semanticsJobs.length === 0 ? (
            <EmptyState message="No semantics jobs found" />
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {semanticsJobs.map((job) => (
                <JobCard key={job.job_id} job={job} />
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-4 pt-4">
          <Button
            variant="outline"
            size="sm"
            onClick={prevPage}
            disabled={!hasPrevPage || isLoading}
          >
            <ChevronLeft className="h-4 w-4 mr-1" />
            Previous
          </Button>
          <span className="text-sm text-muted-foreground">
            Page {currentPage} of {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={nextPage}
            disabled={!hasNextPage || isLoading}
          >
            Next
            <ChevronRight className="h-4 w-4 ml-1" />
          </Button>
        </div>
      )}
    </div>
  )
}

function JobListSkeleton() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="border rounded-lg p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Skeleton className="h-8 w-8 rounded" />
            <div className="space-y-1">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-3 w-16" />
            </div>
          </div>
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-3 w-20" />
        </div>
      ))}
    </div>
  )
}

function EmptyState({ message = 'No jobs found' }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
      <p>{message}</p>
      <p className="text-sm mt-1">Jobs will appear here when you run video analysis</p>
    </div>
  )
}
