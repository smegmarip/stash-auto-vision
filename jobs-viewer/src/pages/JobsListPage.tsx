import { JobList } from '@/components/jobs/JobList'

export function JobsListPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold tracking-tight">Jobs</h1>
      </div>
      <JobList />
    </div>
  )
}
