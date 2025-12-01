import { useQuery, useInfiniteQuery, useQueryClient } from '@tanstack/react-query'
import { listJobs, countJobs, getJobStatus, getJobResults } from '@/api/client'
import type { JobFilters, ListJobsResponse, JobCountResponse, JobStatusResponse, JobResults } from '@/api/types'

// Query keys
export const jobsKeys = {
  all: ['jobs'] as const,
  lists: () => [...jobsKeys.all, 'list'] as const,
  list: (filters: JobFilters) => [...jobsKeys.lists(), filters] as const,
  infiniteLists: () => [...jobsKeys.all, 'infiniteList'] as const,
  infiniteList: (filters: JobFilters) => [...jobsKeys.infiniteLists(), filters] as const,
  counts: () => [...jobsKeys.all, 'count'] as const,
  count: (filters: Omit<JobFilters, 'limit' | 'offset' | 'include_results'>) => [...jobsKeys.counts(), filters] as const,
  details: () => [...jobsKeys.all, 'detail'] as const,
  detail: (id: string) => [...jobsKeys.details(), id] as const,
  status: (id: string) => [...jobsKeys.detail(id), 'status'] as const,
  results: (id: string) => [...jobsKeys.detail(id), 'results'] as const,
}

// Hook to fetch job list
export function useJobs(filters: JobFilters = {}) {
  return useQuery<ListJobsResponse>({
    queryKey: jobsKeys.list(filters),
    queryFn: () => listJobs(filters),
    refetchInterval: 5000, // Poll every 5 seconds
  })
}

// Hook to fetch job list with infinite scroll (accumulates pages)
export function useJobsInfinite(filters: JobFilters = {}) {
  return useInfiniteQuery<ListJobsResponse>({
    queryKey: jobsKeys.infiniteList(filters),
    queryFn: ({ pageParam = 0 }) =>
      listJobs({ ...filters, offset: pageParam as number, limit: filters.limit || 20 }),
    getNextPageParam: (lastPage) => {
      const hasMore = lastPage.offset + lastPage.limit < lastPage.total
      return hasMore ? lastPage.offset + lastPage.limit : undefined
    },
    initialPageParam: 0,
    refetchInterval: 5000, // Poll every 5 seconds
  })
}

// Hook to fetch job counts
export function useJobsCount(filters: Omit<JobFilters, 'limit' | 'offset' | 'include_results'> = {}) {
  return useQuery<JobCountResponse>({
    queryKey: jobsKeys.count(filters),
    queryFn: () => countJobs(filters),
    refetchInterval: 5000, // Poll every 5 seconds to stay in sync
  })
}

// Hook to fetch job status
export function useJobStatus(jobId: string, options?: { enabled?: boolean }) {
  return useQuery<JobStatusResponse>({
    queryKey: jobsKeys.status(jobId),
    queryFn: () => getJobStatus(jobId),
    enabled: options?.enabled ?? !!jobId,
    refetchInterval: (query) => {
      // Stop polling when job is completed or failed
      const data = query.state.data
      if (data?.status === 'completed' || data?.status === 'failed') {
        return false
      }
      return 2000 // Poll every 2 seconds for active jobs
    },
  })
}

// Hook to fetch job results
export function useJobResults(jobId: string, options?: { enabled?: boolean }) {
  return useQuery<JobResults>({
    queryKey: jobsKeys.results(jobId),
    queryFn: () => getJobResults(jobId),
    enabled: options?.enabled ?? !!jobId,
    staleTime: 60000, // Results don't change, cache for 1 minute
  })
}

// Hook to invalidate jobs cache
export function useInvalidateJobs() {
  const queryClient = useQueryClient()

  return {
    invalidateList: () => queryClient.invalidateQueries({ queryKey: jobsKeys.lists() }),
    invalidateJob: (jobId: string) => queryClient.invalidateQueries({ queryKey: jobsKeys.detail(jobId) }),
    invalidateAll: () => queryClient.invalidateQueries({ queryKey: jobsKeys.all }),
  }
}
