import type {
  ListJobsResponse,
  JobFilters,
  JobStatusResponse,
  JobResults,
} from './types'

const API_BASE = '/api/vision'

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: unknown
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  })

  if (!response.ok) {
    let data
    try {
      data = await response.json()
    } catch {
      data = { message: response.statusText }
    }
    throw new ApiError(
      data.message || `API Error: ${response.status}`,
      response.status,
      data
    )
  }

  return response.json()
}

// Build query string from filters object
function buildQueryString(filters: JobFilters): string {
  const params = new URLSearchParams()

  for (const [key, value] of Object.entries(filters)) {
    if (value !== undefined && value !== null && value !== '') {
      params.append(key, String(value))
    }
  }

  const queryString = params.toString()
  return queryString ? `?${queryString}` : ''
}

// ============================================
// Jobs API
// ============================================

export async function listJobs(filters: JobFilters = {}): Promise<ListJobsResponse> {
  const queryString = buildQueryString(filters)
  return fetchApi<ListJobsResponse>(`/jobs${queryString}`)
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  return fetchApi<JobStatusResponse>(`/jobs/${jobId}/status`)
}

export async function getJobResults(jobId: string): Promise<JobResults> {
  return fetchApi<JobResults>(`/jobs/${jobId}/results`)
}

// ============================================
// URL Helpers
// ============================================

export function isExternalUrl(path: string): boolean {
  return path.startsWith('http://') || path.startsWith('https://')
}

export function getImageUrl(
  source: string,
  timestamp: number,
  options: { enhance?: boolean } = {}
): string {
  if (isExternalUrl(source)) {
    // Route external URLs through proxy to avoid CORS issues
    return `/api/proxy?url=${encodeURIComponent(source)}`
  }
  return getFrameUrl(source, timestamp, options)
}

// ============================================
// Frame API
// ============================================

export function getFrameUrl(
  videoPath: string,
  timestamp: number,
  options: {
    format?: 'jpeg' | 'png'
    quality?: number
    enhance?: boolean
  } = {}
): string {
  const params = new URLSearchParams({
    video_path: videoPath,
    timestamp: String(timestamp),
    output_format: options.format || 'jpeg',
  })

  if (options.quality) {
    params.append('quality', String(options.quality))
  }

  if (options.enhance) {
    params.append('enhance', '1')
  }

  return `/api/frames/extract-frame?${params.toString()}`
}

// ============================================
// Export
// ============================================

export { ApiError }
