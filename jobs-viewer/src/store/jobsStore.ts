import { create } from 'zustand'
import type { JobFilters } from '@/api/types'

type ViewMode = 'rollup' | 'faces' | 'scenes'

interface JobsState {
  // Filters
  filters: JobFilters
  setFilters: (filters: Partial<JobFilters>) => void
  resetFilters: () => void

  // View mode
  viewMode: ViewMode
  setViewMode: (mode: ViewMode) => void

  // Pagination helpers
  nextPage: () => void
  prevPage: () => void
  setPage: (page: number) => void
}

const DEFAULT_FILTERS: JobFilters = {
  limit: 20,
  offset: 0,
}

export const useJobsStore = create<JobsState>((set) => ({
  // Filters
  filters: DEFAULT_FILTERS,
  setFilters: (newFilters) =>
    set((state) => ({
      filters: { ...state.filters, ...newFilters, offset: 0 }, // Reset offset on filter change
    })),
  resetFilters: () => set({ filters: DEFAULT_FILTERS }),

  // View mode
  viewMode: 'rollup',
  setViewMode: (viewMode) => set({ viewMode }),

  // Pagination
  nextPage: () =>
    set((state) => ({
      filters: {
        ...state.filters,
        offset: (state.filters.offset || 0) + (state.filters.limit || 20),
      },
    })),
  prevPage: () =>
    set((state) => ({
      filters: {
        ...state.filters,
        offset: Math.max(0, (state.filters.offset || 0) - (state.filters.limit || 20)),
      },
    })),
  setPage: (page) =>
    set((state) => ({
      filters: {
        ...state.filters,
        offset: page * (state.filters.limit || 20),
      },
    })),
}))
