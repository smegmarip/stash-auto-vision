import { useEffect, useRef } from "react";
import { useJobsStore } from "@/store/jobsStore";

interface LazyLoaderProps {
  hasMore: boolean;
  isLoading: boolean;
  onLoadMore?: () => void; // Optional callback for custom load behavior
}

export function LazyLoader({ hasMore, isLoading, onLoadMore }: LazyLoaderProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const { nextPage } = useJobsStore();

  useEffect(() => {
    if (!hasMore || isLoading) return;
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) {
        // Use custom callback if provided, otherwise use store's nextPage
        if (onLoadMore) {
          onLoadMore();
        } else {
          nextPage();
        }
      }
    });

    observer.observe(el);
    return () => observer.disconnect();
  }, [hasMore, isLoading, onLoadMore, nextPage]);

  return <div ref={ref} className="h-2" />;
}

export default LazyLoader;
