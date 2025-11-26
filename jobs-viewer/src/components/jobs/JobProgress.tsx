import { cn } from '@/lib/utils'

interface JobProgressProps {
  progress: number
  className?: string
}

export function JobProgress({ progress, className }: JobProgressProps) {
  const percentage = Math.round(progress * 100)

  return (
    <div className={cn('w-full', className)}>
      <div className="flex justify-between text-xs text-muted-foreground mb-1">
        <span>Progress</span>
        <span>{percentage}%</span>
      </div>
      <div className="h-2 bg-muted rounded-full overflow-hidden">
        <div
          className="h-full bg-primary transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  )
}
