import { Link } from 'react-router-dom'
import { Eye } from 'lucide-react'

export function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-14 items-center px-4">
        <Link to="/" className="flex items-center space-x-2">
          <Eye className="h-6 w-6" />
          <span className="font-bold">Jobs Viewer</span>
        </Link>
        <nav className="ml-6 flex items-center space-x-4 text-sm font-medium">
          <Link
            to="/jobs"
            className="transition-colors hover:text-foreground/80 text-foreground"
          >
            Jobs
          </Link>
        </nav>
        <div className="ml-auto flex items-center space-x-4">
          <span className="text-xs text-muted-foreground">
            Stash Auto Vision
          </span>
        </div>
      </div>
    </header>
  )
}
