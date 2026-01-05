import { Outlet, Link, useLocation } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Home, History, LogOut, Upload, Leaf } from 'lucide-react'
import toast from 'react-hot-toast'
import ThemeToggle from './ThemeToggle'

export default function Layout() {
  const { user, logout } = useAuth()
  const location = useLocation()

  const handleLogout = () => {
    logout()
    toast.success('Logged out successfully')
  }

  const isActive = (path) => location.pathname === path

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-dark-bg transition-colors duration-300">
      {/* Navbar */}
      <nav className="sticky top-0 z-50 bg-white/80 dark:bg-dark-surface/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-700 transition-colors duration-300">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Link to="/" className="flex items-center gap-2 group">
                <div className="bg-primary-50 dark:bg-primary-900/30 p-2 rounded-lg group-hover:bg-primary-100 dark:group-hover:bg-primary-900/50 transition-colors">
                  <Leaf className="w-6 h-6 text-primary-600 dark:text-primary-400" />
                </div>
                <span className="text-xl font-bold text-gray-900 dark:text-white transition-colors">AgroAI</span>
              </Link>
            </div>

            <div className="flex items-center gap-4">
              <Link
                to="/"
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${isActive('/')
                    ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                  }`}
              >
                <Upload className="w-5 h-5" />
                <span className="hidden sm:inline">Upload</span>
              </Link>
              <Link
                to="/history"
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${isActive('/history')
                    ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                  }`}
              >
                <History className="w-5 h-5" />
                <span className="hidden sm:inline">History</span>
              </Link>

              <div className="h-6 w-px bg-gray-200 dark:bg-gray-700 mx-2" />

              <ThemeToggle />

              <div className="flex items-center gap-3 pl-2">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-200 hidden sm:inline">
                  {user?.username}
                </span>
                <button
                  onClick={handleLogout}
                  className="flex items-center gap-2 p-2 text-gray-500 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-all"
                  title="Logout"
                >
                  <LogOut className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 animate-in fade-in duration-500">
        <Outlet />
      </main>
    </div>
  )
}
