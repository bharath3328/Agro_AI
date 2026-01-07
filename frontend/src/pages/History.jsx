import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Calendar, Eye, AlertTriangle, CheckCircle, Search, Filter } from 'lucide-react'
import api from '../services/api'
import toast from 'react-hot-toast'

export default function History() {
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(true)
  const [showFilters, setShowFilters] = useState(false)
  const [filterStatus, setFilterStatus] = useState('all') // all, healthy, diseased, unknown
  const [sortBy, setSortBy] = useState('newest') // newest, oldest

  useEffect(() => {
    fetchHistory()
  }, [])

  const fetchHistory = async () => {
    try {
      const response = await api.get('/diagnosis/history')
      setPredictions(response.data)
    } catch (error) {
      toast.error('Failed to load prediction history')
    } finally {
      setLoading(false)
    }
  }



  const getStatusIcon = (prediction) => {
    if (prediction.is_unknown) {
      return <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
    }
    return <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
  }

  const getStatusColor = (prediction) => {
    if (prediction.is_unknown) {
      return 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
    }
    return 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
  }

  const filteredPredictions = predictions
    .filter(prediction => {
      // Status Filter
      if (filterStatus !== 'all') {
        if (filterStatus === 'unknown' && !prediction.is_unknown) return false
        if (filterStatus === 'healthy' && (prediction.is_unknown || !prediction.disease_name.toLowerCase().includes('healthy'))) return false
        if (filterStatus === 'diseased' && (prediction.is_unknown || prediction.disease_name.toLowerCase().includes('healthy'))) return false
      }



      return true
    })
    .sort((a, b) => {
      const dateA = new Date(a.created_at)
      const dateB = new Date(b.created_at)
      return sortBy === 'newest' ? dateB - dateA : dateA - dateB
    })

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="relative">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 dark:border-primary-400"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Prediction History</h1>
          <p className="text-gray-600 dark:text-gray-400">View and track your crop health analysis results</p>
        </div>

        <div className="relative">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`flex items-center gap-2 px-4 py-2 bg-white dark:bg-dark-surface border rounded-lg transition-colors ${showFilters
              ? 'border-primary-500 text-primary-600 dark:text-primary-400 ring-2 ring-primary-100 dark:ring-primary-900/30'
              : 'border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
          >
            <Filter className="w-4 h-4" />
            <span>Filter</span>
          </button>

          {showFilters && (
            <div className="absolute right-0 mt-2 w-64 bg-white dark:bg-dark-surface rounded-xl shadow-xl border border-gray-100 dark:border-gray-700 p-4 z-20 animate-in fade-in slide-in-from-top-2">
              <div className="space-y-4">
                <div>
                  <label className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2 block uppercase tracking-wider">Status</label>
                  <div className="space-y-1">
                    {['all', 'healthy', 'diseased', 'unknown'].map((status) => (
                      <button
                        key={status}
                        onClick={() => setFilterStatus(status)}
                        className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${filterStatus === status
                          ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 font-medium'
                          : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-800'
                          }`}
                      >
                        {status.charAt(0).toUpperCase() + status.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>



                <div>
                  <label className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2 block uppercase tracking-wider">Sort By</label>
                  <div className="space-y-1">
                    {[
                      { value: 'newest', label: 'Date (Newest)' },
                      { value: 'oldest', label: 'Date (Oldest)' }
                    ].map((option) => (
                      <button
                        key={option.value}
                        onClick={() => setSortBy(option.value)}
                        className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${sortBy === option.value
                          ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 font-medium'
                          : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-800'
                          }`}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {filteredPredictions.length === 0 ? (
        <div className="bg-white dark:bg-dark-surface rounded-2xl shadow-lg border border-gray-100 dark:border-gray-800 p-12 text-center transition-colors">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gray-50 dark:bg-gray-800 mb-6">
            <Search className="w-10 h-10 text-gray-400" />
          </div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">No predictions found</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-md mx-auto">
            {predictions.length === 0
              ? "Your diagnosis history will appear here once you analyze your first plant image."
              : "No predictions match your current filter settings."}
          </p>
          {predictions.length === 0 ? (
            <Link
              to="/"
              className="inline-flex items-center gap-2 bg-primary-600 text-white px-6 py-3 rounded-xl font-semibold hover:bg-primary-700 transition-colors shadow-lg hover:shadow-xl hover:-translate-y-0.5 transform transition-all"
            >
              Start Diagnosis
            </Link>
          ) : (
            <button
              onClick={() => {
                setFilterStatus('all')
              }}
              className="inline-flex items-center gap-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-6 py-3 rounded-xl font-semibold hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            >
              Clear Filters
            </button>
          )}
        </div>
      ) : (
        <div className="grid gap-4">
          {filteredPredictions.map((prediction) => (
            <Link
              key={prediction.id}
              to={`/prediction/${prediction.id}`}
              className="group block bg-white dark:bg-dark-surface rounded-xl shadow-sm hover:shadow-md border border-gray-100 dark:border-gray-800 p-6 transition-all duration-200"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-start gap-4 flex-1">
                  <div className={`p-3 rounded-xl ${getStatusColor(prediction)} transition-colors`}>
                    {getStatusIcon(prediction)}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 mb-1">
                      <h3 className="text-lg font-bold text-gray-900 dark:text-white truncate">
                        {prediction.is_unknown ? 'Unknown Disease' : prediction.disease_name}
                      </h3>
                      {!prediction.is_unknown && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300">
                          {Math.round(prediction.confidence_score * 100)}%
                        </span>
                      )}
                    </div>

                    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-gray-500 dark:text-gray-400 mb-3">
                      <div className="flex items-center gap-1.5">
                        <Calendar className="w-4 h-4" />
                        {new Date(prediction.created_at).toLocaleDateString(undefined, {
                          year: 'numeric', month: 'long', day: 'numeric'
                        })}
                      </div>


                      {prediction.disease_stage && (
                        <div className="ml-auto sm:ml-0">
                          <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${prediction.disease_stage === 'Early' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' :
                            prediction.disease_stage === 'Mid' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300' :
                              'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                            }`}>
                            {prediction.disease_stage} Stage
                          </span>
                        </div>
                      )}
                    </div>

                    {prediction.recommended_action && (
                      <p className="text-sm text-gray-600 dark:text-gray-300 line-clamp-1">
                        <span className="font-medium text-gray-900 dark:text-gray-200">Action:</span> {prediction.recommended_action}
                      </p>
                    )}
                  </div>
                </div>

                <div className="flex items-center justify-center w-10 h-10 rounded-full bg-gray-50 dark:bg-gray-800 text-gray-400 group-hover:bg-primary-50 dark:group-hover:bg-primary-900/20 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
                  <Eye className="w-5 h-5" />
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
