import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Calendar, Eye, AlertTriangle, CheckCircle, Search, Filter } from 'lucide-react'
import api from '../services/api'
import toast from 'react-hot-toast'

export default function History() {
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(true)

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

        {/* Placeholder for future filter controls */}
        <div className="flex gap-2">
          <button className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-dark-surface border border-gray-200 dark:border-gray-700 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
            <Filter className="w-4 h-4" />
            <span>Filter</span>
          </button>
        </div>
      </div>

      {predictions.length === 0 ? (
        <div className="bg-white dark:bg-dark-surface rounded-2xl shadow-lg border border-gray-100 dark:border-gray-800 p-12 text-center transition-colors">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gray-50 dark:bg-gray-800 mb-6">
            <Calendar className="w-10 h-10 text-gray-400" />
          </div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">No predictions yet</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-md mx-auto">
            Your diagnosis history will appear here once you analyze your first plant image.
          </p>
          <Link
            to="/"
            className="inline-flex items-center gap-2 bg-primary-600 text-white px-6 py-3 rounded-xl font-semibold hover:bg-primary-700 transition-colors shadow-lg hover:shadow-xl hover:-translate-y-0.5 transform transition-all"
          >
            Start Diagnosis
          </Link>
        </div>
      ) : (
        <div className="grid gap-4">
          {predictions.map((prediction) => (
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
                      {prediction.crop_type && (
                        <>
                          <span className="w-1 h-1 rounded-full bg-gray-300 dark:bg-gray-600" />
                          <span>{prediction.crop_type}</span>
                        </>
                      )}

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
