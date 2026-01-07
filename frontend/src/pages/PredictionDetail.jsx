import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, AlertTriangle, CheckCircle, Image as ImageIcon, Calendar, MapPin, FileText, Layers, Flag } from 'lucide-react'
import api from '../services/api'
import toast from 'react-hot-toast'
import AIAdvisoryCard from '../components/AIAdvisoryCard'
import ReportModal from '../components/ReportModal'

export default function PredictionDetail() {
  const { id } = useParams()
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(true)
  const [gradcamUrl, setGradcamUrl] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [isReportModalOpen, setIsReportModalOpen] = useState(false)

  useEffect(() => {
    fetchPrediction()
  }, [id])

  const fetchPrediction = async () => {
    try {
      const response = await api.get(`/diagnosis/history/${id}`)
      setPrediction(response.data)

      // Fetch uploaded image
      try {
        const imageResponse = await api.get(`/diagnosis/image/${id}`, {
          responseType: 'blob'
        })
        setImageUrl(URL.createObjectURL(imageResponse.data))
      } catch (error) {
        console.error('Failed to load uploaded image:', error)
      }

      if (response.data.gradcam_path) {
        try {
          const gradcamResponse = await api.get(`/diagnosis/gradcam/${id}`, {
            responseType: 'blob'
          })
          setGradcamUrl(URL.createObjectURL(gradcamResponse.data))
        } catch (error) {
          console.error('Failed to load Grad-CAM:', error)
        }
      }
    } catch (error) {
      toast.error('Failed to load prediction details')
    } finally {
      setLoading(false)
    }
  }

  const getStageColor = (stage) => {
    switch (stage) {
      case 'Early': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
      case 'Mid': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
      case 'Late': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300'
    }
  }

  const getConfidenceLevel = (score) => {
    if (score > 0.9) return { label: 'High Confidence', color: 'text-green-600 dark:text-green-400' }
    if (score > 0.7) return { label: 'Moderate Confidence', color: 'text-yellow-600 dark:text-yellow-400' }
    return { label: 'Low Confidence', color: 'text-red-600 dark:text-red-400' }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="relative">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-primary-600 dark:border-primary-400"></div>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="h-8 w-8 bg-primary-100 dark:bg-primary-900 rounded-full animate-pulse"></div>
          </div>
        </div>
      </div>
    )
  }

  if (!prediction) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600 dark:text-gray-400 mb-4">Prediction not found</p>
        <Link
          to="/history"
          className="px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          Back to History
        </Link>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Navigation */}
      <Link
        to="/history"
        className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
      >
        <ArrowLeft className="w-5 h-5" />
        <span className="font-medium">Back to History</span>
      </Link>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Left Column - Images & Core Info */}
        <div className="lg:col-span-2 space-y-6">
          {/* Main Result Card */}
          <div className="bg-white dark:bg-dark-surface rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 overflow-hidden">
            <div className="border-b border-gray-100 dark:border-gray-700 p-6 flex justify-between items-start">
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                  Prediction Results
                </h1>
                <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                  <div className="flex items-center gap-1.5">
                    <Calendar className="w-4 h-4" />
                    <span>{new Date(prediction.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
              </div>

              {!prediction.is_unknown && (
                <div className="text-right">
                  <div className="text-3xl font-bold text-primary-600 dark:text-primary-400">
                    {Math.round(prediction.confidence_score * 100)}%
                  </div>
                  <div className={`text-sm font-medium ${getConfidenceLevel(prediction.confidence_score).color}`}>
                    {getConfidenceLevel(prediction.confidence_score).label}
                  </div>
                </div>
              )}
            </div>

            <div className="p-6">
              <div className="grid md:grid-cols-2 gap-6">
                {/* Original Image */}
                <div className="space-y-2">
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Analyzed Image</p>
                  <div className="aspect-square rounded-xl overflow-hidden bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 shadow-inner">
                    {imageUrl ? (
                      <img
                        src={imageUrl}
                        alt="Uploaded Plant"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-gray-400">
                        <ImageIcon className="w-12 h-12" />
                      </div>
                    )}
                  </div>
                </div>

                {/* Grad-CAM */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Heatmap Analysis</p>
                    {prediction.cam_coverage > 0 && (
                      <span className="text-xs bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 px-2 py-0.5 rounded-full">
                        {Math.round(prediction.cam_coverage * 100)}% Coverage
                      </span>
                    )}
                  </div>

                  {gradcamUrl ? (
                    <div className="aspect-square rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700 shadow-inner">
                      <img
                        src={gradcamUrl}
                        alt="AI Attention Map"
                        className="w-full h-full object-cover"
                      />
                    </div>
                  ) : (
                    <div className="aspect-square rounded-xl bg-gray-50 dark:bg-gray-800/50 flex items-center justify-center border border-gray-200 dark:border-gray-700 border-dashed">
                      <p className="text-sm text-gray-500 dark:text-gray-400">Heatmap unavailable</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* AI Advisory Section (Prominent) */}
          {prediction.ai_advisory && !prediction.is_unknown && !prediction.disease_name.toLowerCase().includes('healthy') && (
            <AIAdvisoryCard advisory={prediction.ai_advisory} />
          )}
        </div>

        {/* Right Column - Stats & Details */}
        <div className="space-y-6">
          {/* Disease Status Card */}
          <div className={`p-6 rounded-2xl border ${prediction.is_unknown
            ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
            : 'bg-white dark:bg-dark-surface border-gray-100 dark:border-gray-700 shadow-sm'
            }`}>
            <h2 className="text-lg font-bold text-gray-900 dark:text-white mb-4">Diagnosis Status</h2>

            {prediction.is_unknown ? (
              <div className="text-center py-4">
                <AlertTriangle className="w-12 h-12 text-yellow-500 mx-auto mb-3" />
                <h3 className="text-xl font-bold text-yellow-700 dark:text-yellow-400 mb-2">Unknown Disease</h3>
                <p className="text-yellow-600 dark:text-yellow-300/80 text-sm mb-4">
                  Our confidence score was too low to verify the disease. Please consult an expert.
                </p>
                <button
                  onClick={() => setIsReportModalOpen(true)}
                  className="w-full py-2 px-4 bg-yellow-100 hover:bg-yellow-200 text-yellow-800 rounded-lg font-medium flex items-center justify-center gap-2 border border-yellow-200 transition-colors"
                >
                  <Flag className="w-4 h-4" />
                  Report to Research Team
                </button>
              </div>
            ) : (
              <div className="space-y-6">
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Detected Pathogen</p>
                  <p className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                    {prediction.disease_name}
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Severity</p>
                    <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-semibold ${getStageColor(prediction.disease_stage)}`}>
                      {prediction.disease_stage}
                    </span>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Impact</p>
                    <p className="text-sm font-semibold text-gray-900 dark:text-white">{prediction.estimated_yield_loss}</p>
                  </div>
                </div>

                {/* Only show recommendation if NOT healthy */}
                {!prediction.disease_name.toLowerCase().includes('healthy') && (
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Recommended Action</p>
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded-lg">
                      <p className="text-blue-800 dark:text-blue-300 font-medium text-sm">
                        {prediction.recommended_action}
                      </p>
                    </div>
                  </div>
                )}

                {prediction.disease_name.toLowerCase().includes('healthy') && (
                  <div className="p-3 bg-green-50 dark:bg-green-900/20 border border-green-100 dark:border-green-800 rounded-lg">
                    <p className="text-green-800 dark:text-green-300 font-medium text-sm text-center">
                      🎉 Plant is healthy! No action needed.
                    </p>
                  </div>
                )}

                {/* Option to report incorrect diagnosis even if known */}
                <div className="pt-4 border-t border-gray-100 dark:border-gray-700">
                  <button
                    onClick={() => setIsReportModalOpen(true)}
                    className="text-xs text-gray-500 hover:text-red-600 flex items-center gap-1 mx-auto transition-colors"
                  >
                    <Flag className="w-3 h-3" />
                    Report Incorrect Diagnosis
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Metadata Card - Only Notes */}
          {prediction.notes && (
            <div className="bg-white dark:bg-dark-surface rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
              <h2 className="text-lg font-bold text-gray-900 dark:text-white mb-4">Field Notes</h2>
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <FileText className="w-5 h-5 text-gray-400 mt-0.5" />
                  <div>
                    <p className="text-sm text-gray-700 dark:text-gray-300 italic">{prediction.notes}</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {isReportModalOpen && (
        <ReportModal
          predictionId={id}
          onClose={() => setIsReportModalOpen(false)}
        />
      )}
    </div>
  )
}
