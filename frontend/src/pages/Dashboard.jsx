import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, Image as ImageIcon, Loader, AlertCircle, Sparkles, Sprout } from 'lucide-react'
import api from '../services/api'
import toast from 'react-hot-toast'
import { useNavigate } from 'react-router-dom'

export default function Dashboard() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [cropType, setCropType] = useState('')
  const [location, setLocation] = useState('')
  const navigate = useNavigate()

  const onDrop = useCallback((acceptedFiles) => {
    const selectedFile = acceptedFiles[0]
    if (selectedFile) {
      setFile(selectedFile)
      const reader = new FileReader()
      reader.onload = () => setPreview(reader.result)
      reader.readAsDataURL(selectedFile)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png']
    },
    maxFiles: 1
  })

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) {
      toast.error('Please select an image')
      return
    }

    setLoading(true)
    const formData = new FormData()
    formData.append('file', file)
    if (cropType) formData.append('crop_type', cropType)
    if (location) formData.append('location', location)

    try {
      const response = await api.post('/diagnosis/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      toast.success('Disease detected successfully!')
      navigate(`/prediction/${response.data.prediction_id}`)
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setPreview(null)
    setCropType('')
    setLocation('')
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-10 space-y-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full text-sm font-medium">
          <Sparkles className="w-4 h-4" />
          <span>AI-Powered Diagnostics</span>
        </div>
        <h1 className="text-4xl font-extrabold text-gray-900 dark:text-white sm:text-5xl">
          Protect Your Crops
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
          Upload a photo of your plant to instantly detect diseases and get expect advice.
        </p>
      </div>

      <div className="bg-white dark:bg-dark-surface rounded-2xl shadow-xl dark:shadow-gray-900/10 p-8 border border-gray-100 dark:border-gray-800 transition-all duration-300">
        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Image Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Plant Image
            </label>
            {!preview ? (
              <div
                {...getRootProps()}
                className={`group relative border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-300 ${isDragActive
                    ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                    : 'border-gray-300 dark:border-gray-700 hover:border-primary-400 dark:hover:border-primary-500 hover:bg-gray-50 dark:hover:bg-gray-800'
                  }`}
              >
                <input {...getInputProps()} />
                <div className="absolute inset-0 bg-gradient-to-br from-primary-500/5 to-transparent rounded-xl opacity-0 group-hover:opacity-100 transition-opacity" />

                <div className="relative z-10">
                  <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 mb-4 group-hover:scale-110 transition-transform">
                    <Upload className="w-8 h-8" />
                  </div>
                  {isDragActive ? (
                    <p className="text-primary-600 dark:text-primary-400 font-medium text-lg">Drop the image here...</p>
                  ) : (
                    <>
                      <p className="text-gray-600 dark:text-gray-300 font-medium text-lg mb-2">
                        Drag & drop an image here, or click to select
                      </p>
                      <p className="text-sm text-gray-400">JPG, PNG up to 10MB</p>
                    </>
                  )}
                </div>
              </div>
            ) : (
              <div className="relative group">
                <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity rounded-xl flex items-center justify-center z-10">
                  <button
                    type="button"
                    onClick={handleReset}
                    className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition-colors transform hover:scale-105"
                  >
                    Remove Image
                  </button>
                </div>
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full h-80 object-cover rounded-xl shadow-md"
                />
              </div>
            )}
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Crop Type */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Crop Type (Optional)
              </label>
              <div className="relative">
                <Sprout className="absolute left-3 top-3.5 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  value={cropType}
                  onChange={(e) => setCropType(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent dark:text-white transition-colors"
                  placeholder="e.g., Tomato, Potato"
                />
              </div>
            </div>

            {/* Location */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Location (Optional)
              </label>
              <input
                type="text"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent dark:text-white transition-colors"
                placeholder="e.g., Greenhouse 1"
              />
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading || !file}
            className="w-full bg-gradient-to-r from-primary-600 to-primary-700 hover:from-primary-700 hover:to-primary-800 text-white py-4 rounded-xl font-bold text-lg shadow-lg hover:shadow-xl transition-all transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <Loader className="w-6 h-6 animate-spin" />
                Analyzing Plant Health...
              </>
            ) : (
              <>
                <ImageIcon className="w-6 h-6" />
                Analyze Image
              </>
            )}
          </button>
        </form>

        {/* Info Box */}
        <div className="mt-8 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded-xl p-4 flex items-start gap-4">
          <AlertCircle className="w-6 h-6 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-blue-800 dark:text-blue-200">
            <p className="font-bold mb-1 text-base">Tips for accurate results:</p>
            <ul className="list-disc list-inside space-y-1 opacity-90">
              <li>Ensure the affected area is clearly visible and in focus</li>
              <li>Avoid shadows or glare on the leaf surface</li>
              <li>Capture the image against a neutral background if possible</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
