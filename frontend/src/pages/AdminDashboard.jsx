import { useState, useEffect } from 'react'
import { Upload, Plus, RefreshCw, CheckCircle, AlertCircle, Shield, Users, Activity, Search } from 'lucide-react'
import api from '../services/api'
import toast from 'react-hot-toast'
import { useAuth } from '../contexts/AuthContext'
import { useNavigate } from 'react-router-dom'



export default function AdminDashboard() {
    const { user } = useAuth()
    const navigate = useNavigate()

    const [activeTab, setActiveTab] = useState('overview')
    const [diseaseName, setDiseaseName] = useState('')
    const [files, setFiles] = useState([])
    const [training, setTraining] = useState(false)
    const [stats, setStats] = useState(null)
    const [diseaseStats, setDiseaseStats] = useState([])
    const [usersList, setUsersList] = useState([])
    const [reports, setReports] = useState([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        // Redirect if not admin
        if (user && !user.is_admin) {
            toast.error("Access denied. Admin only.")
            navigate('/dashboard')
            return
        }
        fetchDashboardData()
    }, [user, navigate])

    const fetchDashboardData = async () => {
        try {
            const [statsRes, diseasesRes, usersRes, reportsRes] = await Promise.all([
                api.get('/admin/stats'),
                api.get('/admin/diseases'),
                api.get('/admin/users'),
                api.get('/admin/reports')
            ])
            setStats(statsRes.data)
            setUsersList(usersRes.data)
            setReports(reportsRes.data)

            // Format for chart: { name: 'Tomato Blight', count: 15 }
            const chartData = diseasesRes.data.map(d => ({
                name: d.disease_name,
                count: d.total_detections
            }))
            setDiseaseStats(chartData)

        } catch (error) {
            console.error("Failed to fetch dashboard data", error)
            toast.error("Failed to load dashboard data")
        } finally {
            setLoading(false)
        }
    }

    const handleFileChange = (e) => {
        if (e.target.files) {
            setFiles(Array.from(e.target.files))
        }
    }

    const handleTrain = async (e) => {
        e.preventDefault()
        if (!diseaseName || files.length === 0) {
            toast.error("Please provide a disease name and at least one image")
            return
        }

        if (files.length < 5) {
            toast("For best results, upload at least 5 images", { icon: 'ℹ️' })
        }

        setTraining(true)
        const formData = new FormData()
        formData.append('disease_name', diseaseName)
        files.forEach(file => {
            formData.append('files', file)
        })

        try {
            const response = await api.post('/admin/train', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            })
            toast.success(response.data.message)
            setDiseaseName('')
            setFiles([])
            fetchDashboardData() // Refresh stats
        } catch (error) {
            console.error(error)
            const msg = error.response?.data?.detail || "Training failed"
            toast.error(msg)
        } finally {
            setTraining(false)
        }
    }

    const handleApproveReport = async (reportId, correctLabel) => {
        try {
            await api.post(`/admin/reports/${reportId}/approve`, {
                correct_label: correctLabel
            })
            toast.success("Case approved and model updated!")
            fetchDashboardData()
        } catch (error) {
            toast.error("Failed to approve report")
        }
    }

    const handleRejectReport = async (reportId) => {
        if (!window.confirm("Reject this report?")) return;
        try {
            await api.post(`/admin/reports/${reportId}/reject`)
            toast.success("Report rejected")
            fetchDashboardData()
        } catch (error) {
            toast.error("Failed to reject report")
        }
    }

    if (loading) {
        return <div className="flex justify-center p-12"><RefreshCw className="animate-spin text-primary-500" /></div>
    }

    return (
        <div className="max-w-6xl mx-auto space-y-8">
            <div className="flex justify-between items-center border-b border-gray-100 dark:border-gray-800 pb-6">
                <div className="flex items-center gap-4">
                    <div className="p-3 bg-indigo-100 dark:bg-indigo-900/30 rounded-2xl text-indigo-600 dark:text-indigo-400">
                        <Shield className="w-8 h-8" />
                    </div>
                    <div>
                        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Admin Console</h1>
                        <p className="text-gray-600 dark:text-gray-400">System metrics and continual learning interface</p>
                    </div>
                </div>

                {/* Tab Navigation */}
                <div className="flex bg-gray-100 dark:bg-gray-800 p-1 rounded-lg">
                    <button
                        onClick={() => setActiveTab('overview')}
                        className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${activeTab === 'overview'
                            ? 'bg-white dark:bg-dark-surface text-primary-600 shadow-sm'
                            : 'text-gray-500 hover:text-gray-900 dark:hover:text-gray-200'
                            }`}
                    >
                        Overview
                    </button>
                    <button
                        onClick={() => setActiveTab('users')}
                        className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${activeTab === 'users'
                            ? 'bg-white dark:bg-dark-surface text-primary-600 shadow-sm'
                            : 'text-gray-500 hover:text-gray-900 dark:hover:text-gray-200'
                            }`}
                    >
                        User Management
                    </button>
                    <button
                        onClick={() => setActiveTab('reports')}
                        className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${activeTab === 'reports'
                            ? 'bg-white dark:bg-dark-surface text-primary-600 shadow-sm'
                            : 'text-gray-500 hover:text-gray-900 dark:hover:text-gray-200'
                            }`}
                    >
                        Reported Cases
                        {reports.length > 0 && (
                            <span className="ml-2 bg-red-100 text-red-600 px-2 py-0.5 rounded-full text-xs">{reports.length}</span>
                        )}
                    </button>
                </div>
            </div>

            {activeTab === 'overview' && (
                <div className="max-w-3xl mx-auto">
                    {/* Training Card */}
                    <div className="bg-white dark:bg-dark-surface rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800 p-8">
                        <div className="mb-8">
                            <h2 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                                <Plus className="w-6 h-6 text-primary-500" />
                                Train New Disease
                            </h2>
                            <p className="text-gray-500 dark:text-gray-400 mt-1">
                                Add a new disease class to the model dynamically. No downtime required.
                            </p>
                        </div>

                        <form onSubmit={handleTrain} className="space-y-6">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Disease Name
                                </label>
                                <input
                                    type="text"
                                    value={diseaseName}
                                    onChange={(e) => setDiseaseName(e.target.value)}
                                    placeholder="e.g. Tomato Early Blight"
                                    className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 focus:ring-2 focus:ring-primary-100 outline-none transition-all"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Training Images
                                </label>
                                <div className="border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-2xl p-8 text-center bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                                    <input
                                        type="file"
                                        multiple
                                        onChange={handleFileChange}
                                        accept="image/*"
                                        className="hidden"
                                        id="images-upload"
                                    />
                                    <label htmlFor="images-upload" className="cursor-pointer block">
                                        <Upload className="w-10 h-10 text-gray-400 mx-auto mb-3" />
                                        <p className="text-gray-900 dark:text-white font-medium mb-1">
                                            {files.length > 0 ? `${files.length} images selected` : 'Click to upload images'}
                                        </p>
                                        <p className="text-xs text-gray-500">
                                            Recommended: 5-20 distinct images
                                        </p>
                                    </label>
                                </div>
                            </div>

                            <button
                                type="submit"
                                disabled={training}
                                className={`w-full py-4 rounded-xl font-bold text-white shadow-lg transition-all ${training
                                    ? 'bg-gray-400 cursor-not-allowed'
                                    : 'bg-primary-600 hover:bg-primary-700 hover:-translate-y-1 hover:shadow-xl'
                                    }`}
                            >
                                {training ? (
                                    <span className="flex items-center justify-center gap-2">
                                        <RefreshCw className="w-5 h-5 animate-spin" />
                                        Training Model...
                                    </span>
                                ) : (
                                    'Start Training'
                                )}
                            </button>
                        </form>
                    </div>
                </div>
            )}

            {activeTab === 'users' && usersList.length > 0 && (
                <div className="bg-white dark:bg-dark-surface rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 overflow-hidden">
                    <div className="overflow-x-auto">
                        <table className="w-full text-left text-sm">
                            <thead className="bg-gray-50 dark:bg-gray-800 border-b border-gray-100 dark:border-gray-700">
                                <tr>
                                    <th className="px-6 py-4 font-semibold text-gray-900 dark:text-white">User</th>
                                    <th className="px-6 py-4 font-semibold text-gray-900 dark:text-white">Status</th>
                                    <th className="px-6 py-4 font-semibold text-gray-900 dark:text-white">Role</th>
                                    <th className="px-6 py-4 font-semibold text-gray-900 dark:text-white">Predictions</th>
                                    <th className="px-6 py-4 font-semibold text-gray-900 dark:text-white">Joined</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
                                {usersList.map((user) => (
                                    <tr key={user.id} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                                        <td className="px-6 py-4">
                                            <div>
                                                <div className="font-medium text-gray-900 dark:text-white">{user.full_name || 'N/A'}</div>
                                                <div className="text-gray-500 dark:text-gray-400">{user.email}</div>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${user.is_active
                                                ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                                                : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                                                }`}>
                                                {user.is_active ? 'Active' : 'Inactive'}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-gray-500 dark:text-gray-400">
                                            {user.is_admin ? 'Admin' : 'User'}
                                        </td>
                                        <td className="px-6 py-4 text-gray-500 dark:text-gray-400">
                                            {user.prediction_count}
                                        </td>
                                        <td className="px-6 py-4 text-gray-500 dark:text-gray-400">
                                            {new Date(user.created_at).toLocaleDateString()}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {activeTab === 'reports' && (
                <ReportedCasesList reports={reports} onApprove={handleApproveReport} onReject={handleRejectReport} />
            )}
        </div>
    )
}

const StatsCard = ({ title, value, icon: Icon, color }) => {
    const colors = {
        blue: 'bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400',
        green: 'bg-green-50 text-green-600 dark:bg-green-900/20 dark:text-green-400',
        purple: 'bg-purple-50 text-purple-600 dark:bg-purple-900/20 dark:text-purple-400',
        orange: 'bg-orange-50 text-orange-600 dark:bg-orange-900/20 dark:text-orange-400',
    }

    return (
        <div className="bg-white dark:bg-dark-surface p-6 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 flex items-center gap-4">
            <div className={`p-3 rounded-lg ${colors[color]}`}>
                <Icon className="w-6 h-6" />
            </div>
            <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
            </div>
        </div>
    )
}

const ReportedCasesList = ({ reports, onApprove, onReject }) => {
    if (reports.length === 0) {
        return (
            <div className="text-center py-12 bg-white dark:bg-dark-surface rounded-xl border border-gray-100 dark:border-gray-700">
                <p className="text-gray-500">No pending reports.</p>
            </div>
        )
    }

    return (
        <div className="grid gap-6">
            {reports.map(report => (
                <div key={report.id} className="bg-white dark:bg-dark-surface p-6 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 flex gap-6">
                    <div className="w-48 h-48 flex-shrink-0">
                        <img
                            src={report.image_url}
                            alt="Reported Case"
                            className="w-full h-full object-cover rounded-lg border border-gray-200 dark:border-gray-600"
                        />
                    </div>

                    <div className="flex-1 space-y-4">
                        <div className="flex justify-between items-start">
                            <div>
                                <h3 className="font-bold text-lg text-gray-900 dark:text-white">
                                    Proposed: {report.proposed_label || "No Label Proposed"}
                                </h3>
                                <p className="text-sm text-gray-500">Reported on {new Date(report.created_at).toLocaleDateString()}</p>
                            </div>
                            <span className="px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-xs font-bold">
                                {report.status}
                            </span>
                        </div>

                        <div className="bg-gray-50 dark:bg-gray-800/50 p-4 rounded-lg">
                            <p className="text-sm font-bold text-gray-700 dark:text-gray-300 mb-1">User Description:</p>
                            <p className="text-sm text-gray-600 dark:text-gray-400 italic">
                                "{report.description || "No description provided."}"
                            </p>
                        </div>

                        <div className="flex gap-4 pt-2">
                            <ApproveForm report={report} onApprove={onApprove} />

                            <button
                                onClick={() => onReject(report.id)}
                                className="px-4 py-2 border border-red-200 text-red-600 rounded-lg hover:bg-red-50 text-sm font-medium"
                            >
                                Reject
                            </button>
                        </div>
                    </div>
                </div>
            ))}
        </div>
    )
}

const ApproveForm = ({ report, onApprove }) => {
    const [label, setLabel] = useState(report.proposed_label || '');

    return (
        <div className="flex items-center gap-2 flex-1 max-w-md">
            <input
                type="text"
                placeholder="Confirm Class Name..."
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white dark:bg-dark-bg dark:border-gray-600"
            />
            <button
                onClick={() => onApprove(report.id, label)}
                disabled={!label}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 text-sm font-medium disabled:opacity-50 whitespace-nowrap"
            >
                Approve & Train
            </button>
        </div>
    )
}
