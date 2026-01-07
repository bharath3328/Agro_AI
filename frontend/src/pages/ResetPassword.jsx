import { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { Lock } from 'lucide-react'
import toast from 'react-hot-toast'
import api from '../services/api'

export default function ResetPassword() {
    const [code, setCode] = useState('')
    const [newPassword, setNewPassword] = useState('')
    const [confirmPassword, setConfirmPassword] = useState('')
    const [loading, setLoading] = useState(false)
    const location = useLocation()
    const navigate = useNavigate()

    const { contact, method, verification_token } = location.state || {}

    const handleSubmit = async (e) => {
        e.preventDefault()

        if (newPassword !== confirmPassword) {
            toast.error('Passwords do not match')
            return
        }

        setLoading(true)

        try {
            const payload = {
                code,
                new_password: newPassword,
                verification_token,
                ...(method === 'email' ? { email: contact } : { phone: contact })
            }

            await api.post('/auth/reset-password', payload)
            toast.success('Password reset successfully! Please login.')
            navigate('/login')
        } catch (error) {
            toast.error(error.response?.data?.detail || 'Reset failed')
        } finally {
            setLoading(false)
        }
    }

    if (!contact) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <p>Invalid access. Please start from <a href="/forgot-password" className="text-primary-600">Forgot Password</a> page.</p>
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 flex items-center justify-center p-4">
            <div className="max-w-md w-full bg-white rounded-2xl shadow-xl p-8">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mb-4">
                        <Lock className="w-8 h-8 text-primary-600" />
                    </div>
                    <h1 className="text-3xl font-bold text-gray-900">Reset Password</h1>
                    <p className="text-gray-600 mt-2">Enter the code sent to {contact}</p>
                </div>

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Verification Code
                        </label>
                        <input
                            type="text"
                            value={code}
                            onChange={(e) => setCode(e.target.value)}
                            required
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent text-center text-xl tracking-widest"
                            placeholder="Code"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            New Password
                        </label>
                        <input
                            type="password"
                            value={newPassword}
                            onChange={(e) => setNewPassword(e.target.value)}
                            required
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                            placeholder="New password"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Confirm New Password
                        </label>
                        <input
                            type="password"
                            value={confirmPassword}
                            onChange={(e) => setConfirmPassword(e.target.value)}
                            required
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                            placeholder="Confirm new password"
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-primary-600 text-white py-3 rounded-lg font-semibold hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                        {loading ? (
                            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                        ) : (
                            'Reset Password'
                        )}
                    </button>
                </form>
            </div>
        </div>
    )
}
