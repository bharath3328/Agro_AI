import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { KeyRound, Mail } from 'lucide-react'
import toast from 'react-hot-toast'
import api from '../services/api'

export default function ForgotPassword() {
    const [email, setEmail] = useState('')
    const [loading, setLoading] = useState(false)
    const navigate = useNavigate()

    const handleSubmit = async (e) => {
        e.preventDefault()
        setLoading(true)

        try {
            const response = await api.post('/auth/forgot-password', { email })
            // Always show success message for security/ux
            toast.success('If an account exists, a code has been sent.')
            // Navigate to reset password page
            navigate('/reset-password', {
                state: {
                    contact: email,
                    method: 'email',
                    verification_token: response.data.verification_token
                }
            })
        } catch (error) {
            toast.error('An error occurred. Please try again.')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 flex items-center justify-center p-4">
            <div className="max-w-md w-full bg-white rounded-2xl shadow-xl p-8">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mb-4">
                        <KeyRound className="w-8 h-8 text-primary-600" />
                    </div>
                    <h1 className="text-3xl font-bold text-gray-900">Forgot Password</h1>
                    <p className="text-gray-600 mt-2">Enter your email to reset password</p>
                </div>

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Email Address
                        </label>
                        <div className="relative">
                            <Mail className="absolute left-3 top-3.5 w-5 h-5 text-gray-400" />
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                                className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                                placeholder="Enter your email"
                            />
                        </div>
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-primary-600 text-white py-3 rounded-lg font-semibold hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                        {loading ? (
                            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                        ) : (
                            'Send Reset Code'
                        )}
                    </button>
                </form>

                <div className="mt-6 text-center">
                    <Link to="/login" className="text-sm text-gray-600 hover:text-gray-900">
                        Back to Login
                    </Link>
                </div>
            </div>
        </div>
    )
}
