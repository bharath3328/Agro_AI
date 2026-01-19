import React, { useState } from 'react';
import api from '../services/api';
import { toast } from 'react-hot-toast';
import { X, AlertTriangle, Send } from 'lucide-react';

const ReportModal = ({ predictionId, onClose }) => {
    const [loading, setLoading] = useState(false);
    const [formData, setFormData] = useState({
        proposed_label: '',
        description: ''
    });

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);

        try {
            await api.post('/diagnosis/report', {
                prediction_id: predictionId,
                ...formData
            });

            toast.success('Report submitted successfully!');
            onClose();
        } catch (error) {
            console.error(error);
            toast.error(error.response?.data?.detail || 'Failed to submit report');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-xl shadow-2xl max-w-md w-full overflow-hidden animate-fade-in">
                <div className="bg-red-50 p-4 border-b border-red-100 flex justify-between items-center">
                    <div className="flex items-center gap-2 text-red-700">
                        <AlertTriangle className="w-5 h-5" />
                        <h3 className="font-semibold">Report Incorrect Diagnosis</h3>
                    </div>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-700">
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="p-6 space-y-4">
                    <p className="text-sm text-gray-600">
                        Help us improve AgroAI! If the model got it wrong or the disease is unknown, please let us know.
                    </p>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            What do you think this is? (Optional)
                        </label>
                        <input
                            type="text"
                            className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                            placeholder="e.g. Tomato Early Blight"
                            value={formData.proposed_label}
                            onChange={(e) => setFormData({ ...formData, proposed_label: e.target.value })}
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            Description / Notes
                        </label>
                        <textarea
                            className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent h-24"
                            placeholder="Describe what you see..."
                            value={formData.description}
                            onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                        />
                    </div>

                    <div className="flex gap-3 pt-2">
                        <button
                            type="button"
                            onClick={onClose}
                            className="flex-1 py-2 px-4 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 font-medium"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={loading}
                            className="flex-1 py-2 px-4 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium flex items-center justify-center gap-2 disabled:opacity-50"
                        >
                            {loading ? 'Sending...' : (
                                <>
                                    <Send className="w-4 h-4" />
                                    Submit Report
                                </>
                            )}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default ReportModal;
