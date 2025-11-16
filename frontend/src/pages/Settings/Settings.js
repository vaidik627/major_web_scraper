import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import {
  UserCircleIcon,
  KeyIcon,
  BellIcon,
  CogIcon,
  TrashIcon,
} from '@heroicons/react/24/outline';
import { useAuthStore } from '../../store/authStore';
import { useThemeStore } from '../../store/themeStore';
import api from '../../services/api';
import ConfirmationModal from '../../components/UI/ConfirmationModal';

const Settings = () => {
  const { user, logout } = useAuthStore();
  const { isDark, toggleTheme } = useThemeStore();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('profile');
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const handleDeleteAccount = async () => {
    setIsDeleting(true);
    
    try {
      const loadingToast = toast.loading('Deleting your account and all data...');
      
      await api.delete('/auth/delete-account');
      
      toast.dismiss(loadingToast);
      toast.success('Account deleted successfully. You will be redirected to the login page.');
      
      // Close modal and clear auth state
      setShowDeleteModal(false);
      logout();
      navigate('/login');
    } catch (error) {
      console.error('Error deleting account:', error);
      
      if (error.response?.status === 401) {
        toast.error('Authentication failed. Please log in again and try deleting your account.');
      } else if (error.response?.status === 404) {
        toast.error('Account not found. Please contact support if this issue persists.');
      } else if (error.response?.status >= 500) {
        toast.error('Server error occurred while deleting account. Please try again later or contact support.');
      } else if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
        toast.error('Request timed out. Please check your internet connection and try again.');
      } else if (!error.response) {
        toast.error('Network error. Please check your internet connection and ensure the backend server is running.');
      } else {
        const errorMessage = error.response?.data?.detail || error.message || 'Unknown error occurred';
        toast.error(`Failed to delete account: ${errorMessage}. Please try again or contact support.`);
      }
    } finally {
      setIsDeleting(false);
    }
  };

  const tabs = [
    { id: 'profile', name: 'Profile', icon: UserCircleIcon },
    { id: 'security', name: 'Security', icon: KeyIcon },
    { id: 'notifications', name: 'Notifications', icon: BellIcon },
    { id: 'preferences', name: 'Preferences', icon: CogIcon },
  ];



  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Settings
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Manage your account settings and preferences
        </p>
      </div>

      <div className="flex flex-col lg:flex-row lg:space-x-8">
        {/* Sidebar */}
        <div className="lg:w-64">
          <nav className="space-y-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                  activeTab === tab.id
                    ? 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-200'
                    : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-700 dark:hover:text-white'
                }`}
              >
                <tab.icon className="mr-3 h-5 w-5" />
                {tab.name}
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        <div className="flex-1 mt-8 lg:mt-0">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
            className="card p-6"
          >
            {activeTab === 'profile' && (
              <div className="space-y-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Profile Information
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                      Username
                    </label>
                    <input
                      type="text"
                      value={user?.username || ''}
                      disabled
                      className="mt-1 input bg-gray-50 dark:bg-gray-700"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                      Email
                    </label>
                    <input
                      type="email"
                      value={user?.email || ''}
                      disabled
                      className="mt-1 input bg-gray-50 dark:bg-gray-700"
                    />
                  </div>
                </div>
                <div className="flex justify-end">
                  <button className="btn-primary" disabled>
                    Update Profile
                  </button>
                </div>
              </div>
            )}

            {activeTab === 'security' && (
              <div className="space-y-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Security Settings
                </h2>
                <div className="space-y-4">
                  <div>
                    <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                      Change Password
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Update your password to keep your account secure.
                    </p>
                    <button className="mt-2 btn-outline" disabled>
                      Change Password
                    </button>
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                      API Keys
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Manage your API keys for programmatic access.
                    </p>
                    <button className="mt-2 btn-outline" disabled>
                      Manage API Keys
                    </button>
                  </div>
                  
                  {/* Delete Account Section */}
                  <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
                    <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                      Delete Account
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                      Permanently delete your account and all associated data.
                    </p>
                    <button
                      onClick={() => setShowDeleteModal(true)}
                      className="mt-3 inline-flex items-center px-3 py-2 border border-red-300 dark:border-red-600 text-sm leading-4 font-medium rounded-md text-red-700 dark:text-red-200 bg-white dark:bg-gray-800 hover:bg-red-50 dark:hover:bg-red-900/20 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 transition-colors duration-200"
                    >
                      <TrashIcon className="h-4 w-4 mr-2" />
                      Delete Account
                    </button>
                  </div>

                </div>
              </div>
            )}

            {activeTab === 'notifications' && (
              <div className="space-y-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Notification Preferences
                </h2>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                        Job Completion
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Get notified when scraping jobs complete.
                      </p>
                    </div>
                    <input
                      type="checkbox"
                      defaultChecked
                      className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                        Error Alerts
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Get notified when jobs fail or encounter errors.
                      </p>
                    </div>
                    <input
                      type="checkbox"
                      defaultChecked
                      className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                    />
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'preferences' && (
              <div className="space-y-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Application Preferences
                </h2>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                        Dark Mode
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Toggle between light and dark themes.
                      </p>
                    </div>
                    <button
                      onClick={toggleTheme}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ${
                        isDark ? 'bg-primary-600' : 'bg-gray-200'
                      }`}
                    >
                      <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                          isDark ? 'translate-x-6' : 'translate-x-1'
                        }`}
                      />
                    </button>
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                        Auto-save Jobs
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Automatically save job configurations.
                      </p>
                    </div>
                    <input
                      type="checkbox"
                      defaultChecked
                      className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                        AI Analysis by Default
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Enable AI analysis for new scraping jobs by default.
                      </p>
                    </div>
                    <input
                      type="checkbox"
                      defaultChecked
                      className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                    />
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </div>
       </div>

       {/* Delete Account Confirmation Modal */}
       <ConfirmationModal
         isOpen={showDeleteModal}
         onClose={() => setShowDeleteModal(false)}
         onConfirm={handleDeleteAccount}
         title="Delete Account"
         message="⚠️ WARNING: This will permanently delete your account and ALL your data! This includes all scraping jobs, scraped data, AI insights, templates, and settings. This action CANNOT be undone!"
         confirmText="Delete Account"
         cancelText="Cancel"
         type="danger"
         requireTextConfirmation={true}
         confirmationText="DELETE"
         isLoading={isDeleting}
         loadingText="Deleting your account and all data..."
       />
    </div>
  );
};

export default Settings;