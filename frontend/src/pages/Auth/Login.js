import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { EyeIcon, EyeSlashIcon, GlobeAltIcon } from '@heroicons/react/24/outline';
import { useAuthStore } from '../../store/authStore';

const Login = () => {
  const [showPassword, setShowPassword] = useState(false);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const { login, isLoading } = useAuthStore();
  const navigate = useNavigate();
  const location = useLocation();
  
  const from = location.state?.from?.pathname || '/dashboard';

  // Monitor online/offline status
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm();

  const [loginError, setLoginError] = useState('');

  const onSubmit = async (data) => {
    setLoginError(''); // Clear any previous errors
    
    // Check if online before attempting login
    if (!isOnline) {
      setLoginError('You are currently offline. Please check your internet connection and try again.');
      return;
    }
    
    try {
      const result = await login(data);
      if (result.success) {
        navigate(from, { replace: true });
      } else if (result.error) {
        setLoginError(result.error);
      }
    } catch (error) {
      console.error('Login submission error:', error);
      setLoginError('An unexpected error occurred. Please try again.');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-brand-50 via-white to-primary-50 dark:from-gray-900 dark:via-gray-900 dark:to-gray-800 py-12 px-4 sm:px-6 lg:px-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="max-w-md w-full"
      >
        <div className="card p-8 shadow-soft">
          {/* Header */}
          <div className="text-center">
            <Link to="/" className="inline-flex items-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-brand-600 to-primary-600 shadow-glow">
                <GlobeAltIcon className="h-7 w-7 text-white" />
              </div>
              <span className="ml-3 text-2xl font-bold text-gradient">
                AI Web Scraper
              </span>
            </Link>
            <h2 className="mt-6 text-3xl font-bold text-gray-900 dark:text-white">
              Welcome back
            </h2>
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
              Sign in to your account to continue scraping
            </p>
          </div>

          {/* Form */}
          <motion.form
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mt-8 space-y-6"
            onSubmit={handleSubmit(onSubmit)}
          >
          <div className="space-y-4">
            {/* Username */}
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Username
              </label>
              <input
                {...register('username', {
                  required: 'Username is required',
                  minLength: {
                    value: 3,
                    message: 'Username must be at least 3 characters',
                  },
                })}
                type="text"
                className={`mt-1 input ${errors.username ? 'input-error' : ''}`}
                placeholder="Enter your username"
              />
              {errors.username && (
                <p className="mt-1 text-sm text-error-600">{errors.username.message}</p>
              )}
            </div>

            {/* Password */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Password
              </label>
              <div className="mt-1 relative">
                <input
                  {...register('password', {
                    required: 'Password is required',
                    minLength: {
                      value: 6,
                      message: 'Password must be at least 6 characters',
                    },
                  })}
                  type={showPassword ? 'text' : 'password'}
                  className={`input pr-10 ${errors.password ? 'input-error' : ''}`}
                  placeholder="Enter your password"
                />
                <button
                  type="button"
                  className="absolute inset-y-0 right-0 pr-3 flex items-center"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? (
                    <EyeSlashIcon className="h-5 w-5 text-gray-400" />
                  ) : (
                    <EyeIcon className="h-5 w-5 text-gray-400" />
                  )}
                </button>
              </div>
              {errors.password && (
                <p className="mt-1 text-sm text-error-600">{errors.password.message}</p>
              )}
            </div>
          </div>

          {/* Offline Warning */}
          {!isOnline && (
            <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 rounded-md p-3 mb-4">
              <p className="text-sm">You appear to be offline. Please check your internet connection.</p>
            </div>
          )}
          
          {/* Error Message */}
          {loginError && (
            <div className="bg-red-50 border border-red-200 text-red-800 rounded-md p-3">
              <p className="text-sm">{loginError}</p>
            </div>
          )}

          {/* Submit Button */}
          <div>
            <button
              type="submit"
              disabled={isLoading}
              className="w-full btn-primary py-3 text-base font-medium disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="loading-spinner mr-2"></div>
                  Signing in...
                </div>
              ) : (
                'Sign in'
              )}
            </button>
          </div>

          {/* Links */}
          <div className="text-center space-y-2">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Don't have an account?{' '}
              <Link
                to="/register"
                className="font-medium text-brand-600 hover:text-brand-500 dark:text-brand-400 dark:hover:text-brand-300"
              >
                Sign up
              </Link>
            </p>
            <Link
              to="/"
              className="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              ← Back to home
            </Link>
          </div>
        </motion.form>
        </div>
      </motion.div>
    </div>
  );
};

export default Login;