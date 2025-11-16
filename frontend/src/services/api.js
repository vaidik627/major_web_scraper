import axios from 'axios';
import toast from 'react-hot-toast';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: `${API_URL}/api`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
  // Add retry logic for network issues
  retry: 3,
  retryDelay: 1000,
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth-storage');
    if (token) {
      try {
        const authData = JSON.parse(token);
        if (authData.state?.token) {
          config.headers.Authorization = `Bearer ${authData.state.token}`;
        }
      } catch (error) {
        console.error('Error parsing auth token:', error);
      }
    }
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add retry logic for network issues
api.interceptors.request.use(
  async (config) => {
    // Set retry count if not set
    config.retryCount = config.retryCount || 0;
    return config;
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  async (error) => {
    const config = error.config;
    const { response } = error;
    
    // If we have no response and we haven't retried too many times
    if (!response && config && config.retryCount < api.defaults.retry) {
      // Increment the retry count
      config.retryCount = config.retryCount ? config.retryCount + 1 : 1;
      
      // Create a new promise to handle the retry
      const backoff = new Promise((resolve) => {
        setTimeout(() => resolve(), api.defaults.retryDelay || 1000);
      });
      
      // Wait for the backoff time and then retry
      await backoff;
      console.log(`Retrying request (${config.retryCount}/${api.defaults.retry}): ${config.url}`);
      return api(config);
    }
    
    if (response) {
      switch (response.status) {
        case 401:
          // Unauthorized - clear auth and redirect to login
          localStorage.removeItem('auth-storage');
          window.location.href = '/login';
          toast.error('Session expired. Please login again.');
          break;
          
        case 403:
          toast.error('Access denied. You do not have permission to perform this action.');
          break;
          
        case 404:
          toast.error('Resource not found.');
          break;
          
        case 422:
          // Validation error
          const detail = response.data?.detail;
          if (Array.isArray(detail)) {
            detail.forEach(err => {
              toast.error(`${err.loc?.join(' ')}: ${err.msg}`);
            });
          } else {
            toast.error(detail || 'Validation error');
          }
          break;
          
        case 429:
          toast.error('Too many requests. Please try again later.');
          break;
          
        case 500:
          toast.error('Internal server error. Please try again later.');
          break;
          
        default:
          toast.error(response.data?.detail || 'An unexpected error occurred');
      }
    } else if (error.request) {
      // Network error
      console.error('Network error details:', error);
      toast.error('Network error. Please check your connection and ensure the backend server is running.');
    } else {
      // Other error
      console.error('Unexpected error:', error);
      toast.error('An unexpected error occurred. Please try again later.');
    }
    
    return Promise.reject(error);
  }
);

export default api;