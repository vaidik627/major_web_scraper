import axios from 'axios';
import toast from 'react-hot-toast';

const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

console.log('API URL:', API_URL);

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
    console.log('Making API request to:', config.url);
    console.log('Full request URL:', config.baseURL + config.url);
    
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
    console.log('API response received:', response.config.url);
    return response;
  },
  async (error) => {
    console.error('API error occurred:', error);
    console.error('Error config:', error.config);
    console.error('Error response:', error.response);
    console.error('Error request:', error.request);
    
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
          if (!config?.silent) toast.error('Unauthorized.');
          break;
          
        case 403:
          if (!config?.silent) toast.error('Access denied. You do not have permission to perform this action.');
          break;
          
        case 404:
          if (!config?.silent) toast.error('Resource not found.');
          break;
          
        case 422:
          // Validation error
          const detail = response.data?.detail;
          if (Array.isArray(detail)) {
            if (!config?.silent) {
              detail.forEach(err => {
                toast.error(`${err.loc?.join(' ')}: ${err.msg}`);
              });
            }
          } else {
            if (!config?.silent) toast.error(detail || 'Validation error');
          }
          break;
          
        case 429:
          if (!config?.silent) toast.error('Too many requests. Please try again later.');
          break;
          
        case 500:
          if (!config?.silent) toast.error('Internal server error. Please try again later.');
          break;
          
        default:
          if (!config?.silent) toast.error(response.data?.detail || 'An unexpected error occurred');
      }
    } else if (error.request) {
      // Network error
      console.error('Network error details:', error);
      if (!config?.silent) {
        toast.error('Network error. Please check your connection and ensure the backend server is running.');
      }
    } else {
      // Other error
      console.error('Unexpected error:', error);
      if (!config?.silent) toast.error('An unexpected error occurred. Please try again later.');
    }
    
    return Promise.reject(error);
  }
);

export default api;