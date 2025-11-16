import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import api from '../services/api';
import toast from 'react-hot-toast';

export const useAuthStore = create(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,

      login: async (credentials) => {
        set({ isLoading: true });
        try {
          console.log('Attempting login...');
          const response = await api.post('/auth/login', credentials);
          const { access_token } = response.data;
          
          if (!access_token) {
            console.error('No access token received from server');
            throw new Error('Authentication failed: No token received');
          }
          
          console.log('Login successful, token received');
          
          // Set token in API headers
          api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
          
          // Get user info
          console.log('Fetching user info...');
          const userResponse = await api.get('/auth/me');
          
          set({
            token: access_token,
            user: userResponse.data,
            isAuthenticated: true,
            isLoading: false,
          });
          
          console.log('User data stored in state');
          toast.success('Login successful!');
          return { success: true };
        } catch (error) {
          console.error('Login error:', error);
          set({ isLoading: false });
          
          // Handle different types of errors
          if (!error.response) {
            // Network error
            toast.error('Network error. Please check your connection and ensure the backend server is running.');
            return { success: false, error: 'Network error' };
          }
          
          const message = error.response?.data?.detail || 'Login failed';
          toast.error(message);
          return { success: false, error: message };
        }
      },

      register: async (userData) => {
        set({ isLoading: true });
        try {
          await api.post('/auth/register', userData);
          set({ isLoading: false });
          toast.success('Registration successful! Please login.');
          return { success: true };
        } catch (error) {
          set({ isLoading: false });
          const message = error.response?.data?.detail || 'Registration failed';
          toast.error(message);
          return { success: false, error: message };
        }
      },

      logout: () => {
        // Remove token from API headers
        delete api.defaults.headers.common['Authorization'];
        
        set({
          user: null,
          token: null,
          isAuthenticated: false,
        });
        
        toast.success('Logged out successfully');
      },

      checkAuth: async () => {
        const { token } = get();
        if (!token) return;

        try {
          // Set token in API headers
          api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
          
          // Verify token
          console.log('Verifying token...');
          const response = await api.get('/auth/verify-token');
          if (response.data.valid) {
            console.log('Token is valid, getting user data...');
            // Get fresh user data
            const userResponse = await api.get('/auth/me');
            set({
              user: userResponse.data,
              isAuthenticated: true,
            });
            console.log('User authenticated successfully');
          } else {
            console.log('Token is invalid, logging out...');
            get().logout();
          }
        } catch (error) {
          console.error('Error verifying token:', error);
          get().logout();
        }
      },

      updateUser: (userData) => {
        set((state) => ({
          user: { ...state.user, ...userData },
        }));
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        token: state.token,
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);