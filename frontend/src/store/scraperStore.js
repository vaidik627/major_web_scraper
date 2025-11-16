import { create } from 'zustand';
import api from '../services/api';
import toast from 'react-hot-toast';

export const useScraperStore = create((set, get) => ({
  // Jobs state
  jobs: [],
  currentJob: null,
  jobsLoading: false,
  
  // Scraped data state
  scrapedData: [],
  dataLoading: false,
  
  // Quick scrape state
  quickScrapeResult: null,
  quickScrapeLoading: false,
  
  // UI state
  selectedUrls: [],
  scrapingConfig: {
    css_selector: '',
    xpath: '',
    keywords: [],
    data_type: 'text',
    use_playwright: false,
    wait_time: 3,
    max_pages: 1,
    follow_links: false,
    extract_images: false,
    extract_links: false,
    custom_headers: {},
  },

  // Actions
  setScrapingConfig: (config) => {
    set((state) => ({
      scrapingConfig: { ...state.scrapingConfig, ...config },
    }));
  },

  setSelectedUrls: (urls) => {
    set({ selectedUrls: urls });
  },

  addUrl: (url) => {
    set((state) => ({
      selectedUrls: [...state.selectedUrls, url],
    }));
  },

  removeUrl: (index) => {
    set((state) => ({
      selectedUrls: state.selectedUrls.filter((_, i) => i !== index),
    }));
  },

  // Job management
  fetchJobs: async () => {
    set({ jobsLoading: true });
    try {
      const response = await api.get('/scraper/jobs');
      set({ jobs: response.data, jobsLoading: false });
    } catch (error) {
      set({ jobsLoading: false });
      toast.error('Failed to fetch jobs');
    }
  },

  fetchJob: async (jobId) => {
    try {
      const response = await api.get(`/scraper/jobs/${jobId}`);
      set({ currentJob: response.data });
      return response.data;
    } catch (error) {
      toast.error('Failed to fetch job details');
      throw error;
    }
  },

  createJob: async (jobData) => {
    try {
      const response = await api.post('/scraper/scrape', jobData);
      set((state) => ({
        jobs: [response.data, ...state.jobs],
      }));
      toast.success('Scraping job created successfully!');
      return response.data;
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to create job';
      toast.error(message);
      throw error;
    }
  },

  deleteJob: async (jobId) => {
    try {
      await api.delete(`/scraper/jobs/${jobId}`);
      set((state) => ({
        jobs: state.jobs.filter(job => job.id !== jobId),
      }));
      toast.success('Job deleted successfully');
    } catch (error) {
      toast.error('Failed to delete job');
      throw error;
    }
  },

  // Data management
  fetchJobData: async (jobId) => {
    set({ dataLoading: true });
    try {
      const response = await api.get(`/scraper/jobs/${jobId}/data`);
      set({ scrapedData: response.data, dataLoading: false });
      return response.data;
    } catch (error) {
      set({ dataLoading: false });
      toast.error('Failed to fetch scraped data');
      throw error;
    }
  },

  // Quick scrape
  quickScrape: async (url, config = {}) => {
    set({ quickScrapeLoading: true });
    try {
      const response = await api.post('/scraper/quick-scrape', null, {
        params: { url, ...config },
      });
      set({ 
        quickScrapeResult: response.data, 
        quickScrapeLoading: false 
      });
      return response.data;
    } catch (error) {
      set({ quickScrapeLoading: false });
      const message = error.response?.data?.detail || 'Quick scrape failed';
      toast.error(message);
      throw error;
    }
  },

  // Export functions
  exportData: async (jobId, format) => {
    try {
      const response = await api.get(`/data/export/${jobId}/${format}`, {
        responseType: 'blob',
      });
      
      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      
      const contentDisposition = response.headers['content-disposition'];
      const filename = contentDisposition
        ? contentDisposition.split('filename=')[1].replace(/"/g, '')
        : `scraped_data_${jobId}.${format}`;
      
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      toast.success(`Data exported as ${format.toUpperCase()}`);
    } catch (error) {
      toast.error(`Failed to export data as ${format.toUpperCase()}`);
      throw error;
    }
  },

  // Statistics
  fetchJobStats: async (jobId) => {
    try {
      const response = await api.get(`/data/stats/${jobId}`);
      return response.data;
    } catch (error) {
      toast.error('Failed to fetch job statistics');
      throw error;
    }
  },

  // Dashboard data
  fetchDashboardData: async () => {
    try {
      const response = await api.get('/data/dashboard');
      return response.data;
    } catch (error) {
      toast.error('Failed to fetch dashboard data');
      throw error;
    }
  },

  // Clear states
  clearQuickScrapeResult: () => {
    set({ quickScrapeResult: null });
  },

  clearCurrentJob: () => {
    set({ currentJob: null });
  },

  clearScrapedData: () => {
    set({ scrapedData: [] });
  },

  // Real-time job updates
  updateJobStatus: (jobId, status) => {
    set((state) => ({
      jobs: state.jobs.map(job =>
        job.id === jobId ? { ...job, status } : job
      ),
      currentJob: state.currentJob?.id === jobId
        ? { ...state.currentJob, status }
        : state.currentJob,
    }));
  },

  updateJobProgress: (jobId, processedUrls) => {
    set((state) => ({
      jobs: state.jobs.map(job =>
        job.id === jobId ? { ...job, processed_urls: processedUrls } : job
      ),
      currentJob: state.currentJob?.id === jobId
        ? { ...state.currentJob, processed_urls: processedUrls }
        : state.currentJob,
    }));
  },
}));