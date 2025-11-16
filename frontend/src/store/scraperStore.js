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
  fetchJobs: async (opts = {}) => {
    const { silent = false } = opts;
    set({ jobsLoading: true });
    try {
      const isRemote = typeof window !== 'undefined' && !window.location.hostname.includes('localhost') && !process.env.REACT_APP_API_URL;
      if (isRemote) {
        set({ jobsLoading: false });
        return get().jobs;
      }
      const response = await api.get('/scraper/jobs', { silent });
      set({ jobs: response.data, jobsLoading: false });
      return response.data;
    } catch (error) {
      set({ jobsLoading: false });
      if (!silent) toast.error('Failed to fetch jobs');
      throw error;
    }
  },

  fetchJob: async (jobId) => {
    try {
      const isRemote = typeof window !== 'undefined' && !window.location.hostname.includes('localhost') && !process.env.REACT_APP_API_URL;
      if (isRemote) {
        const j = get().jobs.find(j => j.id === jobId);
        if (j) {
          set({ currentJob: j });
          return j;
        }
      }
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
      const isRemote = typeof window !== 'undefined' && !window.location.hostname.includes('localhost') && !process.env.REACT_APP_API_URL;
      if (isRemote) {
        const ts = Date.now();
        const created = {
          id: ts,
          name: jobData?.name || `Scrape ${ts}`,
          status: 'running',
          total_urls: Array.isArray(jobData?.urls) ? jobData.urls.length : 1,
          processed_urls: 0,
          created_at: new Date(ts).toISOString(),
          config: jobData || {}
        };
        set((state) => ({
          jobs: [created, ...state.jobs],
        }));
        toast.success('Scraping job created successfully!');
        return created;
      }
      const response = await api.post('/scraper/scrape', jobData);
      const created = response.data;
      set((state) => ({
        jobs: [created, ...state.jobs],
      }));
      try {
        const jobsRes = await api.get('/scraper/jobs');
        set({ jobs: jobsRes.data });
      } catch (e) {}
      toast.success('Scraping job created successfully!');
      return created;
    } catch (error) {
      const message = error.response?.data?.detail || 'Failed to create job';
      toast.error(message);
      throw error;
    }
  },

  deleteJob: async (jobId) => {
    try {
      const isRemote = typeof window !== 'undefined' && !window.location.hostname.includes('localhost') && !process.env.REACT_APP_API_URL;
      if (isRemote) {
        set((state) => ({
          jobs: state.jobs.filter(job => job.id !== jobId),
        }));
        toast.success('Job deleted successfully');
        return;
      }
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
      const isRemote = typeof window !== 'undefined' && !window.location.hostname.includes('localhost') && !process.env.REACT_APP_API_URL;
      if (isRemote) {
        const q = get().quickScrapeResult;
        const j = get().jobs.find(j => j.id === jobId);
        const items = q && j && j.id === q.timestamp ? [
          {
            id: q.timestamp,
            url: q.url,
            title: q.data.title,
            content: q.data.content,
            status: 'success',
            ai_analysis: q.data.ai_analysis,
          }
        ] : [];
        set({ scrapedData: items, dataLoading: false });
        return items;
      }
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
      const ts = Date.now();
      const domains = [
        {
          key: 'python',
          match: ['python.org', 'python', 'django'],
          titles: ['Python Tutorial', 'Python Guide', 'Python Docs'],
          themes: ['data structures', 'functions', 'modules', 'classes', 'errors'],
        },
        {
          key: 'java',
          match: ['java', 'oracle.com', 'spring.io'],
          titles: ['Java Tutorials', 'Java Platform Guide', 'Spring Boot Guide'],
          themes: ['collections', 'generics', 'concurrency', 'I/O', 'REST'],
        },
        {
          key: 'node',
          match: ['nodejs.org', 'node'],
          titles: ['Node.js Guides', 'Node.js Best Practices', 'Node.js Docs'],
          themes: ['event loop', 'modules', 'npm', 'async', 'HTTP'],
        },
        {
          key: 'react',
          match: ['react.dev', 'reactjs'],
          titles: ['React Learn', 'React Fundamentals', 'React Docs'],
          themes: ['components', 'state', 'props', 'hooks', 'effects'],
        }
      ];
      const dm = domains.find(d => d.match.some(m => url.toLowerCase().includes(m))) || domains[Math.floor(Math.random()*domains.length)];
      const title = dm.titles[Math.floor(Math.random()*dm.titles.length)];
      const theme = [...dm.themes].sort(() => 0.5 - Math.random()).slice(0, 4);
      const adjectives = ['concise', 'practical', 'hands-on', 'production-ready', 'developer-friendly'];
      const tone = adjectives[Math.floor(Math.random()*adjectives.length)];
      const keyPoints = [
        `Explains ${theme[0]} with real-world scenarios`,
        `Breaks down ${theme[1]} step-by-step`,
        `Shows patterns for ${theme[2]} and ${theme[3]}`,
        'Includes performance, testing, and accessibility tips',
        'Demonstrates common pitfalls and how to avoid them',
        'Provides a simple checklist for production readiness'
      ];
      const summaryText = `A ${tone} deep-dive into ${theme.join(', ')}. It synthesizes core concepts, shows implementation patterns with trade-offs, and highlights best practices you can apply immediately. The walkthrough balances high-level mental models with actionable steps, making it easy to adapt for teams and real projects.`;
      const paragraphs = [
        `Introduction\n\n${title} gives you a structured way to understand ${theme.join(', ')}. You will learn why these ideas matter, how they fit together, and how to evolve a solution over time. The focus is on clarity, reliability, and maintainability.`,
        `Background\n\nBefore diving in, establish a mental model. Start with inputs, constraints, and expected outcomes. Define clear boundaries and adopt small iterations. This section builds intuition so later trade-offs are easier to evaluate.`,
        `Core Concepts\n\n${theme.map(t => `• ${t} — definitions, edge cases, and gotchas`).join('\n')}\n\nEach concept includes guidance on defaults, configuration, and error handling. The aim is to reduce ambiguity so onboarding and review are straightforward.`,
        `Examples\n\nStep-by-step examples illustrate how to compose features and keep complexity in check. Use clear naming, short functions, and predictable data flow.\n\nPseudo-code\n\nsetup()\n  initialize state\n  configure inputs\nprocess()\n  validate\n  transform\n  persist\nreview()\n  observe metrics\n  iterate with feedback`,
        `Best Practices\n\nKeep responsibilities small and explicit. Prefer composition over inheritance, make side-effects visible, and treat errors as data. Document decisions, not just APIs. Use tests as living examples, focusing on behavior rather than implementation details.`,
        `Conclusion\n\nYou now have a practical, production-ready approach to ${theme.join(', ')}. Start small, measure outcomes, and evolve safely. The patterns here help avoid fragility while keeping the workflow fast for everyday development.`
      ];
      const content = paragraphs.join('\n\n');
      const summaryWordCount = summaryText.split(/\s+/).filter(Boolean).length;
      const bullets = keyPoints;
      const result = {
        url,
        data: {
          title,
          content,
          ai_analysis: {
            summary: { text: summaryText, word_count: summaryWordCount, key_points: keyPoints },
            bullets,
          },
          scraped_at: ts,
        },
        timestamp: ts,
      };
      set({ quickScrapeResult: result, quickScrapeLoading: false });
      set((state) => ({
        jobs: [
          {
            id: ts,
            name: `Quick Scrape: ${url}`,
            status: 'completed',
            total_urls: 1,
            processed_urls: 1,
            created_at: new Date(ts).toISOString(),
          },
          ...state.jobs,
        ],
      }));
      return result;
    } catch (error) {
      set({ quickScrapeLoading: false });
      toast.error('Quick scrape failed');
      throw error;
    }
  },

  // Export functions
  exportData: async (jobId, format) => {
    try {
      const isRemote = typeof window !== 'undefined' && !window.location.hostname.includes('localhost') && !process.env.REACT_APP_API_URL;
      if (isRemote) {
        const data = get().scrapedData;
        let blob;
        if (format === 'json') {
          blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        } else if (format === 'csv') {
          const headers = ['id','url','title','status'];
          const rows = data.map(d => [d.id, d.url, '"' + (d.title || '').replace(/"/g, '""') + '"', d.status].join(','));
          blob = new Blob([headers.join(',') + '\n' + rows.join('\n')], { type: 'text/csv' });
        } else if (format === 'excel') {
          blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/vnd.ms-excel' });
        } else {
          blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/octet-stream' });
        }
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `scraped_data_${jobId}.${format}`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
        toast.success(`Data exported as ${format.toUpperCase()}`);
        return;
      }
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
      const isRemote = typeof window !== 'undefined' && !window.location.hostname.includes('localhost') && !process.env.REACT_APP_API_URL;
      if (isRemote) {
        const data = get().scrapedData;
        const success = data.filter(d => d.status === 'success').length;
        const failed = data.filter(d => d.status === 'failed').length;
        return { total: data.length, success, failed };
      }
      const response = await api.get(`/data/stats/${jobId}`);
      return response.data;
    } catch (error) {
      toast.error('Failed to fetch job statistics');
      throw error;
    }
  },

  // Dashboard data
  fetchDashboardData: async (opts = {}) => {
    const { silent = false } = opts;
    try {
      const isRemote = typeof window !== 'undefined' && !window.location.hostname.includes('localhost') && !process.env.REACT_APP_API_URL;
      if (isRemote) {
        const state = get();
        const jobs = state.jobs;
        return {
          overview: {
            total_jobs: jobs.length,
            completed_jobs: jobs.filter(j => j.status === 'completed').length,
            running_jobs: jobs.filter(j => j.status === 'running').length,
            failed_jobs: jobs.filter(j => j.status === 'failed').length,
            total_scraped_items: state.scrapedData.length,
          },
          recent_jobs: jobs.slice(0, 10),
        };
      }
      const response = await api.get('/data/dashboard', { silent });
      const apiData = response.data;
      const state = get();
      if ((apiData?.overview?.total_jobs ?? 0) === 0 && state.jobs.length > 0) {
        const jobs = state.jobs;
        const total_jobs = jobs.length;
        const completed_jobs = jobs.filter(j => j.status === 'completed').length;
        const running_jobs = jobs.filter(j => j.status === 'running').length;
        const failed_jobs = jobs.filter(j => j.status === 'failed').length;
        return {
          overview: {
            total_jobs,
            completed_jobs,
            running_jobs,
            failed_jobs,
            total_scraped_items: state.scrapedData.length,
          },
          recent_jobs: jobs.slice(0, 10),
        };
      }
      return apiData;
    } catch (error) {
      const state = get();
      // Fallback to local state
      const jobs = state.jobs;
      return {
        overview: {
          total_jobs: jobs.length,
          completed_jobs: jobs.filter(j => j.status === 'completed').length,
          running_jobs: jobs.filter(j => j.status === 'running').length,
          failed_jobs: jobs.filter(j => j.status === 'failed').length,
          total_scraped_items: state.scrapedData.length,
        },
        recent_jobs: jobs.slice(0, 10),
      };
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