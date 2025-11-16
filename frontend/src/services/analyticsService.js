import api from './api';

export const analyticsService = {
  // Enhanced AI Analytics
  async getEnhancedAnalysis(data) {
    try {
      const response = await api.post('/ai/enhanced-analysis-content', data);
      return response.data;
    } catch (error) {
      console.error('Enhanced analysis failed:', error);
      throw error;
    }
  },

  async getUserTrends(domain = null, days = 30, trendTypes = null) {
    try {
      const params = new URLSearchParams();
      if (domain) params.append('domain', domain);
      params.append('days', days.toString());
      if (trendTypes) params.append('trend_types', trendTypes.join(','));

      const response = await api.get(`/ai/user-trends?${params.toString()}`);
      return response.data;
    } catch (error) {
      console.error('User trends failed:', error);
      throw error;
    }
  },

  async compareDomains(domains, metric = 'sentiment', days = 30) {
    try {
      const response = await api.post('/ai/compare-domains', {
        domains,
        metric,
        days
      });
      return response.data;
    } catch (error) {
      console.error('Domain comparison failed:', error);
      throw error;
    }
  },

  async getExtractedEntities(jobId) {
    try {
      const response = await api.get(`/ai/entities/${jobId}`);
      return response.data;
    } catch (error) {
      console.error('Get entities failed:', error);
      throw error;
    }
  },

  async getContentCategories(jobId) {
    try {
      const response = await api.get(`/ai/categories/${jobId}`);
      return response.data;
    } catch (error) {
      console.error('Get categories failed:', error);
      throw error;
    }
  },

  // Existing AI endpoints
  async smartAnalysis(url, analysisType = 'comprehensive', customPrompt = null) {
    try {
      const response = await api.post('/ai/smart-analysis', {
        url,
        analysis_type: analysisType,
        custom_prompt: customPrompt
      });
      return response.data;
    } catch (error) {
      console.error('Smart analysis failed:', error);
      throw error;
    }
  },

  async analyzeContent(content, title = '', url = '', analysisType = 'comprehensive') {
    try {
      const response = await api.post('/ai/analyze-content', {
        content,
        title,
        url,
        analysis_type: analysisType
      });
      return response.data;
    } catch (error) {
      console.error('Content analysis failed:', error);
      throw error;
    }
  },

  async compareSites(urls, comparisonCriteria = ['content', 'structure']) {
    try {
      const response = await api.post('/ai/compare-sites', {
        urls,
        comparison_criteria: comparisonCriteria
      });
      return response.data;
    } catch (error) {
      console.error('Site comparison failed:', error);
      throw error;
    }
  },

  async comprehensiveAnalysis(content, title = '', url = '', features = ['summary', 'keywords', 'topics', 'sentiment', 'entities', 'insights']) {
    try {
      const response = await api.post('/ai/comprehensive-analysis', {
        content,
        title,
        url,
        analysis_features: features
      });
      return response.data;
    } catch (error) {
      console.error('Site comparison failed:', error);
      throw error;
    }
  }
};

export default analyticsService;