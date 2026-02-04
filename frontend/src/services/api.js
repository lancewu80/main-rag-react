import axios from 'axios';

// 根據環境變數設定 API 基礎 URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// 創建 axios 實例
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30秒超時
  headers: {
    'Content-Type': 'application/json',
  }
});

// 請求攔截器（可添加 token 等）
api.interceptors.request.use(
  config => {
    // 可在這裡添加認證 token
    // const token = localStorage.getItem('token');
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

// 響應攔截器
api.interceptors.response.use(
  response => response.data,
  error => {
    console.error('API 錯誤:', error.response?.data || error.message);

    // 統一的錯誤處理
    if (error.response) {
      // 伺服器返回錯誤
      const errorData = error.response.data;
      throw {
        message: errorData.detail || errorData.message || '伺服器錯誤',
        status: error.response.status,
        data: errorData
      };
    } else if (error.request) {
      // 請求發送但無回應
      throw {
        message: '無法連接到伺服器，請檢查網路連接',
        status: 0,
        data: null
      };
    } else {
      // 其他錯誤
      throw {
        message: error.message,
        status: -1,
        data: null
      };
    }
  }
);

// ============ 系統相關 API ============

/**
 * 獲取系統健康狀態
 */
export const getSystemHealth = async () => {
  return await api.get('/system/health');
};

/**
 * 獲取系統資訊
 */
export const getSystemInfo = async () => {
  return await api.get('/system/info');
};

/**
 * 獲取系統配置
 */
export const getSystemConfig = async () => {
  return await api.get('/system/config');
};

// ============ 文件管理 API ============

/**
 * 獲取文件狀態
 */
export const getDocumentStatus = async () => {
  return await api.get('/documents/status');
};

/**
 * 上傳文件
 * @param {File} file - 要上傳的文件
 */
export const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  return await api.post('/documents/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    }
  });
};

/**
 * 刪除文件
 * @param {string} filename - 文件名
 */
export const deleteDocument = async (filename) => {
  return await api.delete(`/documents/${filename}`);
};

/**
 * 列出所有文件
 */
export const listDocuments = async () => {
  return await api.get('/documents/list');
};

// ============ 問答相關 API ============

/**
 * 通用問答
 * @param {string} question - 問題
 * @param {string} type - 問答類型 (rag/web/hybrid)
 * @param {object} options - 其他選項
 */
export const askQuestion = async (question, type = 'rag', options = {}) => {
  return await api.post('/qa/ask', {
    question,
    type,
    options
  });
};

/**
 * RAG 問答（本地文件）
 * @param {string} question - 問題
 */
export const ragQA = async (question) => {
  return await api.post('/qa/rag', { question });
};

/**
 * 網路搜尋問答
 * @param {string} question - 問題
 */
export const webQA = async (question) => {
  return await api.post('/qa/web', { question });
};

/**
 * 混合問答
 * @param {string} question - 問題
 */
export const hybridQA = async (question) => {
  return await api.post('/qa/hybrid', { question });
};

// ============ 知識庫管理 API ============

/**
 * 建立/重建知識庫
 * @param {boolean} force - 是否強制重建
 * @param {boolean} incremental - 是否增量更新
 */
export const buildKnowledgeBase = async (force = false, incremental = true) => {
  return await api.post('/knowledge/build', { force, incremental });
};

/**
 * 獲取建置任務狀態
 * @param {string} taskId - 任務 ID
 */
export const getBuildTaskStatus = async (taskId) => {
  return await api.get(`/knowledge/task/${taskId}`);
};

/**
 * 獲取知識庫狀態
 */
export const getKnowledgeBaseStatus = async () => {
  return await api.get('/knowledge/status');
};

/**
 * 增量更新知識庫
 */
export const updateKnowledgeBase = async () => {
  return await api.post('/knowledge/update');
};

export default api;
