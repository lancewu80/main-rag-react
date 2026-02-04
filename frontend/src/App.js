import {
  Api,
  ContentCopy,
  Dashboard,
  Folder,
  History,
  Info,
  Menu as MenuIcon,
  Merge,
  QuestionAnswer,
  Send,
  SmartToy,
  Storage,
  Web
} from '@mui/icons-material';
import {
  Alert,
  AppBar,
  Avatar,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  CssBaseline,
  Divider,
  Drawer,
  Grid,
  IconButton,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  Snackbar,
  TextField,
  ThemeProvider,
  Toolbar,
  Typography,
  createTheme
} from '@mui/material';
import { useEffect, useState } from 'react';
import './App.css';

// 創建主題
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    success: {
      main: '#4caf50',
    },
    warning: {
      main: '#ff9800',
    },
    error: {
      main: '#f44336',
    },
    background: {
      default: '#f8f9fa',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 700,
    },
    h5: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 12,
  },
});

// API 服務
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

class ApiService {
  static async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const defaultOptions = {
      headers: {
        'Content-Type': 'application/json',
      },
    };

    try {
      const response = await fetch(url, { ...defaultOptions, ...options });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: '伺服器錯誤' }));
        throw new Error(error.detail || `HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API 請求失敗:', error);
      throw error;
    }
  }

  // 問答相關
  static async askQuestion(type, question) {
    if (type === 'hybrid') {
      // 使用 hybrid_qa_internal 端點
      return this.request('/qa/hybrid', {
        method: 'POST',
        body: JSON.stringify({ question }),
      });
    } else {
      return this.request(`/qa/${type}`, {
        method: 'POST',
        body: JSON.stringify({ question }),
      });
    }
  }

  // 系統相關
  static async getSystemInfo() {
    return this.request('/system/info');
  }

  static async getSystemHealth() {
    return this.request('/system/health');
  }

  // 搜尋測試
  static async testSearch() {
    return this.request('/qa/search-test');
  }

  // 文件相關
  static async listDocuments() {
    return this.request('/documents/list');
  }

  static async uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);

    return fetch(`${API_BASE_URL}/documents/upload`, {
      method: 'POST',
      body: formData,
    }).then(res => res.json());
  }

  static async deleteDocument(filename) {
    return this.request(`/documents/${filename}`, {
      method: 'DELETE',
    });
  }

  // 知識庫相關
  static async buildKnowledgeBase(force = false) {
    return this.request('/knowledge/build', {
      method: 'POST',
      body: JSON.stringify({ force }),
    });
  }

  static async getKnowledgeStatus() {
    return this.request('/knowledge/status');
  }
}

// 歷史記錄管理函數
class HistoryManager {
  static HISTORY_KEY = 'ai-qa-history-v2';

  // 獲取所有歷史記錄
  static getAllHistory() {
    try {
      const saved = localStorage.getItem(this.HISTORY_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        return Array.isArray(parsed) ? parsed : [];
      }
    } catch (error) {
      console.error('讀取歷史記錄失敗:', error);
    }
    return [];
  }

  // 獲取指定類型的歷史記錄
  static getHistoryByType(type) {
    const allHistory = this.getAllHistory();
    return allHistory.filter(item => item.type === type);
  }

  // 添加歷史記錄
  static addHistory(item) {
    const allHistory = this.getAllHistory();
    // 防止重複（根據問題內容和時間判斷）
    const newHistory = [item, ...allHistory.filter(h =>
      !(h.question === item.question && h.timestamp === item.timestamp)
    )].slice(0, 20); // 最多保留20條

    try {
      localStorage.setItem(this.HISTORY_KEY, JSON.stringify(newHistory));
      return newHistory;
    } catch (error) {
      console.error('保存歷史記錄失敗:', error);
      return allHistory;
    }
  }

  // 清除指定類型的歷史記錄
  static clearHistoryByType(type) {
    const allHistory = this.getAllHistory();
    const filteredHistory = allHistory.filter(item => item.type !== type);

    try {
      localStorage.setItem(this.HISTORY_KEY, JSON.stringify(filteredHistory));
    } catch (error) {
      console.error('清除歷史記錄失敗:', error);
    }

    return filteredHistory;
  }

  // 清除所有歷史記錄
  static clearAllHistory() {
    localStorage.removeItem(this.HISTORY_KEY);
  }
}

// 側邊欄組件
function Sidebar({ activeTab, onTabChange }) {
  const menuItems = [
    { id: 'dashboard', label: '儀表板', icon: <Dashboard />, color: 'primary' },
    { id: 'rag', label: 'AI 智能問答', icon: <QuestionAnswer />, color: 'secondary', badge: 'AI' },
    { id: 'web', label: '網路資訊分析', icon: <Web />, color: 'info' },
    { id: 'hybrid', label: '綜合 AI 分析', icon: <Merge />, color: 'warning', badge: '智能' },
    { id: 'documents', label: '文件管理', icon: <Folder />, color: 'success' },
    { id: 'knowledge', label: '知識庫', icon: <Storage />, color: 'primary' },
    { id: 'api', label: 'API 測試', icon: <Api />, color: 'secondary' },
    { id: 'info', label: '系統資訊', icon: <Info />, color: 'info' },
  ];

  return (
    <Paper sx={{ height: '100%', borderRadius: 3, boxShadow: 3 }}>
      <Box sx={{ p: 3, textAlign: 'center', bgcolor: 'primary.main', color: 'white', borderRadius: '12px 12px 0 0' }}>
        <Avatar sx={{ width: 60, height: 60, bgcolor: 'white', color: 'primary.main', mb: 2, mx: 'auto' }}>
          <SmartToy fontSize="large" />
        </Avatar>
        <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
          🤖 AI 智能問答
        </Typography>
        <Typography variant="caption" sx={{ opacity: 0.8 }}>
          Ollama 驅動版
        </Typography>
      </Box>
      <List sx={{ p: 2 }}>
        {menuItems.map((item) => (
          <ListItem
            key={item.id}
            button
            selected={activeTab === item.id}
            onClick={() => onTabChange(item.id)}
            sx={{
              borderRadius: 2,
              mb: 1,
              '&.Mui-selected': {
                bgcolor: `${item.color}.light`,
                color: `${item.color}.main`,
                '&:hover': {
                  bgcolor: `${item.color}.light`,
                },
              },
            }}
          >
            <ListItemIcon sx={{ color: activeTab === item.id ? `${item.color}.main` : 'inherit' }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText
              primary={item.label}
              primaryTypographyProps={{
                fontWeight: activeTab === item.id ? 'bold' : 'normal'
              }}
            />
            {item.badge && (
              <Chip
                label={item.badge}
                size="small"
                color={item.color}
                sx={{ ml: 1 }}
              />
            )}
          </ListItem>
        ))}
      </List>
    </Paper>
  );
}

// 問答面板組件
function QAPanel({ type }) {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState(() => HistoryManager.getHistoryByType(type));
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });

  const config = {
    rag: {
      title: '🤖 AI 智能問答',
      description: '使用 AI 模型回答您的問題',
      icon: <QuestionAnswer sx={{ fontSize: 40 }} />,
      color: 'secondary',
    },
    web: {
      title: '🌐 網路資訊分析',
      description: '使用 DuckDuckGo 搜尋網路相關資訊並由 AI 分析',
      icon: <Web sx={{ fontSize: 40 }} />,
      color: 'info',
    },
    hybrid: {
      title: '🔀 綜合 AI 分析',
      description: '結合內部知識庫和網路搜尋的綜合回答',
      icon: <Merge sx={{ fontSize: 40 }} />,
      color: 'warning',
    },
  };

  const { title, description, icon, color } = config[type] || config.rag;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setSnackbar({ open: false, message: '', severity: 'info' });
    setAnswer(null);

    try {
      console.log(`發送問題: ${question}, 類型: ${type}`);
      const response = await ApiService.askQuestion(type, question);
      console.log('API 響應:', response);

      setAnswer(response);

      // 創建新的歷史項目
      const newHistoryItem = {
        id: Date.now(),
        question,
        answer: response.answer,
        timestamp: new Date().toLocaleString('zh-TW', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
          hour12: false
        }),
        type,
        metadata: response.metadata,
      };

      // 保存到歷史記錄
      HistoryManager.addHistory(newHistoryItem);
      // 更新本地狀態
      setHistory(HistoryManager.getHistoryByType(type));

      setQuestion('');

      // 根據回答質量顯示不同消息
      const answerQuality = response.metadata?.answer_source ||
                           (response.metadata?.ai_available ? 'good' : 'basic');

      const messages = {
        'ollama_ai': '✅ AI 回答生成成功！',
        'good': '✅ 回答生成完成',
        'basic': 'ℹ️  基礎回答生成',
        'simulation': '⚠️  模擬模式回答',
        'hybrid': '🔀 綜合分析完成'
      };

      setSnackbar({
        open: true,
        message: messages[answerQuality] || messages[type] || '回答生成完成',
        severity: 'success'
      });

    } catch (error) {
      console.error('問答失敗:', error);

      // 提供有用的錯誤信息
      const errorMessage = error.message || '未知錯誤';
      setSnackbar({
        open: true,
        message: `問答失敗: ${errorMessage}`,
        severity: 'error'
      });

      // 提供用戶友好的錯誤回答
      setAnswer({
        answer: `抱歉，處理問題時發生錯誤。\n\n錯誤信息：${errorMessage}\n\n請檢查：\n1. 後端伺服器是否運行\n2. Ollama 服務是否啟動\n3. 網絡連接是否正常`,
        sources: [],
        metadata: {
          type,
          error: errorMessage,
          processing_time: 0
        }
      });

    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    HistoryManager.clearHistoryByType(type);
    setHistory([]);

    setSnackbar({
      open: true,
      message: `${title} 的歷史記錄已清空`,
      severity: 'info'
    });
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setSnackbar({
      open: true,
      message: '已複製到剪貼板',
      severity: 'success'
    });
  };

  const handleSnackbarClose = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  // 當類型變化時，更新歷史記錄
  useEffect(() => {
    setHistory(HistoryManager.getHistoryByType(type));
  }, [type]);

  // 監聽 localStorage 變化（用於跨選項卡同步）
  useEffect(() => {
    const handleStorageChange = (e) => {
      if (e.key === HistoryManager.HISTORY_KEY) {
        setHistory(HistoryManager.getHistoryByType(type));
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, [type]);

  return (
    <Box sx={{ width: '100%' }}>
      <Card sx={{ mb: 3, bgcolor: `${color}.light`, color: `${color}.dark` }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            {icon}
            <Box sx={{ ml: 2 }}>
              <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                {title}
              </Typography>
              <Typography variant="body1">
                {description}
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* 輸入區域 */}
      <Paper sx={{ p: 3, mb: 3, borderRadius: 3 }}>
        <form onSubmit={handleSubmit}>
          <Typography variant="h6" gutterBottom>
            💬 輸入您的問題
          </Typography>
          <TextField
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={`例如：${type === 'rag' ? '什麼是深度學習？' :
                                  type === 'web' ? '台灣最新科技趨勢？' :
                                  '結合內部知識庫和網路搜尋的綜合分析？'}`}
            multiline
            rows={4}
            fullWidth
            variant="outlined"
            disabled={loading}
            sx={{ mb: 2 }}
          />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="caption" color="text.secondary">
              字符數: {question.length}
            </Typography>
            <Button
              type="submit"
              variant="contained"
              color={color}
              disabled={loading || !question.trim()}
              startIcon={loading ? <CircularProgress size={20} /> : <Send />}
              size="large"
            >
              {loading ? 'AI 思考中...' : '發送問題'}
            </Button>
          </Box>
        </form>
      </Paper>

      {loading && (
        <Paper sx={{ p: 3, mb: 3, textAlign: 'center' }}>
          <CircularProgress sx={{ mb: 2 }} />
          <Typography>
            {type === 'web' ? '🌐 正在搜尋網路資訊並由 AI 分析中...' :
             type === 'hybrid' ? '🔀 正在綜合分析內部知識庫和網路資訊...' :
             '🤔 AI 正在思考中，請稍候...'}
          </Typography>
          <LinearProgress sx={{ mt: 2 }} />
        </Paper>
      )}

      {/* 回答區域 */}
      {answer && (
        <Paper sx={{ p: 3, mb: 3, borderRadius: 3, bgcolor: 'background.paper' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
              📋 AI 回答
            </Typography>
            <Box>
              <IconButton onClick={() => copyToClipboard(answer.answer)} title="複製回答">
                <ContentCopy />
              </IconButton>
            </Box>
          </Box>
          <Divider sx={{ mb: 3 }} />

          <Box sx={{
            p: 3,
            bgcolor: 'grey.50',
            borderRadius: 2,
            borderLeft: `4px solid`,
            borderColor: `${color}.main`,
            whiteSpace: 'pre-wrap',
            lineHeight: 1.8,
            fontSize: '1.1rem',
            minHeight: '200px'
          }}>
            {answer.answer}
          </Box>

          {/* 元數據 */}
          {answer.metadata && !answer.metadata.error && (
            <Box sx={{ mt: 3, p: 2, bgcolor: 'info.50', borderRadius: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <Typography variant="caption" color="text.secondary">模型</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {answer.metadata.model || answer.metadata.model_used || 'AI 模型'}
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="caption" color="text.secondary">處理時間</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {answer.metadata.processing_time ? `${answer.metadata.processing_time}秒` : 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="caption" color="text.secondary">回答類型</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {type === 'rag' ? '智能問答' :
                     type === 'web' ? '網路分析' :
                     type === 'hybrid' ? '綜合分析' : '其他'}
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="caption" color="text.secondary">回答來源</Typography>
                  <Typography variant="body2" fontWeight="bold" color={type === 'hybrid' ? 'warning.main' : 'success.main'}>
                    {type === 'hybrid' ? '內部知識庫 + 網路' :
                     type === 'web' ? '網路搜尋' : 'AI 知識庫'}
                  </Typography>
                </Grid>
              </Grid>

              {/* 混合分析專用資訊 */}
              {type === 'hybrid' && answer.metadata && (
                <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid #e0e0e0' }}>
                  <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
                    🔀 綜合分析資訊
                  </Typography>
                  <Grid container spacing={2}>
                    {answer.metadata.search_engine && (
                      <Grid item xs={6} md={4}>
                        <Typography variant="caption" color="text.secondary">搜尋引擎</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {answer.metadata.search_engine === 'simulated' ? '模擬搜尋' : 'DuckDuckGo'}
                        </Typography>
                      </Grid>
                    )}
                    {answer.metadata.answer_source && (
                      <Grid item xs={6} md={4}>
                        <Typography variant="caption" color="text.secondary">主要來源</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {answer.metadata.answer_source === 'internal_knowledge' ? '內部知識庫' :
                           answer.metadata.answer_source === 'web_search' ? '網路搜尋' :
                           answer.metadata.answer_source === 'hybrid' ? '綜合來源' : '未知'}
                        </Typography>
                      </Grid>
                    )}
                    {answer.metadata.search_results_count !== undefined && (
                      <Grid item xs={6} md={4}>
                        <Typography variant="caption" color="text.secondary">搜尋結果數</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {answer.metadata.search_results_count}
                        </Typography>
                      </Grid>
                    )}
                    {answer.metadata.internal_knowledge_used !== undefined && (
                      <Grid item xs={6} md={4}>
                        <Typography variant="caption" color="text.secondary">內部知識庫</Typography>
                        <Typography variant="body2" fontWeight="bold"
                          color={answer.metadata.internal_knowledge_used ? 'success.main' : 'warning.main'}>
                          {answer.metadata.internal_knowledge_used ? '已使用' : '未使用'}
                        </Typography>
                      </Grid>
                    )}
                  </Grid>
                </Box>
              )}

              {/* 網路搜尋相關資訊 */}
              {type === 'web' && answer.metadata.search_engine && (
                <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid #e0e0e0' }}>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">搜尋引擎</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {answer.metadata.search_engine === 'simulated' ? '模擬搜尋' : 'DuckDuckGo'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">搜尋狀態</Typography>
                      <Typography variant="body2" fontWeight="bold"
                        color={answer.metadata.search_status === 'success' ? 'success.main' : 'warning.main'}>
                        {answer.metadata.search_status === 'success' ? '成功' : answer.metadata.search_status || '未知'}
                      </Typography>
                    </Grid>
                    {answer.metadata.search_results_count !== undefined && (
                      <Grid item xs={6}>
                        <Typography variant="caption" color="text.secondary">搜尋結果數</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {answer.metadata.search_results_count}
                        </Typography>
                      </Grid>
                    )}
                  </Grid>
                </Box>
              )}
            </Box>
          )}

          {answer.metadata?.error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              AI 服務錯誤: {answer.metadata.error}
            </Alert>
          )}
        </Paper>
      )}

      {/* 歷史記錄 */}
      <Paper sx={{
        p: 3,
        borderRadius: 3,
        mt: 3,
        border: '1px solid',
        borderColor: 'divider'
      }}>
        <Box sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3
        }}>
          <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
            <History sx={{ mr: 1, verticalAlign: 'middle' }} />
            問答歷史 {history.length > 0 && `(${history.length})`}
          </Typography>
          {history.length > 0 && (
            <Button
              onClick={clearHistory}
              size="small"
              color="error"
              variant="outlined"
            >
              清空歷史
            </Button>
          )}
        </Box>

        {history.length > 0 ? (
          <Box sx={{
            maxHeight: 400,
            overflow: 'auto',
            pr: 1,
            '&::-webkit-scrollbar': {
              width: '8px',
            },
            '&::-webkit-scrollbar-track': {
              background: '#f1f1f1',
              borderRadius: '4px',
            },
            '&::-webkit-scrollbar-thumb': {
              background: '#888',
              borderRadius: '4px',
            },
            '&::-webkit-scrollbar-thumb:hover': {
              background: '#555',
            }
          }}>
            {history.map((item, index) => (
              <Paper
                key={item.id || index}
                sx={{
                  p: 2,
                  mb: 2,
                  borderRadius: 2,
                  cursor: 'pointer',
                  border: '1px solid',
                  borderColor: 'divider',
                  '&:hover': {
                    bgcolor: 'action.hover',
                    boxShadow: 2,
                    transform: 'translateY(-2px)',
                    transition: 'all 0.2s ease'
                  },
                  transition: 'all 0.2s ease'
                }}
                onClick={() => {
                  setAnswer({
                    answer: item.answer,
                    sources: [],
                    metadata: item.metadata
                  });
                  setQuestion(item.question);
                  // 滾動到頂部
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
              >
                <Box sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'flex-start',
                  mb: 1
                }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Chip
                      label={item.type === 'rag' ? 'AI 問答' :
                             item.type === 'web' ? '網路分析' :
                             item.type === 'hybrid' ? '綜合分析' : '其他'}
                      size="small"
                      color={config[item.type]?.color || 'primary'}
                      variant="outlined"
                    />
                    <Typography variant="caption" color="text.secondary">
                      #{history.length - index}
                    </Typography>
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    {item.timestamp || new Date().toLocaleTimeString()}
                  </Typography>
                </Box>
                <Typography
                  variant="body1"
                  fontWeight="medium"
                  gutterBottom
                  sx={{
                    display: '-webkit-box',
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis'
                  }}
                >
                  Q: {item.question}
                </Typography>
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{
                    display: '-webkit-box',
                    WebkitLineClamp: 3,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    fontSize: '0.875rem',
                    lineHeight: 1.6
                  }}
                >
                  A: {item.answer || '無回答內容'}
                </Typography>
                {item.metadata?.processing_time && (
                  <Box sx={{
                    display: 'flex',
                    justifyContent: 'flex-end',
                    mt: 1
                  }}>
                    <Typography variant="caption" color="primary">
                      處理時間: {item.metadata.processing_time}秒
                    </Typography>
                  </Box>
                )}
              </Paper>
            ))}
          </Box>
        ) : (
          <Box sx={{
            textAlign: 'center',
            py: 4,
            color: 'text.secondary'
          }}>
            <History sx={{ fontSize: 48, opacity: 0.5, mb: 2 }} />
            <Typography variant="body1">
              暫無問答歷史
            </Typography>
            <Typography variant="body2">
              開始提問後，您的歷史記錄將會顯示在這裡
            </Typography>
          </Box>
        )}
      </Paper>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleSnackbarClose} severity={snackbar.severity}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}

// 系統資訊面板
function SystemInfoPanel() {
  const [systemInfo, setSystemInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [apiStatus, setApiStatus] = useState('checking');
  const [searchStatus, setSearchStatus] = useState('checking');
  const [searchTestResult, setSearchTestResult] = useState(null);

  useEffect(() => {
    checkSystemStatus();
    checkSearchStatus();
  }, []);

  const checkSystemStatus = async () => {
    setLoading(true);
    try {
      const [info, health] = await Promise.all([
        ApiService.getSystemInfo().catch(() => null),
        ApiService.getSystemHealth().catch(() => null)
      ]);

      setSystemInfo(info);
      setApiStatus(info ? 'connected' : 'disconnected');
    } catch (error) {
      console.error('檢查系統狀態失敗:', error);
      setApiStatus('error');
    } finally {
      setLoading(false);
    }
  };

  const checkSearchStatus = async () => {
    try {
      const result = await ApiService.testSearch();
      setSearchTestResult(result);

      if (result.status === 'ok' && result.duckduckgo_available) {
        setSearchStatus('connected');
      } else {
        setSearchStatus('disconnected');
      }
    } catch (error) {
      console.error('檢查搜尋狀態失敗:', error);
      setSearchStatus('error');
    }
  };

  const getApiStatusColor = () => {
    switch (apiStatus) {
      case 'connected': return 'success';
      case 'disconnected': return 'error';
      case 'checking': return 'info';
      default: return 'warning';
    }
  };

  const getApiStatusText = () => {
    switch (apiStatus) {
      case 'connected': return '已連接';
      case 'disconnected': return '未連接';
      case 'checking': return '檢查中';
      default: return '錯誤';
    }
  };

  const getSearchStatusColor = () => {
    switch (searchStatus) {
      case 'connected': return 'success';
      case 'disconnected': return 'error';
      case 'checking': return 'info';
      default: return 'warning';
    }
  };

  const getSearchStatusText = () => {
    switch (searchStatus) {
      case 'connected': return '已連接';
      case 'disconnected': return '未連接';
      case 'checking': return '檢查中';
      default: return '錯誤';
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold', mb: 4 }}>
        🖥️ 系統資訊與狀態
      </Typography>

      <Grid container spacing={3}>
        {/* API 狀態卡片 */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: `${getApiStatusColor()}.main`, mr: 2 }}>
                  <Api />
                </Avatar>
                <Box>
                  <Typography variant="h6">後端 API 狀態</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {API_BASE_URL}
                  </Typography>
                </Box>
              </Box>

              <Alert
                severity={getApiStatusColor()}
                sx={{ mb: 2 }}
                action={
                  <Button
                    size="small"
                    onClick={checkSystemStatus}
                    disabled={loading}
                  >
                    重新檢查
                  </Button>
                }
              >
                {getApiStatusText()}
              </Alert>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                <Button
                  variant="outlined"
                  onClick={() => window.open(`${API_BASE_URL.replace('/api', '/docs')}`, '_blank')}
                >
                  API 文檔
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => window.open(API_BASE_URL.replace('/api', ''), '_blank')}
                >
                  測試連接
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* 搜尋狀態卡片 */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: `${getSearchStatusColor()}.main`, mr: 2 }}>
                  <Web />
                </Avatar>
                <Box>
                  <Typography variant="h6">網路搜尋狀態</Typography>
                  <Typography variant="body2" color="text.secondary">
                    DuckDuckGo 搜尋引擎
                  </Typography>
                </Box>
              </Box>

              <Alert
                severity={getSearchStatusColor()}
                sx={{ mb: 2 }}
                action={
                  <Button
                    size="small"
                    onClick={checkSearchStatus}
                    disabled={loading}
                  >
                    重新檢查
                  </Button>
                }
              >
                {getSearchStatusText()}
              </Alert>

              {searchTestResult && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="caption" color="text.secondary">
                    搜尋測試結果:
                  </Typography>
                  <Typography variant="body2">
                    {searchTestResult.status === 'ok' ? '✅ 搜尋功能正常' : '❌ 搜尋功能異常'}
                  </Typography>
                  {searchTestResult.duckduckgo_available !== undefined && (
                    <Typography variant="body2">
                      DuckDuckGo: {searchTestResult.duckduckgo_available ? '✅ 可用' : '❌ 不可用'}
                    </Typography>
                  )}
                </Box>
              )}

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                <Button
                  variant="outlined"
                  color="info"
                  onClick={checkSearchStatus}
                >
                  測試搜尋
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* 系統資訊卡片 */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                系統資訊
              </Typography>
              {loading ? (
                <Box sx={{ textAlign: 'center', py: 3 }}>
                  <CircularProgress />
                </Box>
              ) : systemInfo ? (
                <Box>
                  <Typography variant="body2" paragraph>
                    <strong>版本:</strong> {systemInfo.version || '1.0.0'}
                  </Typography>
                  <Typography variant="body2" paragraph>
                    <strong>狀態:</strong> {systemInfo.status || '運行中'}
                  </Typography>
                  <Typography variant="body2" paragraph>
                    <strong>服務:</strong> {systemInfo.service || '智能問答系統'}
                  </Typography>
                  {systemInfo.ollama && (
                    <>
                      <Typography variant="body2" paragraph>
                        <strong>Ollama 狀態:</strong> {systemInfo.ollama.available ? '✅ 已連接' : '❌ 未連接'}
                      </Typography>
                      <Typography variant="body2" paragraph>
                        <strong>AI 模型:</strong> {systemInfo.ollama.preferred_model || '未設定'}
                      </Typography>
                    </>
                  )}
                  {systemInfo.search && (
                    <Typography variant="body2" paragraph>
                      <strong>搜尋引擎:</strong> {systemInfo.search.duckduckgo_available ? '✅ DuckDuckGo' : '❌ 未連接'}
                    </Typography>
                  )}
                  <Typography variant="body2">
                    <strong>更新時間:</strong> {systemInfo.timestamp ? new Date(systemInfo.timestamp).toLocaleString() : '未知'}
                  </Typography>
                </Box>
              ) : (
                <Alert severity="warning">
                  無法獲取系統資訊
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* 功能模塊卡片 */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🚀 系統功能模塊
              </Typography>
              <Grid container spacing={2}>
                {[
                  { name: 'AI 智能問答', status: 'active', icon: <QuestionAnswer /> },
                  { name: '網路資訊分析', status: searchStatus === 'connected' ? 'active' : 'inactive', icon: <Web /> },
                  { name: '綜合 AI 分析', status: apiStatus === 'connected' && searchStatus === 'connected' ? 'active' : 'inactive', icon: <Merge /> },
                  { name: 'Ollama AI 服務', status: apiStatus === 'connected' ? 'active' : 'inactive', icon: <SmartToy /> },
                  { name: 'DuckDuckGo 搜尋', status: searchStatus === 'connected' ? 'active' : 'inactive', icon: <Web /> },
                  { name: '文件管理', status: 'inactive', icon: <Folder /> },
                  { name: '知識庫管理', status: 'inactive', icon: <Storage /> },
                ].map((module, idx) => (
                  <Grid item xs={12} sm={6} md={4} key={idx}>
                    <Paper sx={{ p: 2, display: 'flex', alignItems: 'center' }}>
                      <Avatar sx={{ bgcolor: module.status === 'active' ? 'success.main' : 'grey.400', mr: 2 }}>
                        {module.icon}
                      </Avatar>
                      <Box>
                        <Typography variant="body1">{module.name}</Typography>
                        <Typography variant="caption" color={module.status === 'active' ? 'success.main' : 'text.secondary'}>
                          {module.status === 'active' ? '已啟用' : '未啟用'}
                        </Typography>
                      </Box>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

// 主應用組件
function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [mobileOpen, setMobileOpen] = useState(false);
  const [systemStatus, setSystemStatus] = useState('loading');

  useEffect(() => {
    checkSystemStatus();
  }, []);

  const checkSystemStatus = async () => {
    try {
      await fetch(`${API_BASE_URL}/system/health`);
      setSystemStatus('connected');
    } catch (error) {
      setSystemStatus('disconnected');
    }
  };

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return (
          <Box>
            <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold', mb: 4 }}>
              🎯 AI 智能問答系統
            </Typography>

            <Alert severity={systemStatus === 'connected' ? 'success' : 'warning'} sx={{ mb: 3 }}>
              {systemStatus === 'connected'
                ? '✅ 後端 API 連接正常，可以開始使用 AI 問答功能'
                : '⚠️  後端 API 未連接，請確保後端服務正在運行'}
            </Alert>

            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Card sx={{ height: '100%', textAlign: 'center', p: 3 }}>
                  <QuestionAnswer sx={{ fontSize: 60, color: 'secondary.main', mb: 2 }} />
                  <Typography variant="h5" gutterBottom>AI 智能問答</Typography>
                  <Typography variant="body2" paragraph>
                    使用 AI 模型回答各種問題
                  </Typography>
                  <Button
                    variant="contained"
                    color="secondary"
                    onClick={() => setActiveTab('rag')}
                  >
                    開始使用
                  </Button>
                </Card>
              </Grid>

              <Grid item xs={12} md={4}>
                <Card sx={{ height: '100%', textAlign: 'center', p: 3 }}>
                  <Web sx={{ fontSize: 60, color: 'info.main', mb: 2 }} />
                  <Typography variant="h5" gutterBottom>網路資訊分析</Typography>
                  <Typography variant="body2" paragraph>
                    使用 DuckDuckGo 搜尋網路資訊並由 AI 分析
                  </Typography>
                  <Button
                    variant="contained"
                    color="info"
                    onClick={() => setActiveTab('web')}
                  >
                    開始使用
                  </Button>
                </Card>
              </Grid>

              <Grid item xs={12} md={4}>
                <Card sx={{ height: '100%', textAlign: 'center', p: 3 }}>
                  <Merge sx={{ fontSize: 60, color: 'warning.main', mb: 2 }} />
                  <Typography variant="h5" gutterBottom>綜合 AI 分析</Typography>
                  <Typography variant="body2" paragraph>
                    結合內部知識庫和網路搜尋的綜合回答
                  </Typography>
                  <Button
                    variant="contained"
                    color="warning"
                    onClick={() => setActiveTab('hybrid')}
                  >
                    開始使用
                  </Button>
                </Card>
              </Grid>
            </Grid>

            <Box sx={{ mt: 4 }}>
              <Typography variant="h5" gutterBottom>📋 使用指南</Typography>
              <Paper sx={{ p: 3 }}>
                <ol>
                  <li style={{ marginBottom: '10px' }}>選擇問答模式：AI 智能問答、網路資訊分析或綜合分析</li>
                  <li style={{ marginBottom: '10px' }}>在輸入框中輸入您的問題</li>
                  <li style={{ marginBottom: '10px' }}>點擊「發送問題」按鈕</li>
                  <li>查看 AI 生成的回答和相關資訊</li>
                </ol>
                <Alert severity="info" sx={{ mt: 2 }}>
                  💡 提示：
                  <ul style={{ marginTop: '8px', marginLeft: '20px' }}>
                    <li>確保 Ollama 服務正在運行以獲得最佳 AI 回答體驗</li>
                    <li>網路資訊分析需要後端能連接到 DuckDuckGo 搜尋引擎</li>
                    <li>綜合分析會結合內部知識庫和網路搜尋結果</li>
                  </ul>
                </Alert>
              </Paper>
            </Box>
          </Box>
        );
      case 'rag':
      case 'web':
      case 'hybrid':
        return <QAPanel key={activeTab} type={activeTab} />;
      case 'info':
        return <SystemInfoPanel />;
      case 'documents':
        return (
          <Box>
            <Typography variant="h4" gutterBottom>
              文件管理
            </Typography>
            <Alert severity="info">
              此功能正在開發中，敬請期待。
            </Alert>
          </Box>
        );
      case 'knowledge':
        return (
          <Box>
            <Typography variant="h4" gutterBottom>
              知識庫管理
            </Typography>
            <Alert severity="info">
              此功能正在開發中，敬請期待。
            </Alert>
          </Box>
        );
      case 'api':
        return (
          <Box>
            <Typography variant="h4" gutterBottom>
              API 測試
            </Typography>
            <Alert severity="info">
              此功能正在開發中，敬請期待。
            </Alert>
          </Box>
        );
      default:
        return (
          <Box>
            <Typography variant="h4" gutterBottom>
              功能開發中
            </Typography>
            <Typography>
              請從左側選單選擇可用功能。
            </Typography>
          </Box>
        );
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />

      <AppBar position="static" elevation={1}>
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
            <SmartToy sx={{ mr: 2 }} />
            <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
              AI 智能問答系統
            </Typography>
            <Chip
              label="Ollama 驅動"
              size="small"
              color="success"
              sx={{ ml: 2 }}
            />
          </Box>
          <Typography variant="caption" sx={{ display: { xs: 'none', sm: 'block' } }}>
            API 狀態: {systemStatus === 'connected' ? '✅ 已連接' : '⚠️  未連接'}
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 4, mb: 6 }}>
        <Grid container spacing={3}>
          {/* 側邊欄 - 桌面 */}
          <Grid item md={3} sx={{ display: { xs: 'none', md: 'block' } }}>
            <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
          </Grid>

          {/* 側邊欄 - 移動 */}
          <Drawer
            variant="temporary"
            open={mobileOpen}
            onClose={handleDrawerToggle}
            sx={{
              display: { xs: 'block', md: 'none' },
              '& .MuiDrawer-paper': { width: 280 },
            }}
          >
            <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
          </Drawer>

          {/* 主內容區 */}
          <Grid item xs={12} md={9}>
            <Paper sx={{
              p: 4,
              minHeight: '80vh',
              borderRadius: 3,
              boxShadow: 2,
              position: 'relative'
            }}>
              {renderContent()}
            </Paper>

            {/* 頁腳 */}
            <Box sx={{ mt: 4, textAlign: 'center', color: 'text.secondary' }}>
              <Typography variant="body2">
                © 2024 AI 智能問答系統 | FastAPI + React + Ollama + DuckDuckGo
              </Typography>
              <Typography variant="caption" display="block">
                版本 1.0.0 | 支援 AI 問答與網路搜尋
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Container>
    </ThemeProvider>
  );
}

export default App;
