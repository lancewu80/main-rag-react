import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Paper,
  CircularProgress,
  Chip,
  Divider,
  IconButton,
  Stack,
  Alert
} from '@mui/material';
import {
  Send,
  ContentCopy,
  Download,
  History
} from '@mui/icons-material';
import { askQuestion } from '../services/api';

function QAPanel({ type }) {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const getTitle = () => {
    switch (type) {
      case 'rag': return '📚 本地文件問答';
      case 'web': return '🌐 網路搜尋問答';
      case 'hybrid': return '🔀 智能混合問答';
      default: return '問答';
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    try {
      const response = await askQuestion(type, question);
      setAnswer(response);

      // 添加到歷史記錄
      setHistory(prev => [{
        question,
        answer: response.answer,
        timestamp: new Date().toLocaleTimeString(),
        type
      }, ...prev.slice(0, 4)]);

      setQuestion('');
    } catch (error) {
      console.error('問答失敗:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        {getTitle()}
      </Typography>

      <form onSubmit={handleSubmit}>
        <TextField
          fullWidth
          multiline
          rows={3}
          variant="outlined"
          placeholder="輸入您的問題..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
          sx={{ mb: 2 }}
        />

        <Box sx={{ display: 'flex', gap: 1, mb: 3 }}>
          <Button
            type="submit"
            variant="contained"
            disabled={loading || !question.trim()}
            startIcon={loading ? <CircularProgress size={20} /> : <Send />}
          >
            {loading ? '思考中...' : '發送問題'}
          </Button>

          <Button
            variant="outlined"
            onClick={() => setQuestion('')}
            disabled={loading}
          >
            清空
          </Button>
        </Box>
      </form>

      {answer && (
        <Paper sx={{ p: 3, mt: 3, bgcolor: 'background.default' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6">回答</Typography>
            <Box>
              <IconButton size="small" title="複製回答">
                <ContentCopy />
              </IconButton>
              <IconButton size="small" title="下載">
                <Download />
              </IconButton>
            </Box>
          </Box>

          <Divider sx={{ mb: 2 }} />

          <Typography paragraph>
            {answer.answer}
          </Typography>

          {answer.sources && answer.sources.length > 0 && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                參考來源:
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                {answer.sources.map((source, idx) => (
                  <Chip
                    key={idx}
                    label={`來源 ${idx + 1}`}
                    size="small"
                    variant="outlined"
                  />
                ))}
              </Stack>
            </Box>
          )}

          {answer.metadata && (
            <Alert severity="info" sx={{ mt: 2 }}>
              使用策略: {answer.metadata.strategy} |
              相關度: {answer.metadata.relevance_score}%
            </Alert>
          )}
        </Paper>
      )}

      {history.length > 0 && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="subtitle1" gutterBottom>
            <History fontSize="small" sx={{ mr: 1 }} />
            最近問答
          </Typography>
          <Stack spacing={1}>
            {history.map((item, idx) => (
              <Paper key={idx} sx={{ p: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  [{item.timestamp}] {item.type === 'rag' ? '📚' : '🌐'} {item.question}
                </Typography>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  {item.answer.substring(0, 100)}...
                </Typography>
              </Paper>
            ))}
          </Stack>
        </Box>
      )}
    </Box>
  );
}

export default QAPanel;
