import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  Grid,
  CircularProgress,
  Alert,
  CssBaseline,
  ThemeProvider,
  createTheme
} from '@mui/material';
import MainMenu from './components/MainMenu';
import DocumentStatus from './components/DocumentStatus';
import SystemInfo from './components/SystemInfo';
import QAPanel from './components/QAPanel';
import { useSystem } from './hooks/useSystem';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  const [activeTab, setActiveTab] = useState('home');
  const { systemInfo, loading, error, refreshSystemInfo } = useSystem();

  const renderContent = () => {
    switch (activeTab) {
      case 'home':
        return <SystemInfo systemInfo={systemInfo} />;
      case 'status':
        return <DocumentStatus />;
      case 'qa':
        return <QAPanel type="rag" />;
      case 'web':
        return <QAPanel type="web" />;
      case 'hybrid':
        return <QAPanel type="hybrid" />;
      default:
        return <SystemInfo systemInfo={systemInfo} />;
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg">
        <Box sx={{ my: 4 }}>
          <Typography variant="h3" component="h1" gutterBottom align="center" color="primary">
            🤖 智能 RAG 問答系統
          </Typography>
          
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              連接伺服器失敗：{error.message}
            </Alert>
          )}

          <Grid container spacing={3}>
            <Grid item xs={12} md={3}>
              <Paper sx={{ p: 2 }}>
                <MainMenu activeTab={activeTab} onTabChange={setActiveTab} />
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={9}>
              <Paper sx={{ p: 3, minHeight: '70vh' }}>
                {renderContent()}
              </Paper>
            </Grid>
          </Grid>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;