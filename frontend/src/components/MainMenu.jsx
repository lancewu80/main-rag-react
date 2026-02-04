import React from 'react';
import {
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Divider,
  Chip
} from '@mui/material';
import {
  Home,
  Storage,
  QuestionAnswer,
  Search,
  Merge,
  Settings,
  Info
} from '@mui/icons-material';

const menuItems = [
  { id: 'home', label: '系統概覽', icon: <Home /> },
  { id: 'status', label: '文件狀態', icon: <Info /> },
  { id: 'build', label: '重建知識庫', icon: <Storage />, action: 'build' },
  { id: 'qa', label: '文件問答', icon: <QuestionAnswer />, chip: 'RAG' },
  { id: 'web', label: '網路搜尋問答', icon: <Search /> },
  { id: 'hybrid', label: '混合問答', icon: <Merge />, chip: '智能' },
  { id: 'settings', label: '系統設定', icon: <Settings /> }
];

function MainMenu({ activeTab, onTabChange }) {
  const handleClick = (item) => {
    if (item.action) {
      // 處理動作
      console.log(`執行動作: ${item.action}`);
    } else {
      onTabChange(item.id);
    }
  };

  return (
    <List>
      {menuItems.map((item) => (
        <React.Fragment key={item.id}>
          <ListItem disablePadding>
            <ListItemButton
              selected={activeTab === item.id}
              onClick={() => handleClick(item)}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.label} />
              {item.chip && (
                <Chip 
                  label={item.chip} 
                  size="small" 
                  color="primary" 
                  variant="outlined"
                />
              )}
            </ListItemButton>
          </ListItem>
          <Divider />
        </React.Fragment>
      ))}
    </List>
  );
}

export default MainMenu;