import React from 'react'
import { Link } from 'react-router-dom'
import Drawer from '@mui/material/Drawer'
import List from '@mui/material/List'
import ListItem from '@mui/material/ListItem'
import ListItemIcon from '@mui/material/ListItemIcon'
import ListItemText from '@mui/material/ListItemText'
import DashboardIcon from '@mui/icons-material/Dashboard'
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet'
import Divider from '@mui/material/Divider'
import Toolbar from '@mui/material/Toolbar'
import { ListItemButton, useTheme } from '@mui/material'

const drawerWidth = 240

const Sidebar: React.FC = () => {
    const theme = useTheme()
    return (
        <Drawer
            variant="permanent"
            sx={{
                width: drawerWidth,
                flexShrink: 0,
                [`& .MuiDrawer-paper`]: {
                    width: drawerWidth,
                    boxSizing: 'border-box',
                },
            }}
        >
            <Toolbar />
            <Divider />
            <List>
                {[
                    {
                        text: 'Dashboard',
                        icon: <DashboardIcon />,
                        link: '/',
                    },
                    {
                        text: 'Portfolio',
                        icon: <AccountBalanceWalletIcon />,
                        link: '/portfolio',
                    },
                ].map((item) => (
                    <ListItem component={Link} to={item.link} key={item.text} style={{padding:0}}>
                        <ListItemButton style={{margin:0}}>
                        <ListItemIcon>{item.icon}</ListItemIcon>

                            <ListItemText primary={item.text} primaryTypographyProps={{ style: { color: theme.palette.text.primary } }} />
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
        </Drawer>
    )
}

export default Sidebar
