import React from 'react'
import AppBar from '@mui/material/AppBar'
import Toolbar from '@mui/material/Toolbar'
import Typography from '@mui/material/Typography'
import Box from '@mui/material/Box'
import { Brightness2, LightMode } from '@mui/icons-material'
import { IconButton } from '@mui/material'

interface IHeaderProps {
    darkMode: boolean
    setDarkMode: (darkMode: boolean) => void
}

const Header: React.FC<IHeaderProps> = (props) => {
    return (
        <AppBar
            position="fixed"
            color="primary"
            sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}
        >
            <Toolbar>
                <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                    Stocks Predict App
                </Typography>
                <Box>
                    <IconButton
                        onClick={() => props.setDarkMode(!props.darkMode)}
                    >
                        {props.darkMode ? <LightMode /> : <Brightness2 />}
                    </IconButton>
                </Box>
            </Toolbar>
        </AppBar>
    )
}

export default Header
