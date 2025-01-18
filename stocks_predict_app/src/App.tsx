/* eslint-disable @typescript-eslint/no-explicit-any */
// src/App.tsx
import React, { useEffect, useState } from 'react'
import { ThemeProvider } from '@mui/material/styles'
import { CssBaseline, Box } from '@mui/material'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import { darkTheme, lightTheme } from './theme'
import { Route, BrowserRouter as Router, Routes } from 'react-router-dom'
import Dashboard from './pages/Dasboard'
import { IStock } from './interfaces/IStock'
import Portfolio from './pages/Portfolio'
import Papa from 'papaparse'

const App: React.FC = () => {
    const [darkMode, setDarkMode] = useState(false)
    const [stockData, setStockData] = useState<IStock[]>([])
    const theme = darkMode ? darkTheme : lightTheme

    // Fetch stocks for the last 10 days
    const fetchStockData = async (stocks: any[]) => {
        const csvFilePath = '/assets/stocks.csv'
        Papa.parse(csvFilePath, {
            download: true,
            header: true,
            skipEmptyLines: true,
            complete: async (result) => {
                const listStocks: IStock[] = []
                //const mainStocks = ['AAPL', 'MSFT', 'AMZN', 'NVDA']
                for (const stock of stocks) {
                    const stockdates: any[] = result.data
                        .filter((entry: any) => entry.Symbol === stock.Symbol)
                        .sort(
                            (a: any, b: any) =>
                                new Date(a.Date).getTime() -
                                new Date(b.Date).getTime()
                        )
                        .slice(-30)
                    console.log(stockdates)
                    const lastValue =  stockdates.length > 0
                    ? stockdates[stockdates.length - 1].Close -
                      stockdates[stockdates.length - 1].Open
                    : 0
                    const element = {
                        ...stock,
                        dates: stockdates.map((entry: any) => ({
                            date: new Date(entry.Date),
                            value: entry.Close,
                        })),
                        value: Number(
                            stockdates.length > 0
                                ? stockdates[stockdates.length - 1].Close
                                : 0
                        ),
                        lastValue:lastValue,
                        percentageValue: stockdates.length > 0 ? lastValue / stockdates[stockdates.length - 1].Open : 0,
                    }
                    listStocks.push(element)
                }
                console.log(listStocks.find((s) => s.Symbol === 'AAPL'))
                setStockData(listStocks)
            },
            error: (error) => {
                console.error('Error reading CSV file:', error)
            },
        })
    }

    // Fetch top stocks
    useEffect(() => {
        const fetchTopStocks = async () => {
            let stocks: any[] = []

            // Caminho do arquivo na pasta 'public/assets'
            const csvFilePath = '/assets/sp500.csv'

            // Configurar o PapaParse para processar o CSV
            Papa.parse(csvFilePath, {
                download: true,
                header: true, // Converte as linhas em objetos usando o cabeçalho
                skipEmptyLines: true,
                complete: async (result) => {
                    stocks = result.data // Atualiza o estado com os dados

                    if (stocks.length > 0) {
                        await fetchStockData(stocks)
                    }
                },
                error: (error) => {
                    console.error('Error reading CSV file:', error)
                },
            })
        }

        fetchTopStocks()
    }, [])

    return (
        <Router>
            <ThemeProvider theme={theme}>
                <CssBaseline />
                <Header darkMode={darkMode} setDarkMode={setDarkMode} />
                <Sidebar />
                <Box
                    component="main"
                    sx={{
                        flexGrow: 1,
                        p: 3,
                        marginLeft: '240px', // Width of the sidebar
                        marginTop: '64px', // Height of the header
                        backgroundColor: theme.palette.background.default,
                        minHeight: '100vh',
                    }}
                >
                    <Routes>
                        <Route
                            path="/"
                            element={<Dashboard stockData={stockData} />}
                        />
                        <Route
                            path="/portfolio"
                            element={<Portfolio stockData={stockData} />}
                        />
                    </Routes>
                </Box>
            </ThemeProvider>
        </Router>
    )
}

export default App
