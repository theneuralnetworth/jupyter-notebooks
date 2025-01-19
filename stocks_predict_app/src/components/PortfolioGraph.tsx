/* eslint-disable @typescript-eslint/no-explicit-any */
import { Box, Card, CardContent, Typography } from '@mui/material'
import { Pie } from 'react-chartjs-2'
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js'
import { IStock } from '../interfaces/IStock'

ChartJS.register(ArcElement, Tooltip, Legend)

interface IStockPortfolio {
    stock: IStock
    weight: number
}

interface IStockGraphProps {
    listStocks: IStockPortfolio[] | []
}

const generateColors = (numColors: number) => {
    const colors = [
        '#3f51b5', // Indigo
        '#f44336', // Red
        '#4caf50', // Green
        '#ff9800', // Orange
        '#2196f3', // Blue
        '#9c27b0', // Purple
        '#00bcd4', // Cyan
        '#8bc34a', // Light Green
        '#ff5722', // Deep Orange
        '#673ab7', // Deep Purple
        '#ffc107', // Amber
        '#03a9f4', // Light Blue
    ]
    return colors.slice(0, numColors)
}

const PortfolioGraph = (props: IStockGraphProps) => {
    const backgroundColors = generateColors(props.listStocks.length)
    const chartData = {
        labels: props.listStocks.map((entry: any) => entry.stock.Company),
        datasets: [
            {
                label: 'Stock Value',
                data: props.listStocks.map((entry: any) => entry.weight),
                backgroundColor: backgroundColors,
                pointBorderColor: '#3f51b5',
                pointBackgroundColor: '#ffffff',
                fill: true,
                tension: 0.4,
            },
        ],
    }

    return (
        <Card style={{ width: 600, height: 600 }}>
            <CardContent>
                <Box>
                    <Typography variant="h5" color="primary">
                        Portfolio
                    </Typography>
                    <Pie
                        data={chartData}
                        options={{
                            plugins: {
                                legend: {
                                    position: 'right', // Move labels to the side
                                    labels: {
                                        usePointStyle: true, // Use points instead of squares in the legend
                                        boxWidth: 10, // Adjust the size of the legend box
                                    },
                                },
                            }
                        }}
                    />
                </Box>
            </CardContent>
        </Card>
    )
}

export default PortfolioGraph
