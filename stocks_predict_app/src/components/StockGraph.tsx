/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState, useEffect } from 'react'
import { Box, Typography, Card, CardContent, useTheme } from '@mui/material'
import { Line } from 'react-chartjs-2'
import {
    Chart as ChartJS,
    LineElement,
    CategoryScale,
    LinearScale,
    PointElement,
    ChartOptions,
} from 'chart.js'
import { IStock } from '../interfaces/IStock'

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement)

interface IStockGraphProps {
    stock: IStock | null
}

const StockGraph = (props: IStockGraphProps) => {
    const [stockData, setStockData] = useState<number[]>([])
    const [labels, setLabels] = useState<string[]>([])

    const theme = useTheme()

    useEffect(() => {
        if (props.stock) {
            console.log(props.stock)
            setLabels(
                props.stock.dates.map(
                    (entry, i) => i === props.stock.dates.length - 2 ? 'Today' :
                    i === props.stock.dates.length - 1 ? 'Tomorrow' :
                        `${entry.date.getFullYear()}-${entry.date.getMonth() + 1}-${entry.date.getDate()}`
                )
            )
            setStockData(props.stock.dates.map((entry) => entry.value))
        }
    }, [props.stock])

    const chartData = {
        labels: labels,
        datasets: [
            {
                label: 'Stock Value',
                data: stockData,
                segment: {
                    borderColor: (ctx: any) => {
                        const index = ctx.p1DataIndex
                        const label = ctx.chart.data.labels[index]
                        const lastLabel =
                            ctx.chart.data.labels[
                                ctx.chart.data.labels.length - 1
                            ]
                        if (label === lastLabel) {
                            return theme.palette.secondary.main
                        }
                        return theme.palette.primary.main
                    },
                },
                backgroundColor: 'rgba(63, 81, 181, 0.1)',
                pointBorderColor: '#3f51b5',
                pointBackgroundColor: '#ffffff',
                fill: true,
                tension: 0.4,
            },
        ],
    }

    const chartOptions: ChartOptions<'line'> = {
        responsive: true,
        plugins: {
            legend: {
                display: true,
                position: 'top',
            },
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Date',
                },
            },
            y: {
                title: {
                    display: true,
                    text: 'Stock Value',
                },
            },
        },
    }

    return (
        <Card sx={{ mt: 3 }}>
            <CardContent>
                <Typography variant="h6" gutterBottom>
                    Stock Performance {props.stock?.Company}
                </Typography>
                <Box width={900} height={'100%'}>
                    <Line data={chartData} options={chartOptions} />
                </Box>
            </CardContent>
        </Card>
    )
}

export default StockGraph
