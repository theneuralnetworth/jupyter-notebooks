/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState, useEffect } from 'react'
import {
    Box,
    Card,
    CardContent,
    useTheme,
    Select,
    FormControl,
    InputLabel,
    MenuItem,
    ListItemText,
    ListItemIcon,
} from '@mui/material'
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
import sp500image from '../assets/sp500.png'

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement)

interface IStockGraphProps {
    listStocks: IStock[]
    stock: IStock | null
}

const StockGraph = (props: IStockGraphProps) => {
    const [stockData, setStockData] = useState<number[]>([])
    const [selectedStock, setSelectedStock] = useState<IStock | null>(
        props.stock
    )
    const [selectedStockName, setSelectedStockName] = useState<string | null>(
        props?.stock?.Symbol ?? null
    )
    const [labels, setLabels] = useState<string[]>([])

    const theme = useTheme()

    useEffect(() => {
        if (selectedStock) {
            console.log(selectedStock)
            setLabels(
                selectedStock.dates.map((entry, i) =>
                    i === selectedStock.dates.length - 2
                        ? 'Today'
                        : i === selectedStock.dates.length - 1
                          ? 'Tomorrow'
                          : `${entry.date.getFullYear()}-${entry.date.getMonth() + 1}-${entry.date.getDate()}`
                )
            )
            setStockData(selectedStock.dates.map((entry) => entry.value))
        }
    }, [selectedStock])

    useEffect(() => {
        if (props.stock) {
            setSelectedStock(props.stock)
            setSelectedStockName(props.stock.Symbol)
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
                <Box display="flex" alignItems="center">
                    <FormControl style={{width:250}}>
                        <InputLabel id="simple-select-label">Stock</InputLabel>
                        <Select
                            labelId="simple-select-label"
                            id="simple-select"
                            value={selectedStockName}
                            label="Stock"
                            size="small"
                            onChange={(e) => {
                                setSelectedStockName(e.target.value as string)
                                setSelectedStock(
                                    props.listStocks.find(
                                        (stock) =>
                                            stock.Symbol === e.target.value
                                    ) ?? null
                                )
                            }}
                        >
                            {props.listStocks.map((stock) => (
                                <MenuItem  value={stock.Symbol}>
                                    <Box display='flex' alignItems='center'>
                                    <ListItemIcon>
                                        <img
                                            src={`https://logo.clearbit.com/${stock.Domain}`}
                                            alt={`${stock.Company} Logo`}
                                            style={{
                                                width: 20,
                                                height: 20,
                                            }}
                                            onError={(e) =>
                                                (e.currentTarget.src =
                                                    sp500image)
                                            }
                                        />
                                    </ListItemIcon>
                                    <ListItemText primary={stock.Company} />
                                    </Box>
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </Box>
                <Box width={900} height={'100%'}>
                    <Line data={chartData} options={chartOptions} />
                </Box>
            </CardContent>
        </Card>
    )
}

export default StockGraph
