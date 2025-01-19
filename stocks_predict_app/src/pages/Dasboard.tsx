/* eslint-disable @typescript-eslint/no-explicit-any */
import {
    Typography,
    Box,
    Button,
    Card,
    CardActions,
    CardContent,
    Skeleton,
} from '@mui/material'
import StockGraph from '../components/StockGraph'
import { IStock } from '../interfaces/IStock'
import { useEffect, useState } from 'react'
import sp500image from '../assets/sp500.png'

interface IDashboardProps {
    stockData: IStock[]
}

const Dashboard = (props: IDashboardProps) => {
    const mainStocks = ['ABNB', 'PEP', 'MSFT', 'NVDA']
    const [selectedStock, setSelectedStock] = useState<IStock | null>(null)
    const [loading, setLoading] = useState<boolean>(true)

    useEffect(() => {
        if (props.stockData.length > 0) {
            setLoading(false)
            if (!selectedStock) {
                const firstStock =
                    props.stockData.find((s) => s.Symbol === mainStocks[0]) ??
                    null
                setSelectedStock(firstStock)
            }
        } else {
            setLoading(true)
        }
    }, [props.stockData])

    const handleStockClick = (stock: IStock) => {
        setSelectedStock(stock)
    }

    return (
        <Box>
            {loading && (
                <Box display="flex" gap={1}>
                    {[1, 2, 3, 4].map((i) => (
                        <Skeleton
                            animation="wave"
                            key={i}
                            variant="rectangular"
                            width={300}
                            height={200}
                        />
                    ))}
                </Box>
            )}
            {!loading && (
                <Box display="flex" gap={1}>
                    {props.stockData
                        .filter((s) => mainStocks.some((ms) => ms === s.Symbol))
                        .map((stock: any) => (
                            <Box key={stock.Company}>
                                <Card sx={{ minWidth: 300 }}>
                                    <CardContent>
                                        <Box display="flex">
                                            <Typography variant="h6" flex={1}>
                                                {stock.Company}
                                            </Typography>
                                            <img
                                                src={`https://logo.clearbit.com/${stock.Domain}`}
                                                alt={`${stock.Company} Logo`}
                                                style={{
                                                    width: 30,
                                                    height: 30,
                                                }}
                                                onError={(e) =>
                                                    (e.currentTarget.src =
                                                        sp500image)
                                                }
                                            />
                                        </Box>
                                        <Typography variant="body1">
                                            {stock.Symbol}
                                        </Typography>

                                        <Typography variant="h4">
                                            ${stock.value.toFixed(3)}
                                        </Typography>
                                        {stock.lastValue > 0 ? (
                                            <Typography
                                                variant="subtitle2"
                                                color="success"
                                            >
                                                $ {stock.lastValue.toFixed(3)}
                                            </Typography>
                                        ) : (
                                            <Typography
                                                variant="subtitle2"
                                                color="error"
                                            >
                                                $ {stock.lastValue.toFixed(3)}
                                            </Typography>
                                        )}
                                    </CardContent>
                                    <CardActions>
                                        <Button
                                            size="small"
                                            variant="contained"
                                            onClick={() =>
                                                handleStockClick(stock)
                                            }
                                        >
                                            View
                                        </Button>
                                    </CardActions>
                                </Card>
                            </Box>
                        ))}
                </Box>
            )}
            {loading && (
                <Skeleton
                    style={{ marginTop: '16px' }}
                    animation="wave"
                    variant="rectangular"
                    height={500}
                />
            )}
            {!loading && selectedStock && (
                <StockGraph
                    stock={selectedStock}
                    listStocks={props.stockData}
                />
            )}
        </Box>
    )
}

export default Dashboard
