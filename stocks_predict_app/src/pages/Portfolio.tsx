/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useState } from 'react'
import { useForm, Controller } from 'react-hook-form'
import {
    TextField,
    Button,
    Typography,
    Box,
    Card,
    CardContent,
    Checkbox,
    FormControlLabel,
    FormControl,
    FormLabel,
    Grid,
} from '@mui/material'
import Lottie from 'react-lottie'
import animation_portfolio_1 from '../lotties/Portfolio_1.json'
import PortfolioGraph from '../components/PortfolioGraph'
import portfolio_empty from '../assets/portofio_empty_state.png'

interface IPortfolioProps {
    stockData: any[]
}

const Portfolio = (props: IPortfolioProps) => {
    const setSectors = new Set(props.stockData.map((s) => s['GICS Sector']))
    const [loading, setLoading] = useState(false)
    const listSectors = [...setSectors]
    const [porfolio, setPortfolio] = React.useState<any[]>([])
    const { handleSubmit, control } = useForm({
        defaultValues: {
            returns:
            props.stockData.map(s=> s.percentageValue),
            sectors: listSectors,
            n_assets: props.stockData.length,
            excluded_sectors: listSectors,
            risk_aversion: 2,
            min_return: 0.05,
            max_weight: 30,
            min_weight: 15,
            weight_penalty: 1000,
            return_penalty: 1000,
            sector_penalty: 10000,
            optimal_fitness_result: 1000,
            no_improvement_generations: 1000,
            generations: 1000,
            crossover_rate: 0,
            mutation_rate: 0.4,
        },
    })
    console.log('listSectors:', listSectors)
    const onSubmit = async (data: any) => {
        try {
            console.log('submiting...', data)
            setLoading(true)

            const response = await fetch(
                'http://127.0.0.1:8000/optimize-portfolio',
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        ...data,
                        max_weight: data.max_weight / 100,
                        min_weight: data.min_weight / 100,
                        returns: data.returns.map(Number),
                        sectors: listSectors.map((s: any) => s.trim()),
                        excluded_sectors: listSectors
                            .filter((s) => !data.sectors.includes(s))
                            .map((s: any) => s.trim()),
                    }),
                }
            )
            if (response.ok) {
                const result = await response.json()
                setPortfolio(result.optimized_portfolio_weights)
                console.log('Success:', result)
            } else {
                console.error('Error:', response.statusText)
            }

            setLoading(false)
        } catch (error) {
            console.error('Submission error:', error)
        }
    }

    return (
        <Box display="flex" flexDirection="column" gap={2}>
            <Box>
                <Typography variant="h5">Portfolio Optimization</Typography>
                <Typography variant="body2">
                    Optimize your portfolio by setting the parameters below.
                </Typography>
            </Box>
            <Box display="flex" gap={2}>
                <Box display="flex" flex={1}>
                    <Card style={{ width: '100%', height: 600 }}>
                        <CardContent>
                            <Box display="flex" flex={1}>
                                <form
                                    onSubmit={handleSubmit(onSubmit)}
                                    style={{ width: '100%' }}
                                >
                                    <Box display="flex" gap={2}>
                                        <Controller
                                            name="risk_aversion"
                                            control={control}
                                            render={({ field }) => (
                                                <TextField
                                                    {...field}
                                                    label="Risk Aversion"
                                                    type="number"
                                                    fullWidth
                                                    margin="normal"
                                                />
                                            )}
                                        />
                                        <Controller
                                            name="min_return"
                                            control={control}
                                            render={({ field }) => (
                                                <TextField
                                                    {...field}
                                                    label="Min Return"
                                                    type="number"
                                                    fullWidth
                                                    margin="normal"
                                                />
                                            )}
                                        />
                                    </Box>
                                    <Box display="flex" gap={2}>
                                        <Controller
                                            name="min_weight"
                                            control={control}
                                            render={({ field }) => (
                                                <TextField
                                                    {...field}
                                                    label="Min Weight (%)"
                                                    type="number"
                                                    fullWidth
                                                    margin="normal"
                                                />
                                            )}
                                        />
                                        <Controller
                                            name="max_weight"
                                            control={control}
                                            render={({ field }) => (
                                                <TextField
                                                    {...field}
                                                    label="Max Weight (%)"
                                                    type="number"
                                                    fullWidth
                                                    margin="normal"
                                                />
                                            )}
                                        />
                                    </Box>
                                    <Box>
                                        <Controller
                                            name="excluded_sectors"
                                            control={control}
                                            defaultValue={[]}
                                            render={({
                                                field: { onChange, value },
                                            }) => (
                                                <FormControl>
                                                    <FormLabel component="legend">
                                                        Sectors
                                                    </FormLabel>
                                                    <Grid container spacing={2}>
                                                        {listSectors.map(
                                                            (sector) => (
                                                                <Grid
                                                                    item
                                                                    xs={6}
                                                                    key={sector}
                                                                >
                                                                    <FormControlLabel
                                                                        control={
                                                                            <Checkbox
                                                                                checked={(
                                                                                    value as string[]
                                                                                ).includes(
                                                                                    sector
                                                                                )}
                                                                                onChange={(
                                                                                    e
                                                                                ) => {
                                                                                    const newValue =
                                                                                        e
                                                                                            .target
                                                                                            .checked
                                                                                            ? [
                                                                                                  ...value,
                                                                                                  sector,
                                                                                              ]
                                                                                            : value.filter(
                                                                                                  (
                                                                                                      s
                                                                                                  ) =>
                                                                                                      s !==
                                                                                                      sector
                                                                                              )
                                                                                    onChange(
                                                                                        newValue
                                                                                    )
                                                                                }}
                                                                            />
                                                                        }
                                                                        label={
                                                                            sector
                                                                        }
                                                                    />
                                                                </Grid>
                                                            )
                                                        )}
                                                    </Grid>
                                                </FormControl>
                                            )}
                                        />
                                    </Box>
                                    <Button
                                        type="submit"
                                        variant="contained"
                                        color="primary"
                                        style={{ marginTop: '20px' }}
                                    >
                                        Submit
                                    </Button>
                                </form>
                            </Box>
                        </CardContent>
                    </Card>
                </Box>

                <Box>
                    {!loading && porfolio.length > 0 ? (
                        <PortfolioGraph
                            listStocks={porfolio
                                .filter((ps) => ps !== 0)
                                .map((s, i) => ({
                                    stock: props.stockData[i],
                                    weight: s.toFixed(3) * 100,
                                }))}
                        />
                    ) : (
                        <Card style={{ width: 600, height: 600 }}>
                            <CardContent>
                                <Box
                                    display="flex"
                                    flexDirection="column"
                                    gap={2}
                                    height={600}
                                >
                                    <Typography variant="h5" color="primary">
                                        Portfolio
                                    </Typography>
                                    {loading ? (
                                        <Box
                                            display="flex"
                                            flexDirection="column"
                                            alignItems="center"
                                            justifyContent="center"
                                        >
                                            <Lottie
                                                options={{
                                                    loop: true,
                                                    autoplay: true,
                                                    animationData:
                                                        animation_portfolio_1,
                                                }}
                                                height={450}
                                                width={450}
                                            />
                                            <Typography variant="h6">
                                                Generating your portfolio
                                            </Typography>
                                        </Box>
                                    ) : (
                                        <Box
                                            display="flex"
                                            flexDirection="column"
                                            alignItems="center"
                                            justifyContent="center"
                                        >

                                        <Typography variant="h5">
                                            Build your portfolio
                                        </Typography>
                                        <img
                                            width={450}
                                            height={450}
                                            src={portfolio_empty}
                                            alt="Empty Portfolio"
                                        />
                                        </Box>
                                    )}
                                </Box>
                            </CardContent>
                        </Card>
                    )}
                </Box>
            </Box>
        </Box>
    )
}

export default Portfolio
