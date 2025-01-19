export interface IStockDate {
    date: Date;
    value: number;
}

export interface IStock {
    Symbol: string;
    Company: string;
    name: Date;
    value: number;
    lastValue: number;
    percentageValue: number;
    dates: IStockDate[]
}

