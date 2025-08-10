PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sector (
       sector TEXT,
       subsector TEXT,
       tickers TEXT,
       created_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS price_data (
       ticker TEXT,
       date TEXT,
       close REAL,
       volume INTEGER,
       PRIMARY KEY (Ticker, Date)
);

 CREATE TABLE stock_criteria (
        ticker TEXT NOT NULL,
        price REAL,
        core_criteria_met BOOLEAN,
        core_criteria_score INTEGER,
        high_20d REAL,
        high_60d REAL,
        volume INTEGER,
        avg_vol_20d INTEGER,
        max_vol_10d INTEGER,
        dma_50 REAL,
        ema_10 REAL,
        dma_50_prev_15d REAL,
        dma_50_25d_ago REAL,
        rsi REAL,
        macd REAL,
        signal REAL,
        breakout_20d INTEGER,
        breakout_60d INTEGER,
        breakout_confirmed INTEGER,
        price_gt_50dma INTEGER,
        price_gt_10ema INTEGER,
        dma_50_rising INTEGER,
        rsi_50_75 INTEGER,
        macd_bullish INTEGER,
        PRIMARY KEY (ticker)
);
