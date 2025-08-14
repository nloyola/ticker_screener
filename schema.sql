PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sector (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       sector TEXT,
       created_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
       UNIQUE(sector)
);

CREATE INDEX IF NOT EXISTS idx_sector_secto ON sector(sector);

CREATE TABLE IF NOT EXISTS subsector (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       sector_id INTEGER,
       subsector TEXT,
       tickers TEXT,
       created_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
       FOREIGN KEY (sector_id) REFERENCES sector(id)
           ON UPDATE CASCADE
           ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS price_data (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       ticker TEXT,
       date TEXT,
       close REAL,
       volume INTEGER,
       UNIQUE (ticker, date)
);

CREATE TABLE IF NOT EXISTS stock_criteria (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
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
       date_created DATE NOT NULL DEFAULT (date('now')),
       UNIQUE (ticker)
);
