PRAGMA journal_mode = WAL;

-- metadata table for simple key/value app state
CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- keep updated_at fresh on updates
CREATE TRIGGER IF NOT EXISTS trg_metadata_updated_at
AFTER UPDATE ON metadata
FOR EACH ROW BEGIN
  UPDATE metadata
  SET updated_at = CURRENT_TIMESTAMP
  WHERE key = NEW.key;
END;

CREATE TABLE IF NOT EXISTS ticker (
  ticker TEXT PRIMARY KEY,
  last REAL,
  prev_close REAL,
  volume INTEGER,
  bid_price REAL,
  ask_price REAL,
  timestamp TEXT
);

CREATE INDEX IF NOT EXISTS idx_ticker_volume ON ticker(volume);
CREATE INDEX IF NOT EXISTS idx_ticker_timestamp ON ticker(timestamp);

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
       subsector_id INTEGER,
       ticker TEXT,
       date TEXT,
       close REAL,
       volume INTEGER,
       UNIQUE (ticker, date)
);

CREATE TABLE IF NOT EXISTS stock_criteria (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       subsector_id INTEGER,
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
