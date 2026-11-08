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

-- date stays TEXT, a YYYY-MM-DD string: declared DATE, the dashboard's Go driver would scan it as a time.
-- The CHECK accepts exactly the real calendar dates in that shape, since date() always returns YYYY-MM-DD
-- or NULL. The upgrade in app/db.py adds it to databases created before it; change both together.
-- high and low are the day's, adjusted on the close's basis; NULL in rows stored before them, until the next fetch
-- replaces those. The upgrade in app/db.py adds them to databases created before them; change both together.
CREATE TABLE IF NOT EXISTS price_data (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       subsector_id INTEGER,
       ticker TEXT,
       date TEXT NOT NULL CHECK (date(date) IS date),
       close REAL,
       volume INTEGER,
       high REAL,
       low REAL,
       UNIQUE (ticker, date)
);

-- volume_ratio is the day's volume over avg_vol_20d. The upgrade in app/db.py drops the columns of the core rules
-- the breakout strategy removed (max_vol_10d, ema_10, dma_50_25d_ago, price_gt_10ema) from databases created
-- before it, and adds volume_ratio, after date_created as ALTER TABLE places it; change both together.
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
       dma_50 REAL,
       dma_50_prev_15d REAL,
       rsi REAL,
       macd REAL,
       signal REAL,
       breakout_20d INTEGER,
       breakout_60d INTEGER,
       breakout_confirmed INTEGER,
       price_gt_50dma INTEGER,
       dma_50_rising INTEGER,
       rsi_50_75 INTEGER,
       macd_bullish INTEGER,
       date_created DATE NOT NULL DEFAULT (date('now')),
       volume_ratio REAL,
       UNIQUE (ticker)
);

-- Each ticker's company name, looked up from the price API named in price_api. Not market data: db-init keeps it,
-- so names survive a re-initialisation. The dashboard writes it too, when an admin adds a ticker. The upgrade in
-- app/db.py creates it in databases from before it; change both together.
CREATE TABLE IF NOT EXISTS company_name (
       ticker TEXT NOT NULL PRIMARY KEY COLLATE NOCASE,
       name TEXT NOT NULL,
       price_api TEXT NOT NULL CHECK (price_api IN ('tiingo', 'eodhd')),
       stored_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
);
