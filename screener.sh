#!/bin/bash

cd /opt/www-data/investing/ticker_screener
#uv run rules.py rule-runner --json output.json > /dev/null 2>&1
/home/nelson/.local/bin/uv run rules.py rule-runner --json output.json
