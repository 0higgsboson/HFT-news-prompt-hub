#!/bin/bash

# Generate stock news using PromptManager
# Usage: ./generate_stock_news.sh [TICKER] [SENTIMENT_SCORE] [MODEL]
# Defaults: TICKER=NVDA, SENTIMENT_SCORE=4, MODEL=claude-3-5-sonnet-20241022

# Set default values
TICKER=${1:-NVDA}
SENTIMENT_SCORE=${2:-4}
MODEL=${3:-claude-3-5-sonnet-20241022}

echo "Generating $TICKER news with sentiment score $SENTIMENT_SCORE using $MODEL..."

python3 ../PromptManager.py run \
    --prompt newsGenerator \
    --model $MODEL \
    --vars ticker=$TICKER sentiment_score=$SENTIMENT_SCORE \
    --output ../output/${TICKER}_news_sentiment_${SENTIMENT_SCORE}_${MODEL##*-}.json

echo "News generated and saved to ../output/${TICKER}_news_sentiment_${SENTIMENT_SCORE}_${MODEL##*-}.json"