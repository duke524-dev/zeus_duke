#!/bin/bash

# Ensure Node 20 is used for PM2
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use 20 > /dev/null 2>&1

set -a
source miner.env
set +a

MINER_PROCESS_NAME="zeus_miner"


if pm2 list | grep -q "$MINER_PROCESS_NAME"; then
  echo "Process '$MINER_PROCESS_NAME' is already running. Deleting it..."
  pm2 delete $MINER_PROCESS_NAME
fi

# Get the absolute path to venv Python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"

pm2 start $VENV_PYTHON --name $MINER_PROCESS_NAME -- \
  neurons/miner.py \
  --netuid $NETUID \
  --subtensor.network $SUBTENSOR_NETWORK \
  --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --axon.port $AXON_PORT \
  --blacklist.force_validator_permit $BLACKLIST_FORCE_VALIDATOR_PERMIT \
  --logging.info

# synchronise the process list with the pm2 ecosystem file
pm2 save