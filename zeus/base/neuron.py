# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import copy
import typing
import time

import bittensor as bt

from abc import ABC, abstractmethod

# Sync calls set weights and also resyncs the metagraph.
from zeus.utils.config import check_config, add_args, config
from zeus.utils.misc import ttl_get_block
from zeus import __spec_version__ as spec_version

class BaseNeuron(ABC):
    """
    Base class for Bittensor miners. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """

    neuron_type: str = "BaseNeuron"

    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"
    spec_version: int = spec_version

    @property
    def block(self):
        return ttl_get_block(self)

    def __init__(self, config=None):
        base_config = copy.deepcopy(config or BaseNeuron.config())
        self.config = self.config()
        self.config.merge(base_config)
        self.check_config(self.config)

        # Set up logging with the provided configuration.
        bt.logging.set_config(config=self.config.logging)

        # If a gpu is required, set the device to cuda:N (e.g. cuda:0)
        self.device = self.config.neuron.device

        # Log the configuration for reference.
        bt.logging.info(self.config)

        # Build Bittensor objects
        # These are core Bittensor classes to interact with the network.
        bt.logging.info("Setting up bittensor objects.")

        # The wallet holds the cryptographic key pairs for the miner.
        if self.config.mock:
            self.wallet = bt.MockWallet(config=self.config)
            # self.subtensor = MockSubtensor(self.config.netuid, wallet=self.wallet)
            # self.metagraph = MockMetagraph(self.config.netuid, subtensor=self.subtensor)
        else:
            self.wallet = bt.Wallet(config=self.config)
            
            # Retry logic for subtensor connection with exponential backoff
            max_retries = 5
            base_delay = 2  # Start with 2 seconds
            subtensor_connected = False
            
            for attempt in range(max_retries):
                try:
                    bt.logging.info(f"Connecting to subtensor (attempt {attempt + 1}/{max_retries})...")
                    self.subtensor = bt.Subtensor(config=self.config)
                    # Test the connection by accessing a property
                    _ = self.subtensor.network
                    bt.logging.success("Successfully connected to subtensor")
                    subtensor_connected = True
                    break
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    # Check if it's a timeout or connection error
                    is_timeout = (
                        "timeout" in error_msg.lower() or 
                        "timed out" in error_msg.lower() or
                        isinstance(e, (TimeoutError, ConnectionError, OSError))
                    )
                    
                    if attempt < max_retries - 1 and is_timeout:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff: 2, 4, 8, 16, 32 seconds
                        bt.logging.warning(
                            f"Failed to connect to subtensor (attempt {attempt + 1}/{max_retries}): {error_type}: {error_msg}"
                        )
                        bt.logging.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    elif attempt < max_retries - 1:
                        # Non-timeout error, retry once more
                        delay = base_delay
                        bt.logging.warning(
                            f"Connection error (attempt {attempt + 1}/{max_retries}): {error_type}: {error_msg}"
                        )
                        bt.logging.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        bt.logging.error(f"Failed to connect to subtensor after {max_retries} attempts: {error_type}: {error_msg}")
                        raise
            
            if not subtensor_connected:
                raise RuntimeError("Failed to establish subtensor connection after all retry attempts")
            
            # Retry logic for metagraph creation with exponential backoff
            metagraph_loaded = False
            for attempt in range(max_retries):
                try:
                    bt.logging.info(f"Loading metagraph (attempt {attempt + 1}/{max_retries})...")
                    self.metagraph = self.subtensor.metagraph(self.config.netuid)
                    # Verify metagraph loaded successfully
                    _ = self.metagraph.n
                    bt.logging.success("Successfully loaded metagraph")
                    metagraph_loaded = True
                    break
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    is_timeout = (
                        "timeout" in error_msg.lower() or 
                        "timed out" in error_msg.lower() or
                        isinstance(e, (TimeoutError, ConnectionError, OSError))
                    )
                    
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        bt.logging.warning(
                            f"Failed to load metagraph (attempt {attempt + 1}/{max_retries}): {error_type}: {error_msg}"
                        )
                        bt.logging.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        bt.logging.error(f"Failed to load metagraph after {max_retries} attempts: {error_type}: {error_msg}")
                        raise
            
            if not metagraph_loaded:
                raise RuntimeError("Failed to load metagraph after all retry attempts")

        bt.logging.info(f"Wallet: {self.wallet}")
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Check if the miner is registered on the Bittensor network before proceeding further.
        self.check_registered()

        # Each miner gets a unique identity (UID) in the network for differentiation.
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running neuron on subnet: {self.config.netuid} with uid {self.uid} using network: {self.subtensor.chain_endpoint}"
        )
        self.step = 0

    @abstractmethod
    def run(self): ...

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        Includes retry logic for network operations.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                self.check_registered()
                break
            except (TimeoutError, ConnectionError, Exception) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    bt.logging.warning(f"Failed to check registration (attempt {attempt + 1}/{max_retries}): {e}")
                    bt.logging.debug(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    bt.logging.error(f"Failed to check registration after {max_retries} attempts: {e}")
                    raise

        if self.should_sync_metagraph():
            self.resync_metagraph()

        if self.should_set_weights():
            self.set_weights()

        # Always save state.
        self.save_state()

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    def should_set_weights(self) -> bool:
        # Don't set weights on initialization.
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False

        # Define appropriate logic for when set weights.
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length and self.neuron_type != "MinerNeuron"  # don't set weights if you're a miner

    def save_state(self):
        bt.logging.trace(
            "save_state() not implemented for this neuron. You can implement this function to save model checkpoints or other useful data."
        )

    def load_state(self):
        bt.logging.trace(
            "load_state() not implemented for this neuron. You can implement this function to load model checkpoints or other useful data."
        )
