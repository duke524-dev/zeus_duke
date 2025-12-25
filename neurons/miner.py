# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Eric (Ørpheus A.I.)
# Copyright © 2025 Ørpheus A.I.
# Copyright © 2025 Zeus Duke

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

import time
import torch
import typing
import pickle
import asyncio
import hashlib
import json
from pathlib import Path
from collections import OrderedDict
import bittensor as bt

import openmeteo_requests

import numpy as np
from zeus.data.converter import get_converter
from zeus.utils.config import get_device_str
from zeus.utils.time import to_timestamp
from zeus.protocol import TimePredictionSynapse
from zeus.base.miner import BaseMinerNeuron
from zeus import __version__ as zeus_version
from zeus.validator.constants import ERA5_DATA_VARS


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior.
    In particular, you should replace the forward function with your own logic.

    Currently the base miner does a request to OpenMeteo (https://open-meteo.com/) for predictions.
    You are encouraged to attempt to improve over this by changing the forward function.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        bt.logging.info("Attaching forward functions to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        
        # Setup device and OpenMeteo API
        self.device: torch.device = torch.device(get_device_str())
        self.openmeteo_api = openmeteo_requests.Client()
        
        # Response caching for API calls (LRU cache with TTL)
        self._cache: OrderedDict = OrderedDict()
        self._cache_ttl = 300  # 5 minutes cache
        self._cache_max_size = 100  # Maximum cache entries
        
        # Load trained models
        self.models = {}
        model_dir = Path("trained_models")
        
        if model_dir.exists():
            for variable in ERA5_DATA_VARS.keys():
                model_files = list(model_dir.glob(f"model_{variable}_*.pkl"))
                if model_files:
                    # Get most recent model by timestamp
                    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                    try:
                        with open(latest_model, 'rb') as f:
                            self.models[variable] = pickle.load(f)
                        bt.logging.info(f"Loaded model for {variable}: {latest_model.name}")
                    except Exception as e:
                        bt.logging.error(f"Failed to load model for {variable}: {e}")
                else:
                    bt.logging.warning(f"No model found for {variable}")
        else:
            bt.logging.warning(f"Model directory {model_dir} does not exist")
        
        # Pre-warm models with dummy input to eliminate cold start
        self._prewarm_models()
        
        # List of additional OpenMeteo models to fetch
        self.additional_models = ["gem_seamless", "gfs_seamless", "jma_seamless", "meteofrance"]
    
    def _prewarm_models(self):
        """Pre-warm models with dummy input to eliminate cold start latency."""
        for variable, model in self.models.items():
            try:
                if hasattr(model, 'forward'):
                    # PyTorch model - warm up with dummy input
                    dummy_input = torch.zeros(1, 4, dtype=torch.float32).to(self.device)
                    model.eval()
                    with torch.no_grad():
                        _ = model(dummy_input)
                    bt.logging.debug(f"Pre-warmed PyTorch model for {variable}")
                elif hasattr(model, 'predict'):
                    # Scikit-learn model - warm up with dummy input
                    dummy_input = np.zeros((1, 4), dtype=np.float32)
                    _ = model.predict(dummy_input)
                    bt.logging.debug(f"Pre-warmed scikit-learn model for {variable}")
            except Exception as e:
                bt.logging.debug(f"Could not pre-warm model for {variable}: {e}")
    
    def _get_cache_key(self, model_name: str, base_params: dict) -> str:
        """Generate cache key from request parameters."""
        key_data = {
            'model': model_name,
            'lat': tuple(base_params['latitude']),  # Convert to tuple for hashability
            'lon': tuple(base_params['longitude']),
            'hourly': base_params['hourly'],
            'start': base_params['start_hour'],
            'end': base_params['end_hour']
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> typing.Optional[torch.Tensor]:
        """Get forecast from cache if available and not expired."""
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                # Move to end (most recently used)
                self._cache.move_to_end(cache_key)
                return cached_data
            else:
                # Expired, remove from cache
                del self._cache[cache_key]
        return None
    
    def _add_to_cache(self, cache_key: str, forecast: torch.Tensor):
        """Add forecast to cache with LRU eviction."""
        # Remove oldest entry if cache is full
        if len(self._cache) >= self._cache_max_size:
            self._cache.popitem(last=False)  # Remove oldest (first) item
        
        self._cache[cache_key] = (forecast, time.time())

    async def forward(self, synapse: TimePredictionSynapse) -> TimePredictionSynapse:
        """
        Processes the incoming TimePredictionSynapse for a prediction.

        Args:
            synapse (TimePredictionSynapse): The synapse object containing the time range and coordinates

        Returns:
            TimePredictionSynapse: The synapse object with the 'predictions' field set".
        """
        # Start timing
        process_start_time = time.time()
        timing_breakdown = {}
        
        # Log incoming synapse details
        coordinates = torch.Tensor(synapse.locations)
        start_time = to_timestamp(synapse.start_time)
        end_time = to_timestamp(synapse.end_time)
        
        bt.logging.info("=" * 80)
        bt.logging.info("INCOMING SYNAPSE:")
        bt.logging.info(f"  Variable: {synapse.variable}")
        bt.logging.info(f"  Requested Hours: {synapse.requested_hours}")
        bt.logging.info(f"  Start Time: {start_time} (timestamp: {synapse.start_time})")
        bt.logging.info(f"  End Time: {end_time} (timestamp: {synapse.end_time})")
        bt.logging.info(f"  Locations Grid Shape: {coordinates.shape}")
        bt.logging.info(f"  Number of Locations: {len(synapse.locations)}")
        bt.logging.info(f"  Validator Version: {synapse.version}")
        bt.logging.info(f"  Predictions (incoming): {len(synapse.predictions) if synapse.predictions else 0} elements")
        if len(synapse.locations) > 0:
            bt.logging.info(f"  First Location: {synapse.locations[0]}")
            bt.logging.info(f"  Last Location: {synapse.locations[-1]}")
        bt.logging.info("=" * 80)
        
        # Timing: Initial setup
        timing_breakdown['initial_setup'] = time.time() - process_start_time

        ##########################################################################################################
        # Step 1: Fetch forecasts from 4 OpenMeteo models
        latitudes, longitudes = coordinates.view(-1, 2).T
        converter = get_converter(synapse.variable)
        
        # Base parameters for OpenMeteo API
        base_params = {
            "latitude": latitudes.tolist(),
            "longitude": longitudes.tolist(),
            "hourly": converter.om_name,
            "start_hour": start_time.isoformat(timespec="minutes"),
            "end_hour": end_time.isoformat(timespec="minutes"),
        }
        
        # Fetch forecasts from 4 models in parallel
        async def fetch_model_forecast(model_name: str):
            """Fetch forecast for a single model asynchronously with caching."""
            model_params = base_params.copy()
            model_params["models"] = model_name
            
            # Check cache first
            cache_key = self._get_cache_key(model_name, base_params)
            cached_forecast = self._get_from_cache(cache_key)
            if cached_forecast is not None:
                bt.logging.debug(f"Cache hit for {model_name}")
                return (model_name, cached_forecast, None)
            
            try:
                # Run synchronous API call in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                model_responses = await loop.run_in_executor(
                    None,
                    lambda: self.openmeteo_api.weather_api(
                        "https://api.open-meteo.com/v1/forecast",
                        params=model_params,
                        method="POST"
                    )
                )
                # Process response asynchronously
                model_forecast = await loop.run_in_executor(
                    None,
                    self._process_openmeteo_responses,
                    model_responses, synapse.requested_hours, coordinates.shape, converter
                )
                
                # Store in cache
                self._add_to_cache(cache_key, model_forecast)
                
                return (model_name, model_forecast, None)  # (name, forecast, error)
            except Exception as e:
                bt.logging.warning(f"Failed to fetch forecast from {model_name}: {e}")
                return (model_name, None, e)  # (name, None, error)
        
        # Execute all 4 API calls in parallel
        api_start_time = time.time()
        bt.logging.debug("Fetching forecasts from 4 models in parallel...")
        results = await asyncio.gather(*[
            fetch_model_forecast(model_name) 
            for model_name in self.additional_models
        ])
        timing_breakdown['api_calls'] = time.time() - api_start_time
        
        # Process results and handle errors
        model_forecasts = []
        first_successful_forecast = None
        successful_models = []
        failed_models = []
        
        for model_name, forecast, error in results:
            if forecast is not None:
                model_forecasts.append(forecast)
                successful_models.append(model_name)
                if first_successful_forecast is None:
                    first_successful_forecast = forecast
            else:
                failed_models.append(model_name)
                # Use first successful forecast as fallback
                if first_successful_forecast is not None:
                    model_forecasts.append(first_successful_forecast)
                    bt.logging.debug(f"Using fallback forecast for failed model: {model_name}")
        
        # Validate we have at least one successful forecast
        if first_successful_forecast is None:
            bt.logging.error(f"All model fetches failed: {failed_models}")
            raise RuntimeError(f"Failed to fetch any model forecasts for {synapse.variable}")
        
        if failed_models:
            bt.logging.warning(f"Some models failed ({failed_models}), using fallback for {len(failed_models)} forecasts")
        
        bt.logging.debug(f"Successfully fetched {len(successful_models)}/{len(self.additional_models)} models: {successful_models}")
        
        # Step 2: Stack all forecasts as input features [time, lat, lon, num_models]
        # Stack: 4 models = 4 forecasts total
        data_processing_start = time.time()
        all_forecasts = torch.stack(model_forecasts, dim=-1)
        # Shape: [requested_hours, lat_grid, lon_grid, 4]
        timing_breakdown['data_processing'] = time.time() - data_processing_start
        
        # Step 3: Pass to trained model if available
        model_inference_start = time.time()
        if synapse.variable in self.models:
            bt.logging.debug(f"Using trained model for {synapse.variable}")
            try:
                output = self._predict_with_model(
                    model=self.models[synapse.variable],
                    forecasts=all_forecasts,
                    variable=synapse.variable,
                    requested_hours=synapse.requested_hours
                )
            except Exception as e:
                bt.logging.error(f"Model prediction failed: {e}. Falling back to first successful forecast.")
                if first_successful_forecast is not None:
                    output = first_successful_forecast
                else:
                    raise RuntimeError(f"Model prediction failed and no fallback forecast available: {e}")
        else:
            bt.logging.warning(f"No trained model for {synapse.variable}. Using first successful forecast.")
            if first_successful_forecast is not None:
                output = first_successful_forecast
            else:
                raise RuntimeError(f"No trained model available and no forecasts fetched for {synapse.variable}")
        timing_breakdown['model_inference'] = time.time() - model_inference_start
        ##########################################################################################################
        bt.logging.debug(f"Output shape is {output.shape}")

        output_preparation_start = time.time()
        synapse.predictions = output.tolist()
        synapse.version = zeus_version
        timing_breakdown['output_preparation'] = time.time() - output_preparation_start
        
        # Log outgoing synapse details
        # Calculate total processing time
        process_end_time = time.time()
        total_processing_time = process_end_time - process_start_time
        
        bt.logging.info("=" * 80)
        bt.logging.info("OUTGOING SYNAPSE:")
        bt.logging.info(f"  Variable: {synapse.variable}")
        bt.logging.info(f"  Requested Hours: {synapse.requested_hours}")
        bt.logging.info(f"  Predictions Shape: {output.shape}")
        bt.logging.info(f"  Predictions List Length: {len(synapse.predictions)}")
        if len(synapse.predictions) > 0:
            bt.logging.info(f"  First Prediction Shape: {len(synapse.predictions[0]) if isinstance(synapse.predictions[0], list) else 'scalar'}")
            if len(synapse.predictions) > 0 and len(synapse.predictions[0]) > 0:
                first_pred = synapse.predictions[0][0]
                if isinstance(first_pred, list):
                    bt.logging.info(f"  First Prediction Value Shape: {len(first_pred)}")
                    bt.logging.info(f"  First Prediction Value Range: [{min(first_pred):.4f}, {max(first_pred):.4f}]")
                else:
                    bt.logging.info(f"  First Prediction Value: {first_pred:.4f}")
        bt.logging.info(f"  Miner Version: {synapse.version}")
        bt.logging.info("-" * 80)
        bt.logging.info("⏱️  TIMING BREAKDOWN:")
        bt.logging.info(f"  Initial Setup:        {timing_breakdown.get('initial_setup', 0):.4f}s ({timing_breakdown.get('initial_setup', 0)*1000:.2f}ms)")
        bt.logging.info(f"  API Calls (parallel):  {timing_breakdown.get('api_calls', 0):.4f}s ({timing_breakdown.get('api_calls', 0)*1000:.2f}ms)")
        bt.logging.info(f"  Data Processing:       {timing_breakdown.get('data_processing', 0):.4f}s ({timing_breakdown.get('data_processing', 0)*1000:.2f}ms)")
        bt.logging.info(f"  Model Inference:       {timing_breakdown.get('model_inference', 0):.4f}s ({timing_breakdown.get('model_inference', 0)*1000:.2f}ms)")
        bt.logging.info(f"  Output Preparation:    {timing_breakdown.get('output_preparation', 0):.4f}s ({timing_breakdown.get('output_preparation', 0)*1000:.2f}ms)")
        bt.logging.info("-" * 80)
        bt.logging.info(f"⏱️  TOTAL PROCESSING TIME: {total_processing_time:.4f} seconds ({total_processing_time*1000:.2f} ms)")
        
        # Calculate percentage breakdown
        if total_processing_time > 0:
            bt.logging.info("  Time Distribution:")
            for stage, duration in timing_breakdown.items():
                percentage = (duration / total_processing_time) * 100
                bt.logging.info(f"    {stage.replace('_', ' ').title()}: {percentage:.1f}%")
        
        bt.logging.info("=" * 80)
        
        return synapse
    

    async def blacklist(self, synapse: TimePredictionSynapse) -> typing.Tuple[bool, str]:
        return await self._blacklist(synapse)
    
    async def priority(self, synapse: TimePredictionSynapse) -> float:
        return await self._priority(synapse)
    
    def _process_openmeteo_responses(
        self, 
        responses, 
        requested_hours: int, 
        coordinates_shape: torch.Size,
        converter
    ) -> torch.Tensor:
        """
        Process OpenMeteo API responses into a tensor (optimized).
        
        Args:
            responses: OpenMeteo API response objects
            requested_hours: Number of hours to predict
            coordinates_shape: Shape of coordinates tensor [lat, lon, 2]
            converter: Variable converter for unit conversion
            
        Returns:
            torch.Tensor: Processed forecast with shape [time, lat, lon]
        """
        # Pre-allocate output tensor for better performance
        lat_dim, lon_dim = coordinates_shape[:2]
        
        # Extract data directly into numpy arrays, then convert to torch once
        data_list = []
        for r in responses:
            var_data = []
            for i in range(r.Hourly().VariablesLength()):
                var_data.append(r.Hourly().Variables(i).ValuesAsNumpy())
            data_list.append(np.stack(var_data, axis=-1))
        
        # Stack all responses and convert to torch in one operation
        stacked_data = np.stack(data_list, axis=1)
        output = torch.from_numpy(stacked_data).reshape(requested_hours, lat_dim, lon_dim, -1)
        
        # [time, lat, lon] in case of single variable output
        output = output.squeeze(dim=-1)
        
        # Convert variable(s) to ERA5 units, combines variables for windspeed
        output = converter.om_to_era5(output)
        return output
    
    def _predict_with_model(
        self,
        model,
        forecasts: torch.Tensor,
        variable: str,
        requested_hours: int
    ) -> torch.Tensor:
        """
        Generate predictions using trained model.
        
        Args:
            model: Trained model (scikit-learn or PyTorch)
            forecasts: Stacked forecasts from multiple models
                      Shape: [time, lat, lon, num_models]
            variable: Variable name
            requested_hours: Number of hours to predict
            
        Returns:
            torch.Tensor: Model predictions with shape [time, lat, lon]
        """
        # Flatten spatial dimensions for model input
        # Shape: [time * lat * lon, num_models]
        time_dim, lat_dim, lon_dim, num_models = forecasts.shape
        
        # Check model type and predict
        if hasattr(model, 'predict'):
            # Scikit-learn style model
            # Convert to numpy in one operation
            forecasts_flat = forecasts.view(-1, num_models).cpu().numpy()
            predictions_flat = model.predict(forecasts_flat)
            # Convert back to torch and reshape
            predictions = torch.from_numpy(predictions_flat).reshape(time_dim, lat_dim, lon_dim)
        elif hasattr(model, 'forward'):
            # PyTorch model - keep on GPU if available
            model.eval()
            with torch.no_grad():
                # Keep tensor on device, avoid CPU round-trip
                input_tensor = forecasts.view(-1, num_models).to(self.device)
                predictions_flat = model(input_tensor)
                
                if isinstance(predictions_flat, torch.Tensor):
                    # Reshape directly without CPU transfer until needed
                    predictions = predictions_flat.view(time_dim, lat_dim, lon_dim)
                    # Only move to CPU at the end
                    if predictions.device != torch.device('cpu'):
                        predictions = predictions.cpu()
                else:
                    predictions = torch.tensor(
                        predictions_flat, 
                        dtype=torch.float32
                    ).reshape(time_dim, lat_dim, lon_dim)
        else:
            raise ValueError(f"Unknown model type: {type(model)}. Model must have 'predict' or 'forward' method.")
        
        return predictions
    
    

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running | uid {miner.uid} | {time.time()}")
            time.sleep(30)
