# Zeus Miner Data Flow

Complete data flow diagram showing how the miner processes requests from initialization to response.

---

## 🚀 1. INITIALIZATION PHASE

```
start_miner.sh
    ↓
PM2 Process Manager (Node 20)
    ↓
Python Process (venv/bin/python3)
    ↓
neurons/miner.py: __main__
    ↓
Miner.__init__()
```

### Initialization Steps:

```
1. BaseNeuron.__init__() [zeus/base/neuron.py:60]
   ├─ Load config
   ├─ Setup logging
   ├─ Create Wallet (cryptographic keys)
   ├─ Create Subtensor (blockchain connection)
   ├─ Create Metagraph (network state)
   ├─ Check registration
   └─ Get UID from network

2. BaseMinerNeuron.__init__() [zeus/base/miner.py:44]
   ├─ Create Axon (request handler)
   └─ Setup threading locks

3. Miner.__init__() [neurons/miner.py:54]
   ├─ Attach forward/blacklist/priority functions to axon
   ├─ Setup device (CPU/GPU)
   ├─ Initialize OpenMeteo API client
   ├─ Initialize LRU cache (100 entries, 5min TTL)
   ├─ Load trained models from trained_models/
   │   └─ For each variable: model_{variable}_*.pkl
   ├─ Pre-warm models (eliminate cold start)
   │   ├─ PyTorch: dummy forward pass
   │   └─ Scikit-learn: dummy predict
   └─ Define additional_models list
       └─ ["gem_seamless", "gfs_seamless", "jma_seamless", "meteofrance"]
```

### Background Thread Startup:

```
Miner.__enter__() [zeus/base/miner.py:162]
    ↓
run_in_background_thread()
    ↓
BaseMinerNeuron.run() [zeus/base/miner.py:70]
    ├─ sync() - Initial network sync
    ├─ axon.serve() - Register on network
    ├─ axon.start() - Start listening
    └─ Main loop:
        ├─ Wait for epoch_length blocks
        ├─ Periodically sync() metagraph
        └─ Log "Miner running | uid X | timestamp" every 30s
```

---

## 📥 2. REQUEST RECEIVED PHASE

```
Validator sends request
    ↓
Bittensor Network (WebSocket)
    ↓
Axon receives TimePredictionSynapse
    ↓
Blacklist Check [zeus/base/miner.py:193]
    ├─ Check if hotkey is registered
    ├─ Check if validator permit required
    ├─ Check minimum stake
    └─ Return (blacklisted: bool, reason: str)
    ↓
Priority Check [zeus/base/miner.py:260]
    └─ Return priority score (based on stake)
    ↓
Forward Function [neurons/miner.py:156]
```

---

## 🔄 3. REQUEST PROCESSING PHASE

### 3.1 Input Parsing

```
TimePredictionSynapse {
    variable: str              # e.g., "2m_temperature"
    locations: List[List[Tuple[float, float]]]  # Grid of (lat, lon)
    start_time: float          # Unix timestamp
    end_time: float             # Unix timestamp
    requested_hours: int        # e.g., 24
    predictions: []            # Empty, to be filled
    version: str                # Validator version
}
    ↓
Extract data:
    ├─ coordinates = torch.Tensor(synapse.locations)
    │   └─ Shape: [lat_grid, lon_grid, 2]
    ├─ latitudes, longitudes = coordinates.view(-1, 2).T
    │   └─ Shape: [num_locations] each
    ├─ start_time = to_timestamp(synapse.start_time)
    ├─ end_time = to_timestamp(synapse.end_time)
    └─ converter = get_converter(synapse.variable)
        └─ Handles unit conversion (OpenMeteo → ERA5)
```

### 3.2 Build API Parameters

```
base_params = {
    "latitude": [lat1, lat2, ...],
    "longitude": [lon1, lon2, ...],
    "hourly": converter.om_name,  # e.g., "temperature_2m"
    "start_hour": start_time.isoformat(),
    "end_hour": end_time.isoformat()
}
```

### 3.3 Fetch Forecasts (Parallel + Caching)

```
For each model in ["gem_seamless", "gfs_seamless", "jma_seamless", "meteofrance"]:
    
    async fetch_model_forecast(model_name):
        1. Generate cache key
           └─ MD5 hash of (model, coords, time, variable)
        
        2. Check cache
           ├─ Cache hit? → Return cached forecast (<1ms)
           └─ Cache miss? → Continue to API call
        
        3. API Call (if cache miss)
           ├─ Run in thread pool (non-blocking)
           ├─ POST to https://api.open-meteo.com/v1/forecast
           ├─ params = base_params + {"models": model_name}
           └─ Process response in executor
        
        4. Process Response
           ├─ Extract hourly data from API response
           ├─ Convert to tensor: [time, lat, lon]
           ├─ Convert units: OpenMeteo → ERA5 (via converter)
           └─ Shape: [requested_hours, lat_grid, lon_grid]
        
        5. Store in cache
           └─ LRU eviction if cache full (100 entries)
        
        6. Return (model_name, forecast, error)

Execute all 4 in parallel:
    results = await asyncio.gather(*[
        fetch_model_forecast("gem_seamless"),
        fetch_model_forecast("gfs_seamless"),
        fetch_model_forecast("jma_seamless"),
        fetch_model_forecast("meteofrance")
    ])
    
    Performance:
    - All cache hits: ~<1ms total
    - All cache misses: ~200ms total (parallel)
    - Mixed: ~50-100ms
```

### 3.4 Process Results

```
For each (model_name, forecast, error) in results:
    ├─ Success? → Add to model_forecasts[]
    ├─ Failure? → Use first_successful_forecast as fallback
    └─ Track successful_models[] and failed_models[]

Validate:
    └─ If all failed → Raise RuntimeError
```

### 3.5 Stack Forecasts

```
all_forecasts = torch.stack(model_forecasts, dim=-1)
    ↓
Shape: [time, lat, lon, num_models]
    ├─ time: requested_hours (e.g., 24)
    ├─ lat: lat_grid size
    ├─ lon: lon_grid size
    └─ num_models: 4 (one per model)

Example at [hour=0, lat=0, lon=0]:
    all_forecasts[0, 0, 0, :] = [273.15, 273.20, 273.18, 273.22]
    (4 model predictions for same location/time)
```

### 3.6 Model Prediction

```
Check if model exists for variable:
    ├─ synapse.variable in self.models?
    │   ├─ YES → Run model prediction
    │   │   └─ _predict_with_model()
    │   │       ├─ Flatten: [time*lat*lon, num_models]
    │   │       ├─ Check model type:
    │   │       │   ├─ Scikit-learn? → model.predict()
    │   │       │   └─ PyTorch? → model.forward() (on GPU if available)
    │   │       └─ Reshape: [time, lat, lon]
    │   │
    │   └─ NO → Use first_successful_forecast
    │
    └─ Error handling:
        ├─ Model prediction fails? → Fallback to first_successful_forecast
        └─ No forecasts? → Raise RuntimeError

Output shape: [time, lat, lon]
```

### 3.7 Prepare Response

```
synapse.predictions = output.tolist()
    └─ Convert tensor to nested list: List[List[List[float]]]
        └─ Structure: [time][lat][lon]

synapse.version = zeus_version
    └─ Set miner version string

Log timing breakdown:
    ├─ Initial setup time
    ├─ API calls time
    ├─ Data processing time
    ├─ Model inference time
    └─ Output preparation time
```

---

## 📤 4. RESPONSE PHASE

```
Return synapse to Axon
    ↓
Axon serializes response
    ↓
Send via Bittensor Network (WebSocket)
    ↓
Validator receives response
    ↓
Validator scores predictions
```

---

## 🔁 5. BACKGROUND OPERATIONS

```
Main Loop (runs continuously):
    ├─ Every 30 seconds:
    │   └─ Log "Miner running | uid X | timestamp"
    │
    ├─ Every epoch_length blocks:
    │   └─ sync()
    │       ├─ check_registered()
    │       ├─ resync_metagraph() (if needed)
    │       │   └─ metagraph.sync(subtensor)
    │       ├─ set_weights() (if validator, not miner)
    │       └─ save_state()
    │
    └─ Continuously:
        └─ Axon listens for incoming requests
```

---

## 📊 DATA SHAPES THROUGHOUT PIPELINE

```
Input:
    locations: List[List[Tuple[float, float]]]
        └─ Shape: [lat_grid, lon_grid, 2]
        Example: [3, 3, 2] = 9 locations

After extraction:
    coordinates: torch.Tensor
        └─ Shape: [lat_grid, lon_grid, 2]
    latitudes: torch.Tensor
        └─ Shape: [lat_grid * lon_grid]
    longitudes: torch.Tensor
        └─ Shape: [lat_grid * lon_grid]

After API calls (each model):
    forecast: torch.Tensor
        └─ Shape: [requested_hours, lat_grid, lon_grid]
        Example: [24, 3, 3]

After stacking:
    all_forecasts: torch.Tensor
        └─ Shape: [requested_hours, lat_grid, lon_grid, num_models]
        Example: [24, 3, 3, 4]

For model input:
    forecasts_flat: numpy array or torch.Tensor
        └─ Shape: [time*lat*lon, num_models]
        Example: [216, 4]  (24*3*3=216)

After model prediction:
    predictions: torch.Tensor
        └─ Shape: [requested_hours, lat_grid, lon_grid]
        Example: [24, 3, 3]

Final output:
    synapse.predictions: List[List[List[float]]]
        └─ Shape: [time][lat][lon]
        Example: [24][3][3]
```

---

## ⚡ PERFORMANCE OPTIMIZATIONS

### 1. Response Caching
- **LRU Cache**: 100 entries max, 5-minute TTL
- **Cache hits**: <1ms (instant)
- **Cache misses**: Normal API time (~200ms)
- **Expected hit rate**: 50-80% for similar requests

### 2. Parallel API Calls
- **4 models fetched simultaneously** via `asyncio.gather()`
- **Sequential would be**: 4 × 200ms = 800ms
- **Parallel is**: ~200ms (slowest single call)
- **~75% reduction** in API call time

### 3. Model Pre-warming
- **At startup**: Dummy inference on all models
- **Eliminates**: 20-50ms cold start overhead
- **PyTorch**: GPU warm-up
- **Scikit-learn**: CPU warm-up

### 4. Optimized Tensor Operations
- Pre-allocated tensors
- Reduced numpy ↔ torch conversions
- GPU tensors stay on GPU until final output
- Direct torch operations (view instead of reshape)

### 5. Async Processing
- API calls in thread pool (non-blocking)
- Response processing in executor
- Non-blocking event loop

### Performance Summary:
- **Best case (all cache hits)**: ~50-150ms
- **Typical case (mixed)**: ~100-300ms
- **Worst case (all cache misses)**: ~300-600ms
- **Previous (no optimizations)**: ~1-2 seconds
- **Overall improvement**: 2-4x faster on average

---

## 🔒 SECURITY & VALIDATION

### Blacklist Function
```
Checks (in order):
1. Hotkey exists in metagraph?
2. Validator permit required? → Check validator_permit[uid]
3. Minimum stake required? → Check metagraph.S[uid] >= minimal_alpha_stake
4. All pass? → Allow request
```

### Priority Function
```
Priority = metagraph.S[caller_uid]
    └─ Higher stake = higher priority
    └─ Requests processed in priority order
```

---

## 🛠️ ERROR HANDLING

### API Call Failures
```
If model API call fails:
    └─ Use first_successful_forecast as fallback
    └─ Log warning but continue

If all models fail:
    └─ Raise RuntimeError
    └─ Request fails
```

### Model Prediction Failures
```
If model prediction fails:
    └─ Fallback to first_successful_forecast
    └─ Log error but continue

If no model available:
    └─ Use first_successful_forecast directly
    └─ Log warning
```

### Cache Management
```
If cache full (100 entries):
    └─ LRU eviction (remove oldest entry)
    └─ Add new entry

If cache entry expired (>5 minutes):
    └─ Remove from cache
    └─ Fetch fresh from API
```

---

## 📝 KEY FILES & FUNCTIONS

### Entry Point
- `neurons/miner.py:472` - `__main__` block
- `neurons/miner.py:54` - `Miner.__init__()`
- `neurons/miner.py:156` - `forward()` - Main processing function

### Base Classes
- `zeus/base/neuron.py:60` - `BaseNeuron.__init__()`
- `zeus/base/miner.py:44` - `BaseMinerNeuron.__init__()`
- `zeus/base/miner.py:70` - `BaseMinerNeuron.run()` - Background loop

### Processing Functions
- `neurons/miner.py:208` - `fetch_model_forecast()` - Async API fetch with caching
- `neurons/miner.py:372` - `_process_openmeteo_responses()` - Process API response
- `neurons/miner.py:413` - `_predict_with_model()` - Model inference

### Security
- `zeus/base/miner.py:193` - `_blacklist()` - Request filtering
- `zeus/base/miner.py:260` - `_priority()` - Request prioritization

### Network Sync
- `zeus/base/neuron.py:106` - `sync()` - Network synchronization
- `zeus/base/miner.py:185` - `resync_metagraph()` - Update network state

---

## 🔗 EXTERNAL DEPENDENCIES

### APIs
- **OpenMeteo API**: `https://api.open-meteo.com/v1/forecast`
  - 4 models: gem_seamless, gfs_seamless, jma_seamless, meteofrance
  - Returns hourly weather forecasts

### Blockchain
- **Bittensor Network**: WebSocket connection
  - Subtensor: Blockchain state
  - Metagraph: Network topology
  - Axon: Request/response handler

### Models
- **Trained Models**: `trained_models/model_{variable}_*.pkl`
  - Scikit-learn or PyTorch format
  - One model per ERA5 variable
  - Pre-trained on historical data

---

This completes the data flow from miner startup to request processing and response.

