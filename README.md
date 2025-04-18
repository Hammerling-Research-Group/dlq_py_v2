# dlq_py_v2
Restructured DLQ code in Python

**This is working code, which is changing daily. For users interested in a stable, older version of the DLQ in Python, please see the legacy version [here](https://github.com/Hammerling-Research-Group/dlq_py)**.

This repository contains the latest two-step Python-based approach to the detection, localization, and quantification (DLQ) algorithm for methane emission events using sensor observations and dispersion model outputs. It contains functions and steps to identify emission spikes, localize to the likely source, and then quantify the emission rate. For more details on the method, see [here](https://doi.org/10.1525/elementa.2023.00110).

## Details

There are two key steps, each of which are separated into individual scripts in this repo: 

  - Step 1: Simulates raw data, performs Gaussian Puff calculations, and saves outputs as five different CSV files (one per source name).

  - Step 2: Processes the outputs from Step 1 to perform emission event detection, localization and quantification.

### Step 1: Simulation and Preprocessing

Step 1 simulates atmospheric methane dispersion using a Gaussian puff forward model. Converts raw sensor and source location data into usable format for further analysis. Supports both Basic Mode (CSV outputs) and Advanced Mode (PKL/JSON outputs).

**Inputs:**
  - Raw ADED sensor data file (CSV format)
  - Source location file (CSV format)
  - Sensor location file (CSV format)

**Outputs:**
  - Simulation output per source as CSV files.
  - A `processed_data/` subdirectory containing:
      - Processed CH4 data
      - Aggregated wind data (`WD` and `WS`)
  - Logs for debugging

The module dependencies are included in: 
  - `helper_gpuff_function.py`: Implements the Gaussian puff forward model
  - `helper_distance_conversions.py`: Handles conversions between geographical distances and coordinates

**Sample usage**:

```
# Required input paths
raw_sensor_observations_path = "/path/to/ADED_data_clean.csv"
sensor_locations_path = "/path/to/sensor_locations.csv"
source_locations_path = "/path/to/source_locations.csv"

# Output directory and format
output_directory = "/desired/output/directory/"
output_format = "advanced"  # Options: "basic" or "advanced"
```

**Simulation Parameters:**

```
num_cores_to_use = 2              # Parallel processing
dt = 1                            # Simulation Frequency 
cutoff_t = 20                     # Cutoff time for puff transport
ignore_dist = 80                  # Ignore puffs beyond this distance (meters)
chunk_size = 54                   # Chunk size for time-based processing
emission_rate = 0.001             # Emission rate in g/s
run_mode = "constant"             # Emission mode
start_time = "2022-04-25 12:00:00" 
end_time = "2022-04-25 18:00:00"
timezone = "America/Denver"
```

###  Step 2: Emission Detection, Localization, and Quantification

Step 2 uses the outputs from *Step 1* and observed CH4 data. *Note*: the user can also pass direct input in form of dataframes, dicts, or texts to the `example_calling_with_dataframes` function (see source code for details). The sequence in Step 2 includes:
  1. Remove background levels from CH4 detections
  2. Detect emission events (using spikes or 30-minute intervals, i.e., *30-minute mode*)
  3. Compute alignment of observed vs simulated data
  4. Quantify emission rates and determine source localization

**Inputs:**
  - `processed_data/` subdirectory from *Step 1*
  - `step1_output/` folder containing individual source simulation CSVs
  - Optional: Data in DataFrames for alternate/flexible usage

**Outputs:**
  - `background_removed_data.csv`
  - `event_detection.csv`
  - `alignment_metrics.csv`
  - `event_details.csv`
  - `final_results.csv`
  - `final_results.pkl`

Some additional notes re: *Step 2*:
  - Accepts file paths or in-memory data (DataFrames, dicts, or lists)
  - Converts formats automatically for:
      - `sensor_data`: DataFrame or dict
      - `times`: List or `pd.DatetimeIndex`
      - `weather_data`: dict of Series/lists
      - `simulation_data`: dict of DataFrames
  - Fallback mechanism:
      - First tries file-based input
      - "Falls back" to user-provided DataFrames
      - Final fall back to internal test data (if configured)
  - Can be used as a standalone script or imported into other tools
  - Configurable via command-line or Python dictionaries
  - Logging and error handling included

**Configuration use case**:

```
{
  "gap_time": 30,
  "length_threshold": 15,
  "do_event_detection": true,
  "output_dir": "./output/"
}
```

**Sample usage**: File-based processing

```
example_calling_with_file_paths(
    input_path="processed_data/processed_measured_ch4.csv",
    sources_dir="step1_output/",
    output_dir="./output/",
    config={
        "gap_time": 30,
        "length_threshold": 15,
        "do_event_detection": True
    }
)
```

**Sample usage**: In-memory DataFrame processing

```
example_calling_with_dataframes(
    sensor_data=sensor_df,
    times=timestamp_list,
    weather_data={"WD": wd_series, "WS": ws_series},
    simulation_data=source_sim_dict,
    output_dir="./output/",
    config={
        "gap_time": 30,
        "length_threshold": 15
    }
)
```

## Final output files

- `background_removed_data.csv`: CH4 data with background subtracted
- `event_detection.csv`: Event numbers and timestamps
- `alignment_metrics.csv`: Correlation between simulated and observed CH4
- `event_details.csv`: Estimated source and emission rates for each event, with start and end date, time
- `final_results.csv/.pkl`: Combined results with event metadata

## Requirements

- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - scipy
  - argparse
  - logging
  - json
  - pickle
