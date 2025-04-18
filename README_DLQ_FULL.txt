
# Methane Emission Event Detection, Localization, and Quantification

This repository provides a comprehensive two-step Python-based framework to analyze methane emission events using sensor observations and dispersion model outputs. It helps identify methane spikes, localize the likely source, and quantify the emission rate.


Which is basically performing DLQ which Will has already wrote in R (https://doi.org/10.1525/elementa.2023.00110)

-------------------------------------------------------

# Project Overview

This project consists of two key steps:

Step 1: Simulates raw data, performs Gaussian Puff calculations, and saves outputs as five different CSV files (one per source name).

Step 2: Processes the outputs from Step 1 to perform emission event detection, localization and quantification.

##  Step 1: Simulation and Preprocessing

 Description:
- Simulates atmospheric methane dispersion using a Gaussian Puff model.
- Converts raw sensor and source location data into usable format for further analysis.
- Supports both Basic Mode (CSV outputs) and Advanced Mode (PKL/JSON outputs).

 Inputs:
- Raw ADED sensor data file (CSV format)
- Source location file (CSV format)
- Sensor location file (CSV format)

 Outputs:
- Simulation output per source as CSV files.
- A `processed_data/` folder containing:
  - Processed CH₄ data
  - Aggregated wind data (WD/WS)
- Intermediate files for debugging and verification.

 Dependencies:
- `helper_gpuff_function.py` – Implements the Gaussian Puff physics.
- `helper_distance_conversions.py` – Handles conversions between geographical distances and coordinates.

 User Configuration:
```
# Required input paths
raw_sensor_observations_path = "/path/to/ADED_data_clean.csv"
sensor_locations_path = "/path/to/sensor_locations.csv"
source_locations_path = "/path/to/source_locations.csv"

# Output directory and format
output_directory = "/desired/output/directory/"
output_format = "advanced"  # Options: "basic" or "advanced"
```
Simulation Parameters:

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
-------------------------------------------------------------

##  Step 2: Emission Detection, Localization, and Quantification

 Description:
- Uses outputs from Step 1 and observed CH₄ data (note: now user can also give direct input in form of dataframes, dicts, or texts in 'example_calling_with_dataframes' function):
  1. Remove background levels from CH₄ readings.
  2. Detect emission events (using spikes or 30-minute intervals).
  3. Compute alignment of observed vs simulated data.
  4. Quantify emission rates and determine source localization.

 Inputs:
- `processed_data/` folder from Step 1
- `step1_output/` folder containing 5 source simulation CSVs
- Optional: Direct data in DataFrames for more flexible usage

 Outputs:
- `background_removed_data.csv`
- `event_detection.csv`
- `alignment_metrics.csv`
- `event_details.csv`
- `final_results.csv`
- `final_results.pkl`

Key Features:

# Flexible Input Formats:
- Accepts file paths or in-memory data (DataFrames, dicts, or lists)
- Converts formats automatically for:
  - `sensor_data`: DataFrame or dict
  - `times`: List or `pd.DatetimeIndex`
  - `weather_data`: dict of Series/lists
  - `simulation_data`: dict of DataFrames

# Fallback Mechanism:
- First tries file-based input
- Falls back to user-provided DataFrames
- Final fallback to internal test data (if configured)

# Modular and Extendable:
- Can be used as a standalone script or imported into other tools
- Configurable via command-line or Python dictionaries
- Logging and error handling included

##  Configuration Example
```
{
  "gap_time": 30,
  "length_threshold": 15,
  "do_event_detection": true,
  "output_dir": "./output/"
}
```
--------------------------------------------------------------

# Usage Examples

# File-based processing

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


# In-memory DataFrame processing

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


---------------------------------------------------------

##  Requirements

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

---------------------------------------------------------------

##  Output File Descriptions

| File |                                       | Description |

| `background_removed_data.csv` | CH₄ data with background subtracted |
| `event_detection.csv`         | Event numbers and timestamps |
| `alignment_metrics.csv`       | Correlation between simulated and observed CH₄ |
| `event_details.csv`           | Estimated source and emission rates for each event, with start and end date, time |
| `final_results.csv/.pkl`      | Combined results with event metadata |

-------------------------------------------------------------------------------

##  Directory Structure

```
├── step1_simulation/
│   ├── helper_gpuff_function.py
│   ├── helper_distance_conversions.py
│   └── simulate.py
├── step2_new.py
├── README.md
├── processed_data/
├── step1_output/
└── output/
```

---------------------------------------------------------------------------------------

##  Author

Developed by Kiran Damodaran  
Department of Applied Mathematics and Statistics  
Colorado School of Mines  
kiran.damodaran@mines.edu

For issues or feature requests, please open a GitHub issue or reach out via email.
