# Description:
# Performs emission event detection, localization, and quantification
# using output from the Gaussian puff dispersion model and CMS concentration observations.
# Author: Kiran Damodaran (kiran.damodaran@mines.edu)
# Last Updated: April 2025

import os
import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import logging
import traceback
import json
import argparse
import pickle
from typing import List, Dict, Union, Optional, Tuple, Any


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_default_config() -> Dict[str, Any]:
    """
    Returns a default configuration dictionary with standard settings.

    Returns:
        dict: Default configuration settings
    """
    return {
        'gap_time': 30,  # Time gap (in minutes) for event detection
        'length_threshold': 15,
        'do_event_detection': False,  # Toggle between normal event detection and 30min mode (True - Normal, False - 30min )
        'output_dir': "./output/",  # Base directory for output files
    }

def setup_directories(output_dir: str) -> None:
    """
    Create necessary output directories.

    Args:
        output_dir (str): Base directory for outputs
    """
    os.makedirs(output_dir, exist_ok=True)

# Helper function for spike detection (converted from R)
def find_spikes(times, obs, going_up_threshold=0.25, return_threshold=5, amp_threshold=1,
                cont_diff_threshold=0.25, cont_diff_num=10, make_plot=False):
    """
       Detects spikes in time series data based on various thresholds and criteria.

       Parameters:
       -----------
       times : pd.Series
           Time values associated with each observation
       obs : np.ndarray or pd.Series
           Observation values to analyze for spikes
       going_up_threshold : float, optional (default=0.25)
           Minimum increase between consecutive points to start a potential spike
       return_threshold : float, optional (default=5)
           Threshold as percentage of current maximum to determine end of spike
       amp_threshold : float, optional (default=1)
           Minimum amplitude required for a valid spike
       cont_diff_threshold : float, optional (default=0.25)
           Maximum difference between consecutive points to consider a plateau
       cont_diff_num : int, optional (default=10)
           Number of consecutive small differences to identify a plateau
       make_plot : bool, optional (default=False)
           Whether to generate a plot of the detected spikes

       Returns:
       --------
       pd.DataFrame
           DataFrame with time values and events column where:
           - NaN values indicate no event
           - Integer values indicate spike IDs (grouped observations in same spike)
       """
    # Convert obs to numpy array if it's a pandas Series
    if isinstance(obs, pd.Series):
        obs = obs.to_numpy()

    # Ensure obs is a numpy array
    obs = np.array(obs)

    # Convert obs to float type, replacing non-numeric values with NaN
    obs = np.array(obs, dtype=float)

    # Check if obs is empty or all NaN
    if len(obs) == 0 or np.all(np.isnan(obs)):
        return pd.DataFrame({'time': times, 'events': np.full(len(times), np.nan)})

    events = np.full(len(obs), np.nan)
    count = 0
    in_event = False
    # Find first non-NaN observation index
    start_ind = np.min(np.where(~np.isnan(obs))[0]) + 1
    last_ob = False
    background = np.nan
    # Main loop to process each observation
    for i in range(start_ind, len(obs)):
        if i == len(obs) - 1:
            last_ob = True
        # Handle NaN values and determine indices
        if np.isnan(obs[i]) and not np.isnan(obs[i - 1]):
            last_ind = i - 1
            continue
        elif np.isnan(obs[i]) and np.isnan(obs[i - 1]):
            continue
        elif not np.isnan(obs[i]) and np.isnan(obs[i - 1]):
            current_ind = i
        else:
            current_ind = i
            last_ind = i - 1

        if not in_event:   # Check for potential new spike start
            current_diff = obs[current_ind] - obs[last_ind]
            threshold_to_use = max(going_up_threshold, amp_threshold) if last_ob else going_up_threshold

            if current_diff > threshold_to_use:    # Start new spike event
                in_event = True
                count += 1
                event_obs = [obs[current_ind]]
                events[current_ind] = count
                background = obs[last_ind]
        else:
            current_max = max(event_obs) - background
            current_ob = obs[current_ind] - background

            if (current_ob < 2 * background and current_ob < return_threshold * current_max / 100) or last_ob:
                in_event = False
                event_seq = range(int(np.min(np.where(events == count)[0])),
                                  int(np.max(np.where(events == count)[0])) + 1)
                events[event_seq] = count

                if last_ob:
                    event_size = max(event_obs) - background
                else:
                    event_size = max(event_obs) - np.mean([background, obs[current_ind]])

                if event_size < amp_threshold:
                    events[events == count] = np.nan
                    count -= 1
            else:
                event_obs.append(obs[current_ind])
                events[i] = count

                if len(event_obs) > cont_diff_num:
                    window_start = len(event_obs) - cont_diff_num
                    obs_in_window = event_obs[window_start:]

                    if all(abs(np.diff(obs_in_window)) < cont_diff_threshold):
                        in_event = False
                        event_seq = range(int(np.min(np.where(events == count)[0])),
                                          int(np.max(np.where(events == count)[0])) + 1)
                        events[event_seq] = count
                        event_size = max(event_obs) - np.mean([background, obs[current_ind]])

                        if event_size < amp_threshold:
                            events[events == count] = np.nan
                            count -= 1
                        else:
                            events[current_ind - cont_diff_num + 1:current_ind + 1] = np.nan

    filtered_events = pd.DataFrame({'time': times, 'events': events})
    # Generate plot if requested
    if make_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(times, obs, linewidth=2)
        plt.xlabel('')
        plt.ylabel('Methane [ppm]')
        plt.ylim(0, np.nanmax(obs))

        if not np.all(np.isnan(events)):
            event_nums = np.unique(events[~np.isnan(events)])
            colors = plt.cm.rainbow(np.linspace(0, 1, len(event_nums)))

            for i, event_num in enumerate(event_nums):
                this_spike = np.where(events == event_num)[0]
                plt.scatter(times.iloc[this_spike], obs[this_spike], color=colors[i])
                plt.plot(times.iloc[this_spike], obs[this_spike], color=colors[i], linewidth=2)

        plt.show()

    return filtered_events

def remove_background(obs, times, gap_time=5, amp_threshold=0.75):
    """
        Removes background signal from time series data by identifying and
        preserving only the spike events.

        Parameters:
        -----------
        obs : pd.DataFrame
            DataFrame containing observations, typically methane concentrations
        times : pd.Series or pd.DatetimeIndex
            Time values corresponding to observations
        gap_time : float, optional (default=5)
            Maximum time gap (in minutes) to consider events as connected
        amp_threshold : float, optional (default=0.75)
            Minimum amplitude required to identify a spike

        Returns:
        --------
        pd.DataFrame
            DataFrame with background removed, same structure as input 'obs'
            Non-spike observations are set to zero, and spike observations have
            their estimated background level subtracted
        """
    # Ensure times is a datetime index or series
    if not isinstance(times, (pd.DatetimeIndex, pd.Series)):
        times = pd.to_datetime(times)

    # Convert obs to pandas DataFrame if it's not already
    if not isinstance(obs, pd.DataFrame):
        obs = pd.DataFrame(obs)

    # Convert columns to numeric where possible
    for col in obs.columns:
        obs[col] = pd.to_numeric(obs[col], errors='coerce')

    # Interpolate NA values only for numeric columns
    obs_interpolated = obs.copy()
    numeric_columns = obs.select_dtypes(include=[np.number]).columns
    obs_interpolated[numeric_columns] = obs_interpolated[numeric_columns].interpolate(method='linear', axis=0,
                                                                                      limit_direction='both')

    to_use = obs_interpolated.columns[obs_interpolated.notna().any()]

    for j in to_use:
        this_raw_obs = obs_interpolated[j]
        to_keep = ~this_raw_obs.isna()
        trimmed_obs = this_raw_obs[to_keep]
        trimmed_times = times[to_keep]
        # Detect spikes in this time series
        spikes = find_spikes(trimmed_times, trimmed_obs.values, amp_threshold=amp_threshold, make_plot=False)

        # Ensure spikes and trimmed_obs have the same index
        spikes.index = trimmed_obs.index

        # Add points immediately before and after the spike to the spike mask
        event_nums = np.unique(spikes['events'].dropna())
        for i in event_nums:
            event_mask = spikes['events'] == i
            event_indices = np.where(event_mask)[0]
            if len(event_indices) > 0:
                min_idx = event_indices.min()
                max_idx = event_indices.max()
                start_idx = max(min_idx - 1, 0)
                end_idx = min(max_idx + 1, len(spikes) - 1)
                spikes.iloc[start_idx:end_idx + 1, spikes.columns.get_loc('events')] = i

        # Combine events separated by less than gap_time
        if len(event_nums) > 1:
            for i in range(1, len(event_nums)):
                this_spike = spikes['events'] == event_nums[i]
                previous_spike = spikes['events'] == event_nums[i - 1]

                this_spike_start_time = trimmed_times[this_spike].min()
                previous_spike_end_time = trimmed_times[previous_spike].max()

                # Ensure both are datetime objects and calculate time difference
                this_spike_start_time = pd.to_datetime(this_spike_start_time)
                previous_spike_end_time = pd.to_datetime(previous_spike_end_time)
                time_diff = (this_spike_start_time - previous_spike_end_time).total_seconds() / 60

                if time_diff < gap_time:
                    spikes.loc[this_spike, 'events'] = event_nums[i - 1]
                    event_nums[i] = event_nums[i - 1]

        # Update event_nums after combining events
        event_nums = np.unique(spikes['events'].dropna())

        if len(event_nums) > 0:
            for i in event_nums:
                event_mask = spikes['events'] == i
                first_ob = event_mask.idxmax()
                last_ob = event_mask[::-1].idxmax()

                # Fill in gaps
                spikes.loc[first_ob:last_ob, 'events'] = i

                # Estimate background
                b_left = trimmed_obs.loc[first_ob]
                b_right = trimmed_obs.loc[last_ob]
                b = (b_left + b_right) / 2

                # Remove background from this spike using boolean indexing
                spike_mask = (trimmed_obs.index >= first_ob) & (trimmed_obs.index <= last_ob)
                trimmed_obs[spike_mask] -= b

        # Remove background from all non-spike data
        non_spike_mask = spikes['events'].isna()
        trimmed_obs[non_spike_mask] = 0

        # Set any negative values to zero
        trimmed_obs[trimmed_obs < 0] = 0

        # Save background removed data
        obs.loc[to_keep, j] = trimmed_obs

    return obs


def detect_events(obs, times, gap_time=30, length_threshold=15, do_event_detection=True):
    """
    Detect emission events using either event detection mode or 30-minute mode
    Parameters:
    -----------
    obs : pd.DataFrame
        DataFrame containing observations from different sensors
    times : pd.Series
        Time values corresponding to observations
    gap_time : int, optional (default=30)
        Maximum time gap (in minutes) to consider events as connected
    length_threshold : int, optional (default=15)
        Minimum number of observations required for a valid event
    do_event_detection : bool, optional (default=True)
        If True, performs event detection based on concentration thresholds
        If False, uses fixed 30-minute intervals

    Returns:
    --------
    tuple(pd.DataFrame, pd.Series)
        - DataFrame with time values and event IDs
        - Series with maximum concentration across all sensors at each time point
    """
    # Compute maximum concentration across sensors
    max_obs = obs.max(axis=1)

    # Add a debug print to check max values
    print(f"Maximum observation values: min={max_obs.min()}, max={max_obs.max()}, mean={max_obs.mean()}")

    if do_event_detection:
        # Create initial event mask where methane concentration is greater than 0
        event_mask = max_obs > 0

        # Create initial spikes DataFrame with the same index as times
        spikes = pd.DataFrame({'time': times, 'events': event_mask}, index=times)

        print(f"Number of points with concentration > 0: {event_mask.sum()}")

        # Combine events that are separated by less than gap_time
        to_replace_indices = []
        first_gap = True
        false_seq = []
        false_seq_indices = []

        for i in range(len(times)):
            if not spikes['events'].iloc[i]:
                false_seq.append(times[i])
                false_seq_indices.append(i)
            else:
                if len(false_seq) <= gap_time and not first_gap:
                    to_replace_indices.extend(false_seq_indices)
                first_gap = False
                false_seq = []
                false_seq_indices = []

        # Use iloc to set values by integer position
        if to_replace_indices:
            spikes.iloc[to_replace_indices, spikes.columns.get_loc('events')] = True

        # Replace boolean values with integers to distinguish between events
        spikes['events'] = spikes['events'].map({False: np.nan, True: 0})

        # Assign unique event numbers
        event_num = 0
        prev_value = np.nan

        for idx in spikes.index:
            current_value = spikes.loc[idx, 'events']
            if pd.notna(current_value):
                if pd.isna(prev_value):  # Start of new event
                    event_num += 1
                spikes.loc[idx, 'events'] = event_num
            prev_value = current_value

        # Filter events by length threshold
        event_lengths = spikes['events'].value_counts()
        print(f"Event lengths before filtering: {event_lengths}")
        short_events = event_lengths[event_lengths < length_threshold].index
        print(f"Events to remove (too short): {short_events}")
        spikes.loc[spikes['events'].isin(short_events), 'events'] = np.nan


    else:

        # 30-minute mode
        # Get range of times, rounded to 30-minute intervals
        date_range = [times.min(), times.max()]
        date_range[0] = pd.Timestamp(date_range[0]).ceil('30min')
        date_range[1] = pd.Timestamp(date_range[1]).floor('30min')
        # Times separating the 30-minute intervals
        int_breaks = pd.date_range(start=date_range[0], end=date_range[1], freq='30min')
        # Create spikes DataFrame for 30-minute intervals
        spikes = pd.DataFrame({'time': times, 'events': np.nan}, index=times)
        # Assign event numbers to 30-minute intervals
        for i in range(len(int_breaks) - 1):

            mask = (times >= int_breaks[i]) & (times < int_breaks[i + 1])

            # Fix: Check if mask is already a NumPy array before calling to_numpy()
            if isinstance(mask, pd.Series):
                mask = mask.to_numpy()

            spikes.loc[mask, 'events'] = i + 1

    return spikes, max_obs


def load_source_simulations(sources_dir, sensor_cols=None):
    """
     Loads simulation data for each source from CSV files in a directory.

    Parameters:
    -----------
    sources_dir : str
        Directory path containing source simulation CSV files
    sensor_cols : list, optional (default=None)
        List of column names to select as sensors. If None, all columns are used.

    Returns:
    --------
    dict
        Dictionary where keys are source names and values are DataFrames containing
        simulation data with timestamps as index and sensor readings as columns
    """
    logging.info(f"Loading source simulations from {sources_dir}")
    sim_data = {}

    # Get list of CSV files in the sources directory
    try:
        source_files = [f for f in os.listdir(sources_dir) if f.endswith('.csv')]
        logging.info(f"Found {len(source_files)} source files: {source_files}")

        if not source_files:
            logging.warning(f"No CSV files found in {sources_dir}")
            return sim_data

        for source_file in source_files:
            source_name = os.path.splitext(source_file)[0]  # Use filename without extension as source name
            file_path = os.path.join(sources_dir, source_file)

            try:
                # Try to read the CSV with different timestamp column names
                try:
                    df = pd.read_csv(file_path, parse_dates=["time"])
                    df["time"] = pd.to_datetime(df["time"], utc=True)
                except KeyError:
                    try:
                        df = pd.read_csv(file_path, parse_dates=["time"])
                        df["time"] = pd.to_datetime(df["time"], utc=True)
                    except KeyError:
                        # Try to infer the timestamp column from the available columns
                        df = pd.read_csv(file_path)
                        time_cols = [col for col in df.columns if "time" in col.lower()]
                        if time_cols:
                            df["time"] = pd.to_datetime(df[time_cols[0]], utc=True)
                        else:
                            logging.error(f"No recognizable time column in {file_path}. Skipping.")
                            continue

                # Set the index and select only sensor columns if specified
                df.set_index("time", inplace=True)

                if sensor_cols is not None:
                    # Find the columns that exist in the DataFrame
                    available_cols = [col for col in sensor_cols if col in df.columns]
                    if not available_cols:
                        logging.warning(f"None of the specified sensor columns found in {file_path}.")
                        # Use all columns except timestamp-related ones as sensors
                        available_cols = [col for col in df.columns if "time" not in col.lower()]

                    df = df[available_cols]

                # Store the DataFrame in the sim_data dictionary
                sim_data[source_name] = df
                logging.info(f"Loaded source {source_name} with columns {df.columns.tolist()}")

            except Exception as e:
                logging.error(f"Error loading {file_path}: {str(e)}")
                continue

    except Exception as e:
        logging.error(f"Error accessing sources directory: {str(e)}")

    return sim_data


def estimate_rate_with_binary_search(all_preds_to_compare, all_obs_to_compare, n_samples=1000):
    """
    Estimates emission rate using binary search algorithm with bootstrap sampling.
    This is more efficient than grid search for finding optimal scaling factor.

    Parameters:
    -----------
    all_preds_to_compare : list or array-like
        Predicted concentration values from simulation
    all_obs_to_compare : list or array-like
        Observed concentration values from measurements
    n_samples : int, optional (default=1000)
        Number of bootstrap samples to use for uncertainty estimation

    Returns:
    --------
    tuple(float, float, float)
        - Rate estimate in kg/hr
        - Lower bound (5th percentile) in kg/hr
        - Upper bound (95th percentile) in kg/hr
        Returns (np.nan, np.nan, np.nan) if estimation fails
    """
    q_vals = []

    # Define reasonable bounds for the search
    # Using wider bounds than the original grid to ensure we don't miss any values
    min_q = 0.00001  # Lower than original 0.0001
    max_q = 5000  # Higher than original 3000

    for _ in range(n_samples):
        # Bootstrap sampling
        this_sample = np.random.choice(len(all_preds_to_compare), size=len(all_preds_to_compare) // 2, replace=True)
        sample_obs = np.array(all_obs_to_compare)[this_sample]
        sample_preds = np.array(all_preds_to_compare)[this_sample]

        # Binary search for optimal q
        left, right = min_q, max_q
        best_q = np.nan
        min_sse = float('inf')

        # Perform binary search with a fixed number of iterations
        # More iterations for better precision
        for _ in range(30):  # 30 iterations is enough for high precision
            # Evaluate three points: left, mid, right
            mid = (left + right) / 2

            # Calculate SSE for mid point
            qxp_mid = mid * sample_preds
            sse_mid = np.sqrt(np.mean((sample_obs - qxp_mid) ** 2))

            # Calculate SSE for points to the left and right of mid
            left_test = mid - (mid - left) / 4
            right_test = mid + (right - mid) / 4

            qxp_left = left_test * sample_preds
            sse_left = np.sqrt(np.mean((sample_obs - qxp_left) ** 2))

            qxp_right = right_test * sample_preds
            sse_right = np.sqrt(np.mean((sample_obs - qxp_right) ** 2))

            # Update best q if we found a better SSE
            if sse_mid < min_sse:
                min_sse = sse_mid
                best_q = mid

            # Update search interval based on which direction has lower SSE
            if sse_left < sse_mid:
                right = mid
            elif sse_right < sse_mid:
                left = mid
            else:
                # If mid has the lowest SSE, narrow the interval around it
                left = left_test
                right = right_test

            # Early stopping if the interval is very small
            if (right - left) < 1e-6 * left:
                break

        q_vals.append(best_q)

    # Filter out NaN values
    valid_q_vals = [q for q in q_vals if not np.isnan(q)]

    if not valid_q_vals:
        return np.nan, np.nan, np.nan

    # Convert to hourly rates
    rate_est = np.mean(valid_q_vals) * 3600
    error_lower = np.quantile(valid_q_vals, 0.05) * 3600 # 5th percentile for lower bound
    error_upper = np.quantile(valid_q_vals, 0.95) * 3600# 95th percentile for lower bound

    return rate_est, error_lower, error_upper

def compute_alignment_and_quantification(
        times: pd.DatetimeIndex,
        obs: pd.DataFrame,
        sims: Dict[str, pd.DataFrame] = None,
        sources_dir: str = None,
        spikes: pd.DataFrame = None,
        event_nums: List[int] = None,
        int_breaks: pd.DatetimeIndex = None,
        max_obs: pd.Series = None,
        do_event_detection: bool = False,
        output_dir: str = "./output/",
        gap_time: int = 5,
        length_threshold: int = 15
) -> Dict[str, pd.DataFrame]:
    """
    Compute alignment metrics and quantification estimates for methane events.

    Args:
        times (pd.DatetimeIndex): Timestamps for the observations
        obs (pd.DataFrame): Observed sensor data with background removed
        sims (dict, optional): Dictionary of simulation DataFrames by source name
        sources_dir (str, optional): Directory with simulation files if sims not provided
        spikes (pd.DataFrame, optional): DataFrame with event classifications
        event_nums (list, optional): List of event numbers to process
        int_breaks (pd.DatetimeIndex, optional): Time interval breaks for fixed-interval mode
        max_obs (pd.Series, optional): Maximum observation values
        do_event_detection (bool): Whether to perform event detection
        output_dir (str): Directory for output files
        gap_time (int): Time gap in minutes for event detection
        length_threshold (int): Minimum length for events

    Returns:
        Dict with results DataFrames
    """
    logging.info("Starting alignment and quantification computation...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load simulations if not provided
    if sims is None and sources_dir is not None:
        logging.info("Loading source simulations...")
        # Determine sensor columns from the observation data
        sensor_cols = obs.columns.tolist()

        try:
            sims = load_source_simulations(sources_dir, sensor_cols)
        except Exception as e:
            logging.error(f"Error loading source simulations: {str(e)}")
            logging.error(traceback.format_exc())
            return None

        if not sims:
            logging.warning("No source simulations available for alignment metrics")
            return None

        logging.info(f"Loaded {len(sims)} source simulations: {list(sims.keys())}")

    # Verify required inputs based on mode
    if do_event_detection:
        if spikes is None or event_nums is None:
            logging.error("Event detection mode requires spikes DataFrame and event_nums")
            return None
    else:
        if int_breaks is None:
            logging.error("Interval mode requires int_breaks")
            return None

    # Determine number of intervals
    if do_event_detection:
        n_ints = len(event_nums)
    else:
        n_ints = len(int_breaks) - 1

    # Ensure max_obs exists if needed for rate estimation
    if max_obs is None:
        max_obs = obs.max(axis=1)

    # STEP 1: Compute alignment metrics
    logging.info("Computing alignment metrics...")
    metrics = pd.DataFrame(index=range(n_ints), columns=sims.keys())

    # Ensure that all DataFrames have the same time index
    common_index = times

    # Handle different indexing approaches
    if isinstance(obs.index, pd.DatetimeIndex) and not isinstance(common_index, pd.DatetimeIndex):
        # Convert common_index to DatetimeIndex if needed
        obs_index_copy = obs.index.copy()
        obs = obs.reset_index(drop=True)
    else:
        obs_index_copy = None
        # Set the index on a copy to avoid modifying the original
        obs = obs.copy()
        if not pd.api.types.is_list_like(common_index):
            # Convert to a list-like if it's a single value
            common_index = [common_index]
        obs.index = common_index

    # Align simulation DataFrames to the observation times
    for s in sims.keys():
        if not isinstance(sims[s].index, pd.DatetimeIndex) or not isinstance(common_index, pd.DatetimeIndex):
            # Set index directly if not DatetimeIndex
            sims[s] = sims[s].copy()
            sims[s].index = common_index
        else:
            # Reindex to match observation times using nearest neighbor interpolation for missing values
            sims[s] = sims[s].reindex(common_index, method='nearest')

    # Initialize arrays to store start and end times for each event/interval
    start_times = np.full(n_ints, None, dtype=object)
    end_times = np.full(n_ints, None, dtype=object)

    # Process each interval/event
    for t in range(n_ints):
        logging.info(f"Processing interval {t + 1}/{n_ints}")

        # Get time mask based on mode
        if do_event_detection:
            # Event detection mode - use spike detection output
            event_mask = spikes['events'] == event_nums[t]
            this_mask = event_mask.values

            if any(this_mask):
                start_idx = np.where(this_mask)[0][0]
                end_idx = np.where(this_mask)[0][-1]

                # Store start and end times for this event
                start_times[t] = times[start_idx]
                end_times[t] = times[end_idx]
        else:
            # Interval mode - use interval breaks
            this_mask = (times >= int_breaks[t]) & (times < int_breaks[t + 1])

            # Store start and end times for this interval
            start_times[t] = int_breaks[t]
            end_times[t] = int_breaks[t + 1]

        # Calculate alignment metrics for each source
        for s in sims.keys():
            preds = sims[s]
            all_obs_to_compare = []
            all_preds_to_compare = []

            # Get data for each sensor
            for r in obs.columns:
                if r in preds.columns:
                    # Get observations and predictions for this interval
                    try:
                        # With more robust indexing:
                        if isinstance(this_mask, np.ndarray):
                            # Get the indices where mask is True
                            valid_indices = np.where(this_mask)[0]
                            if len(valid_indices) > 0:
                                obs_int = obs.iloc[valid_indices][r]
                                preds_int = preds.iloc[valid_indices][r]
                            else:
                                continue
                        else:
                            # Convert pandas Series mask to numpy array if needed
                            if isinstance(this_mask, pd.Series):
                                this_mask = this_mask.values

                            # Get indices where mask is True
                            valid_indices = np.where(this_mask)[0]
                            if len(valid_indices) > 0:
                                obs_int = obs.iloc[valid_indices][r]
                                preds_int = preds.iloc[valid_indices][r]
                            else:
                                continue

                        # Filter out NaN values
                        valid_mask = ~obs_int.isna() & ~preds_int.isna()
                        all_obs_to_compare.extend(obs_int[valid_mask])
                        all_preds_to_compare.extend(preds_int[valid_mask])
                    except Exception as e:
                        logging.error(f"Error processing sensor {r} for interval {t + 1}: {str(e)}")
                        logging.error(traceback.format_exc())
                        continue

            # Calculate correlation if there's enough data
            if len(all_obs_to_compare) > 0:
                x = np.array(all_preds_to_compare)
                y = np.array(all_obs_to_compare)

                # Set very small values to 0
                x[x < 1e-30] = 0
                y[y < 1e-30] = 0

                # Calculate correlation coefficient if there's variation in the data
                if not all(x == 0) and not all(y == 0):
                    try:
                        metrics.loc[t, s] = np.corrcoef(x, y)[0, 1]
                    except Exception as e:
                        logging.error(f"Error computing correlation for interval {t + 1}, source {s}: {str(e)}")
                        metrics.loc[t, s] = np.nan

    # Replace NaN with 0 in metrics
    metrics = metrics.fillna(0)
    # Apply the recommended fix to handle the future warning
    metrics = metrics.infer_objects(copy=False)

    # Add event numbers column
    metrics['event_number'] = event_nums if do_event_detection else list(range(n_ints))

    # Save metrics to CSV
    metrics_path = os.path.join(output_dir, "alignment_metrics.csv")
    metrics.to_csv(metrics_path)
    logging.info(f"Alignment metrics saved to {metrics_path}")

    # STEP 2: Compute localization and quantification estimates
    logging.info("Computing localization and quantification estimates...")

    # Initialize arrays for results
    rate_est_all_events = np.full(n_ints, np.nan)
    loc_est_all_events = np.full(n_ints, '', dtype=object)
    error_lower_all_events = np.full(n_ints, np.nan)
    error_upper_all_events = np.full(n_ints, np.nan)

    # Collect predictions and observations across all events
    all_preds_to_compare_all_events = []
    all_obs_to_compare_all_events = []

    # Process each interval/event for localization and quantification
    for t in range(n_ints):
        logging.info(f"Processing interval {t + 1}/{n_ints} for localization")

        # Get metric values for this event/interval
        these_metrics = metrics.iloc[t].drop('event_number')

        # Find the source with the highest metric value
        if not these_metrics.isna().all():
            max_index = these_metrics.idxmax()
            loc_est_all_events[t] = max_index
        else:
            continue

        # Get the mask for this event/interval
        if do_event_detection:
            # Event detection mode
            event_mask = spikes['events'] == event_nums[t]
            this_mask = np.zeros(len(times), dtype=bool)
            if any(event_mask):
                start_idx = np.where(event_mask)[0][0]
                end_idx = np.where(event_mask)[0][-1]
                this_mask[start_idx:end_idx + 1] = True
        else:
            # Interval mode
            this_mask = (times >= int_breaks[t]) & (times < int_breaks[t + 1])

        # Convert the mask to boolean numpy array to avoid index issues
        mask_array = this_mask.values if hasattr(this_mask, 'values') else np.array(this_mask)

        # Get indices where the mask is True
        valid_indices = np.where(mask_array)[0]

        # Make sure indices are within bounds of all dataframes
        max_idx = min(len(sims[loc_est_all_events[t]]), len(obs), len(times))
        valid_indices = valid_indices[valid_indices < max_idx]

        # Use .iloc to index by position instead of .loc which uses the index
        event_preds = sims[loc_est_all_events[t]].iloc[valid_indices]
        event_obs = obs.iloc[valid_indices]

        # For times, handle both Series and arrays
        if isinstance(times, pd.Series) or isinstance(times, pd.DatetimeIndex):
            event_times = times[valid_indices]
        else:
            event_times = times[valid_indices]

        all_preds_to_compare = []
        all_obs_to_compare = []

        # Loop through sensors
        for r in obs.columns:
            if r not in event_preds.columns:
                continue

            these_preds = event_preds[r]
            these_obs = event_obs[r]

            if these_obs.isna().all():
                continue

            try:
                # Ensure event_times is in the right format for find_spikes
                if isinstance(event_times, pd.DatetimeIndex):
                    aligned_event_times = event_times
                else:
                    # Convert to DatetimeIndex if necessary
                    aligned_event_times = pd.DatetimeIndex(event_times)

                # Reset indices for consistent handling
                these_preds_reset = these_preds.reset_index(drop=True)
                these_obs_reset = these_obs.reset_index(drop=True)

                # Find spikes in both predictions and observations
                preds_spikes = find_spikes(aligned_event_times, these_preds_reset.values,
                                           amp_threshold=1, make_plot=False)
                obs_spikes = find_spikes(aligned_event_times, these_obs_reset.values,
                                         amp_threshold=1, make_plot=False)

                # Mask for times in which both preds and obs are in a spike
                both_in_spike_mask = ~preds_spikes['events'].isna() & ~obs_spikes['events'].isna()

                if both_in_spike_mask.any():
                    all_preds_to_compare.extend(these_preds_reset[both_in_spike_mask])
                    all_obs_to_compare.extend(these_obs_reset[both_in_spike_mask])

            except Exception as e:
                logging.error(f"Error processing column {r} for interval {t + 1}: {str(e)}")
                continue

        # If there is enough data to compare, compute rate estimate
        if len(all_preds_to_compare) > 4:
            all_preds_to_compare_all_events.extend(all_preds_to_compare)
            all_obs_to_compare_all_events.extend(all_obs_to_compare)

            # Use binary search instead of grid search
            rate_est, error_lower, error_upper = estimate_rate_with_binary_search(
                all_preds_to_compare,
                all_obs_to_compare,
                n_samples=1000
            )

            rate_est_all_events[t] = rate_est
            error_lower_all_events[t] = error_lower
            error_upper_all_events[t] = error_upper
        else:
            # Check if there are no concentration enhancements
            try:
                # Convert mask to array and get valid indices
                mask_array = this_mask.values if hasattr(this_mask, 'values') else np.array(this_mask)
                valid_indices = np.where(mask_array)[0]

                # Ensure indices are within bounds
                max_idx = len(max_obs)
                valid_indices = valid_indices[valid_indices < max_idx]

                # Extract values based on valid indices
                if isinstance(max_obs, pd.Series):
                    max_obs_values = max_obs.iloc[valid_indices].values
                else:
                    max_obs_values = np.array(max_obs)[valid_indices]

                # Check if all concentrations are zero
                no_concentrations = (max_obs_values == 0).all() if len(max_obs_values) > 0 else True

                # Set rate estimates
                if no_concentrations:
                    # No concentration enhancements, likely no emissions happening
                    rate_est_all_events[t] = 0
                    error_lower_all_events[t] = 0
                    error_upper_all_events[t] = 0
                else:
                    # Emissions could be happening, but not enough alignment to quantify
                    rate_est_all_events[t] = np.nan
                    error_lower_all_events[t] = np.nan
                    error_upper_all_events[t] = np.nan
            except Exception as e:
                logging.error(f"Error checking concentration enhancements for interval {t + 1}: {str(e)}")
                rate_est_all_events[t] = np.nan
                error_lower_all_events[t] = np.nan
                error_upper_all_events[t] = np.nan

    # Format date-time values for output
    formatted_start_times = []
    formatted_end_times = []

    for t in range(n_ints):
        if start_times[t] is not None:
            if hasattr(start_times[t], 'strftime'):
                formatted_start_times.append(start_times[t].strftime('%Y-%m-%dT%H:%M:%S+00:00'))
            else:
                formatted_start_times.append(str(start_times[t]))
        else:
            formatted_start_times.append(None)

        if end_times[t] is not None:
            if hasattr(end_times[t], 'strftime'):
                formatted_end_times.append(end_times[t].strftime('%Y-%m-%dT%H:%M:%S+00:00'))
            else:
                formatted_end_times.append(str(end_times[t]))
        else:
            formatted_end_times.append(None)

    # Save detailed results
    event_details = pd.DataFrame({
        'event_number': event_nums if do_event_detection else list(range(n_ints)),
        'source_location': loc_est_all_events,
        'emission_rate': rate_est_all_events,
        'error_lower': error_lower_all_events,
        'error_upper': error_upper_all_events
    })

    event_details_path = os.path.join(output_dir, "event_details.csv")
    event_details.to_csv(event_details_path, index=False)
    logging.info(f"Event details saved to {event_details_path}")

    # Save final output with start and end times
    final_output = pd.DataFrame({
        'event_number': event_nums if do_event_detection else list(range(n_ints)),
        'start_time': formatted_start_times,
        'end_time': formatted_end_times,
        'localization_estimates': loc_est_all_events,
        'rate_estimates': rate_est_all_events,
        'error_lower': error_lower_all_events,
        'error_upper': error_upper_all_events
    })

    final_output_path = os.path.join(output_dir, "final_results.csv")
    final_output.to_csv(final_output_path, index=False)
    logging.info(f"Final results saved to {final_output_path}")

    final_output_pkl_path = os.path.join(output_dir, "final_results.pkl")
    with open(final_output_pkl_path, 'wb') as f:
        pickle.dump(final_output, f)
    logging.info(f"Final results saved to pickle file: {final_output_pkl_path}")

    # Return all results
    return {
        "metrics": metrics,
        "event_details": event_details,
        "final_output": final_output
    }


def process_methane_data(
        input_data_path: str = None,
        sensor_data: pd.DataFrame = None,
        times: Union[pd.DatetimeIndex, List] = None,
        weather_data: Dict[str, pd.Series] = None,
        sources_dir: str = None,
        simulation_data: Dict[str, pd.DataFrame] = None,
        output_dir: str = "./output/",
        do_event_detection: bool = True,
        gap_time: int = 5,
        length_threshold: int = 15
) -> Dict[str, pd.DataFrame]:
    """
    Complete methane data processing workflow including:
    1. Load data (if paths provided) or use directly provided data
    2. Remove background concentration
    3. Detect emission events
    4. Compute alignment metrics
    5. Estimate emission rates and source locations

    Args:
        input_data_path (str, optional): Path to input CSV file with methane data
        sensor_data (pd.DataFrame, optional): DataFrame with sensor readings (alternative to input_data_path)
        times (Union[pd.DatetimeIndex, List], optional): Timestamps for the data
        weather_data (Dict[str, pd.Series], optional): Dictionary of weather data series (WD, WS)
        sources_dir (str, optional): Directory with source simulations
        simulation_data (Dict[str, pd.DataFrame], optional): Dictionary of simulation DataFrames by source (alternative for source_dir)
        output_dir (str): Output directory path
        do_event_detection (bool): Whether to perform event detection
        gap_time (int): Time gap in minutes for event detection
        length_threshold (int): Minimum length for events

    Returns:
        Dict with results DataFrames
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Input validation and error handling
    if sensor_data is None and input_data_path is None:
        logging.error("Either sensor_data or input_data_path must be provided")
        return None

    if simulation_data is None and sources_dir is None:
        logging.error("Either simulation_data or sources_dir must be provided")
        return None

    # STEP 1: Load data if needed
    if sensor_data is None and input_data_path is not None:
        logging.info(f"Loading methane data from {input_data_path}")
        try:
            df = pd.read_csv(input_data_path)

            if times is None:
                # Extract timestamps from the data
                if 'time' in df.columns:
                    times = pd.to_datetime(df['time'])
                else:
                    logging.error("No 'time' column found in input data")
                    return None

            # Extract weather data if not provided
            if weather_data is None:
                weather_data = {}
                if 'WD' in df.columns:
                    weather_data['WD'] = df['WD']
                if 'WS' in df.columns:
                    weather_data['WS'] = df['WS']

            # Get concentration column(s)
            if 'concentration' in df.columns:
                sensor_data = df[['concentration']]
            else:
                # Assume all columns except time, WD, WS are sensor data
                sensor_cols = list(set(df.columns) - {'time', 'WD', 'WS'})
                if not sensor_cols:
                    logging.error("No sensor data columns found in input file")
                    return None

                logging.info(f"Using sensor columns: {sensor_cols}")
                sensor_data = df[sensor_cols]

        except Exception as e:
            logging.error(f"Error loading methane data: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    # Ensure times is a DatetimeIndex
    if times is not None and not isinstance(times, pd.DatetimeIndex):
        try:
            times = pd.DatetimeIndex(times)
        except Exception as e:
            logging.error(f"Error converting times to DatetimeIndex: {str(e)}")
            return None

    # Ensure sensor_data has appropriate index
    if sensor_data is not None and len(times) == len(sensor_data):
        try:
            # Make a copy to avoid modifying the original
            sensor_data = sensor_data.copy()
            sensor_data.index = times
        except Exception as e:
            logging.error(f"Error setting index on sensor_data: {str(e)}")
            logging.error(traceback.format_exc())
            # Continue without setting index, will handle later

    # STEP 2: Remove background
    logging.info("Removing background concentration...")
    obs = remove_background(sensor_data, times, gap_time=gap_time, amp_threshold=0.75)

    # Add the timestamps back to the processed data
    background_removed_data = obs.copy()
    background_removed_data.insert(0, 'time', times)

    # Add back weather data if available
    if weather_data:
        for key, data in weather_data.items():
            background_removed_data[key] = data

    # Save the processed data if output directory provided
    bgr_path = os.path.join(output_dir, "background_removed_data.csv")
    background_removed_data.to_csv(bgr_path, index=False)
    logging.info(f"Background-removed data saved to {bgr_path}")

    # STEP 3: Event Detection
    logging.info("Detecting emission events...")
    spikes, max_obs = detect_events(
        obs,
        times,
        gap_time=gap_time,
        length_threshold=length_threshold,
        do_event_detection=do_event_detection
    )

    # Get unique event numbers (excluding NaN)
    event_nums = pd.Series(spikes['events'].unique())
    event_nums = event_nums.dropna().astype(int).sort_values().tolist()
    n_ints = len(event_nums)

    logging.info(f"Detected {n_ints} events: {event_nums}")

    # Store int_breaks for 30-minute mode
    int_breaks = None
    if not do_event_detection:
        date_range = [times.min(), times.max()]
        date_range[0] = pd.Timestamp(date_range[0]).ceil('30min')
        date_range[1] = pd.Timestamp(date_range[1]).floor('30min')
        int_breaks = pd.date_range(start=date_range[0], end=date_range[1], freq='30min')

    # Ensure spikes has the same index as times
    spikes = spikes.reindex(times)

    # If max_obs is a Series, ensure it has the same index
    if isinstance(max_obs, pd.Series):
        max_obs = max_obs.reindex(times)

    # Create event results DataFrame
    event_results = pd.DataFrame({
        'time': times.strftime('%Y-%m-%d %H:%M:%S'),
        'max_concentration': max_obs.round(4),
        'event_number': spikes['events'].astype('Int64')
    })

    # Replace NaN with 'NA' string to match original output
    event_results['event_number'] = event_results['event_number'].astype(str).replace('nan', 'NA')

    # Save event detection results
    event_path = os.path.join(output_dir, "event_detection.csv")
    event_results.to_csv(event_path, index=False)
    logging.info(f"Event detection results saved to {event_path}")

    # STEP 4-5: Compute alignment metrics and quantification estimates
    results = compute_alignment_and_quantification(
        times=times,
        obs=obs,
        sims=simulation_data,
        sources_dir=sources_dir,
        spikes=spikes,
        event_nums=event_nums,
        int_breaks=int_breaks,
        max_obs=max_obs,
        do_event_detection=do_event_detection,
        output_dir=output_dir,
        gap_time=gap_time,
        length_threshold=length_threshold
    )

    # Combine all results
    all_results = {
        "background_removed_data": background_removed_data,
        "event_detection": event_results
    }

    if results:
        all_results.update(results)

    logging.info("Methane data processing completed successfully.")
    return all_results

#-------------------------------------------------------------------------------
# Can be called in another seperate code/Program as well
from step2_new import (
    process_methane_data,
    compute_alignment_and_quantification,
    get_default_config
)

def example_calling_with_file_paths(input_path, sources_dir, output_dir, config):
    """
    Process methane data using file paths

    Parameters:
    -----------
    input_path : str
        Path to input data file
    sources_dir : str
        Path to directory containing source simulation files
    output_dir : str
        Directory to save results
    config : dict
        Configuration dictionary with processing parameters

    Returns:
    --------
    tuple
        (success, results) - Boolean success status and results dictionary if successful
    """
    # Check if files exist
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return False, None

    if not os.path.exists(sources_dir):
        logging.error(f"Sources directory not found: {sources_dir}")
        return False, None

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Process the data with file paths
        results = process_methane_data(
            input_data_path=input_path,
            sources_dir=sources_dir,
            output_dir=output_dir,
            do_event_detection=config.get('do_event_detection', True),
            gap_time=config.get('gap_time', 30),
            length_threshold=config.get('length_threshold', 15)
        )

        # Print results overview
        if results:
            logging.info(f"File-based analysis completed successfully!")
            logging.info(f"Number of events detected: {len(results['event_details'])}")
            logging.info(f"Results saved to: {output_dir}")
            return True, results
        else:
            logging.error("File-based analysis failed. Check logs for details.")
            return False, None

    except Exception as e:
        logging.error(f"Error in file path processing: {str(e)}")
        logging.error(traceback.format_exc())
        return False, None


def example_calling_with_dataframes(sensor_data=None, times=None, weather_data=None, simulation_data=None,
                                    output_dir=None, config=None):
    """
    Process methane data using provided dataframes

    Parameters:
    -----------
    sensor_data : pd.DataFrame
        DataFrame containing sensor readings with directional columns (E, N, NE, etc.)
        Example format:
        {
            'E': [1.977, 1.926, 1.929, ...],
            'N': [1.985, 1.968, 1.981, ...],
            'NE': [1.703, 1.687, 1.690, ...],
            'NW': [1.992, 1.992, 1.956, ...],
            'S': [1.801, 1.793, 1.806, ...],
            'SE': [2.527, 2.800, 2.126, ...],
            'SW': [2.345, 2.349, 2.349, ...],
            'W': [2.234, 2.234, 2.240, ...]
        }
    times : list or pd.DatetimeIndex
        Timestamps corresponding to each row in sensor_data
        Example format: ['2022-04-29 15:00:00-06:00', '2022-04-29 15:01:00-06:00', ...]
    weather_data : dict
        Dictionary with 'WD' (wind direction) and 'WS' (wind speed) as pd.Series
        Example format:
        {
            'WD': pd.Series([45, 90, 135, ...]),  # Wind direction in degrees
            'WS': pd.Series([2.5, 3.0, 3.5, ...])  # Wind speed values
        }
    simulation_data : dict
        Dictionary of DataFrames with source names as keys and simulation data as values
        Each source should have the same directional columns as sensor_data
        Example format:
        {
            'source1': DataFrame with same columns as sensor_data,
            'source2': DataFrame with same columns as sensor_data,
            ...
        }
    output_dir : str
        Directory to save results
    config : dict
        Configuration dictionary with processing parameters

    Returns:
    --------
    tuple
        (success, results) - Boolean success status and results dictionary if successful
    """
    # Validate inputs
    if config is None:
        config = get_default_config()
        logging.info("Using default configuration")

    if output_dir is None:
        output_dir = config.get('output_dir', './output')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Validate required inputs
    if sensor_data is None:
        logging.error("Sensor data is required but was not provided")
        return False, None

    # Convert to DataFrame if dict
    if isinstance(sensor_data, dict):
        try:
            sensor_data = pd.DataFrame(sensor_data)
            logging.info("Converted sensor_data dictionary to DataFrame")
        except Exception as e:
            logging.error(f"Failed to convert sensor_data dict to DataFrame: {str(e)}")
            return False, None

    if not isinstance(sensor_data, pd.DataFrame):
        logging.error(f"Sensor data must be a pandas DataFrame or convertible dict, got {type(sensor_data)}")
        return False, None

    if len(sensor_data) == 0:
        logging.error("Sensor data is empty")
        return False, None

    # Validate times
    if times is None:
        logging.error("Times data is required but was not provided")
        return False, None

    if len(times) != len(sensor_data):
        logging.error(f"Length mismatch: times ({len(times)}) and sensor_data ({len(sensor_data)})")
        return False, None

    # Convert times to pandas DatetimeIndex if it's a list
    if isinstance(times, list):
        try:
            times = pd.DatetimeIndex(times)
        except Exception as e:
            logging.error(f"Failed to convert times to DatetimeIndex: {str(e)}")
            return False, None

    # Validate weather data
    if weather_data is None:
        logging.error("Weather data is required but was not provided")
        return False, None

    if not isinstance(weather_data, dict) or 'WD' not in weather_data or 'WS' not in weather_data:
        logging.error("Weather data must be a dictionary with 'WD' and 'WS' keys")
        return False, None

    # Convert Series to pandas Series if they are lists
    if isinstance(weather_data['WD'], list):
        weather_data['WD'] = pd.Series(weather_data['WD'])
    if isinstance(weather_data['WS'], list):
        weather_data['WS'] = pd.Series(weather_data['WS'])

    if len(weather_data['WD']) != len(sensor_data) or len(weather_data['WS']) != len(sensor_data):
        logging.error(
            f"Length mismatch: weather data (WD: {len(weather_data['WD'])}, WS: {len(weather_data['WS'])}) and sensor_data ({len(sensor_data)})")
        return False, None

    # Validate simulation data
    if simulation_data is None:
        logging.error("Simulation data is required but was not provided")
        return False, None

    if not isinstance(simulation_data, dict) or len(simulation_data) == 0:
        logging.error("Simulation data must be a non-empty dictionary of source simulations")
        return False, None

    # Convert dictionary simulation data to DataFrames if needed
    for source_name, source_data in simulation_data.items():
        if isinstance(source_data, dict):
            try:
                simulation_data[source_name] = pd.DataFrame(source_data)
                logging.info(f"Converted simulation data for {source_name} from dict to DataFrame")
            except Exception as e:
                logging.error(f"Failed to convert simulation data for {source_name} from dict to DataFrame: {str(e)}")
                return False, None

    # Check that each simulation source has the same columns as sensor_data
    for source_name, source_df in simulation_data.items():
        if not isinstance(source_df, pd.DataFrame):
            logging.error(
                f"Simulation data for {source_name} must be a DataFrame or convertible dict, got {type(source_df)}")
            return False, None

        if not all(col in source_df.columns for col in sensor_data.columns):
            logging.error(f"Simulation data for {source_name} is missing columns that are in sensor_data")
            logging.error(f"Sensor columns: {sensor_data.columns.tolist()}")
            logging.error(f"Simulation columns: {source_df.columns.tolist()}")
            return False, None

        if len(source_df) != len(sensor_data):
            logging.error(
                f"Length mismatch: simulation data for {source_name} ({len(source_df)}) and sensor_data ({len(sensor_data)})")
            return False, None

    logging.info("All input validation checks passed.")

    try:
        # Process the data with direct inputs, user should pass the dataframe or file
        results = process_methane_data(
            sensor_data=sensor_data,
            times=times,
            weather_data=weather_data,
            simulation_data=simulation_data,
            output_dir=output_dir,
            do_event_detection=config.get('do_event_detection', True),
            gap_time=config.get('gap_time', 30),  # Default gap time in minutes
            length_threshold=config.get('length_threshold', 15)  # Default length threshold
        )

        # Print results overview
        if results:
            logging.info(f"Dataframe-based analysis completed successfully!")
            logging.info(f"Number of events detected: {len(results['event_details'])}")
            logging.info(f"Results saved to: {output_dir}")
            return True, results
        else:
            logging.error("Dataframe-based analysis failed.")
            return False, None

    except Exception as e:
        logging.error(f"Error in dataframe processing: {str(e)}")
        logging.error(traceback.format_exc())
        return False, None


def process_with_fallback(input_path=None, sources_dir=None, output_dir=None, config=None,
                          sensor_data=None, times=None, weather_data=None, simulation_data=None):
    """
    Process data with file paths first, then fall back to provided dataframes

    Parameters:
    -----------
    input_path : str
        Path to input data file
    sources_dir : str
        Path to directory containing source simulation files
    output_dir : str
        Directory to save results
    config : dict
        Configuration dictionary
    sensor_data : pd.DataFrame or dict
        DataFrame or dict containing sensor readings (for fallback)
    times : list or pd.DatetimeIndex
        Timestamps corresponding to each row in sensor_data (for fallback)
    weather_data : dict
        Dictionary with 'WD' and 'WS' keys (for fallback)
    simulation_data : dict
        Dictionary of DataFrames or dicts with source names as keys (for fallback)

    Returns:
    --------
    dict or None
        Processing results if successful, None otherwise
    """
    # Use provided paths or defaults
    if input_path is None:
        input_path = "/home/kiran/dlqcheck/dlq_py/output_data_newcheck/processed_data/processed_measured_ch4.csv"

    if sources_dir is None:
        sources_dir = "/home/kiran/dlqcheck/dlq_py/output_data_newcheck/step1_output/"

    # Get default configuration if not provided
    if config is None:
        config = get_default_config()
        config.update({
            'output_dir': output_dir,
            'gap_time': 30,  # Custom gap time in minutes
            'length_threshold': 15,
            'do_event_detection': True,  # make sure to match with get_default_config
        })

    # First try with file paths
    file_success = False
    results = None

    try:
        logging.info(f"Attempting file-based processing with input: {input_path}")
        file_success, results = example_calling_with_file_paths(input_path, sources_dir, output_dir, config)
    except Exception as e:
        logging.error(f"Error in file path processing: {str(e)}")
        logging.error(traceback.format_exc())
        file_success = False

    # If file processing failed, try with provided dataframes
    if not file_success:
        if all(param is not None for param in [sensor_data, times, weather_data, simulation_data]):
            logging.info("File-based processing failed, falling back to provided dataframes")
            try:
                dataframe_success, results = example_calling_with_dataframes(
                    sensor_data=sensor_data,
                    times=times,
                    weather_data=weather_data,
                    simulation_data=simulation_data,
                    output_dir=output_dir,
                    config=config
                )
                if not dataframe_success:
                    logging.error("Both file-based and provided dataframe processing failed.")
                else:
                    return results
            except Exception as e:
                logging.error(f"Error in provided dataframe processing: {str(e)}")
                logging.error(traceback.format_exc())
                logging.error("No sample data will be used - please provide valid input data.")
        else:
            logging.error("File-based processing failed and no valid dataframes were provided.")
            logging.error("Please provide valid input data via files or dataframes.")

    return results


def main():
    """Main function to set up logging and run the analysis"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("methane_analysis.log"),
            logging.StreamHandler()
        ]
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Methane analysis tool')
    parser.add_argument('--mode', choices=['file', 'dataframe', 'fallback'], default='fallback',
                        help='Processing mode: file, dataframe, or fallback (tries file then dataframe)')
    parser.add_argument('--input', help='Input file path for file mode')
    parser.add_argument('--sources', help='Sources directory for file mode')
    parser.add_argument('--output', default='./output', help='Output directory')
    parser.add_argument('--config', help='Path to configuration JSON file')
    args = parser.parse_args()

    # Load configuration
    config = get_default_config()
    if args.config:
        try:
            with open(args.config, 'r') as f:
                user_config = json.load(f)
                config.update(user_config)
        except Exception as e:
            logging.error(f"Error loading configuration file: {str(e)}")

    # Update output directory from args
    config['output_dir'] = args.output

    # Choose processing mode
    if args.mode == 'file':
        if not args.input or not args.sources:
            logging.error("Input file and sources directory are required for file mode")
            return
        print(f"Processing file: {args.input}")
        success, results = example_calling_with_file_paths(args.input, args.sources, args.output, config)
        if success:
            print("File processing completed successfully!")
        else:
            print("File processing failed.")
    elif args.mode == 'dataframe':
        print(
            "For dataframe mode, please import this module and call example_calling_with_dataframes() with your dataframes")
        print("Example usage:")
        print("  from methane_processing import example_calling_with_dataframes")
        print("  success, results = example_calling_with_dataframes(")
        print("      sensor_data=your_sensor_df,  # Can be DataFrame or dict")
        print("      times=your_timestamps,  # Can be list or DatetimeIndex")
        print("      weather_data={'WD': wind_direction_series, 'WS': wind_speed_series},  # Can use lists or Series")
        print("      simulation_data={'source1': source1_df, 'source2': source2_df},  # Can use DataFrames or dicts")
        print("      output_dir='./output',")
        print("      config={'gap_time': 30, 'length_threshold': 15}")
        print("  )")
        print("See the function docstring for detailed examples of expected data formats.")
    elif args.mode == 'fallback':
        print("Running with fallback mode (tries file path first, then dataframes)")
        if args.input and args.sources:
            process_with_fallback(args.input, args.sources, args.output, config)
        else:
            print("No input file or sources specified, will try with defaults")
            process_with_fallback(output_dir=args.output, config=config)
    else:
        logging.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()