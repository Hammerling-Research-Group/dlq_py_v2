# Description:
# Helper functions needed for step2 DLQ.
# Author: Kiran Damodaran (kiran.damodaran@mines.edu)
# Last Updated: April 2025

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt


def find_spikes(times, obs, going_up_threshold=None, return_threshold=None, amp_threshold=None,
                cont_diff_threshold=None, cont_diff_num=None, make_plot=None):
    """
    Detects spikes in time series data based on various thresholds and criteria.

    Parameters:
    -----------
    times : pd.Series
        Time values associated with each observation
    obs : np.ndarray or pd.Series
        Observation values to analyze for spikes
    going_up_threshold : float, optional (default=None)
        Minimum increase between consecutive points to start a potential spike
    return_threshold : float, optional (default=None)
        Threshold as percentage of current maximum to determine end of spike
    amp_threshold : float, optional (default=None)
        Minimum amplitude required for a valid spike
    cont_diff_threshold : float, optional (default=None)
        Maximum difference between consecutive points to consider a plateau
    cont_diff_num : int, optional (default=None)
        Number of consecutive small differences to identify a plateau
    make_plot : bool, optional (default=None)
        Whether to generate a plot of the detected spikes

    Returns:
    --------
    pd.DataFrame
        DataFrame with time values and events column where:
        - NaN values indicate no event
        - Integer values indicate spike IDs (grouped observations in same spike)
    """
    # Set default values if None is provided
    going_up_threshold = 0.25 if going_up_threshold is None else going_up_threshold
    return_threshold = 5 if return_threshold is None else return_threshold
    amp_threshold = 1 if amp_threshold is None else amp_threshold
    cont_diff_threshold = 0.25 if cont_diff_threshold is None else cont_diff_threshold
    cont_diff_num = 10 if cont_diff_num is None else cont_diff_num
    make_plot = False if make_plot is None else make_plot

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

        if not in_event:  # Check for potential new spike start
            current_diff = obs[current_ind] - obs[last_ind]
            threshold_to_use = max(going_up_threshold, amp_threshold) if last_ob else going_up_threshold

            if current_diff > threshold_to_use:  # Start new spike event
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


def remove_background(obs, times, gap_time=None, amp_threshold=None):
    """
    Removes background signal from time series data by identifying and
    preserving only the spike events.

    Parameters:
    -----------
    obs : pd.DataFrame
        DataFrame containing observations, typically methane concentrations
    times : pd.Series or pd.DatetimeIndex
        Time values corresponding to observations
    gap_time : float, optional (default=None)
        Maximum time gap (in minutes) to consider events as connected
    amp_threshold : float, optional (default=None)
        Minimum amplitude required to identify a spike

    Returns:
    --------
    pd.DataFrame
        DataFrame with background removed, same structure as input 'obs'
        Non-spike observations are set to zero, and spike observations have
        their estimated background level subtracted
    """
    # Set default values if None is provided
    gap_time = 5 if gap_time is None else gap_time
    amp_threshold = 0.75 if amp_threshold is None else amp_threshold

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


def detect_events(obs, times, gap_time=None, length_threshold=None, do_event_detection=None):
    """
    Detect emission events using either event detection mode or 30-minute mode

    Parameters:
    -----------
    obs : pd.DataFrame
        DataFrame containing observations from different sensors
    times : pd.Series
        Time values corresponding to observations
    gap_time : int, optional (default=None)
        Maximum time gap (in minutes) to consider events as connected
    length_threshold : int, optional (default=None)
        Minimum number of observations required for a valid event
    do_event_detection : bool, optional (default=None)
        If True, performs event detection based on concentration thresholds
        If False, uses fixed 30-minute intervals

    Returns:
    --------
    tuple(pd.DataFrame, pd.Series)
        - DataFrame with time values and event IDs
        - Series with maximum concentration across all sensors at each time point
    """
    # Set default values if None is provided
    gap_time = 30 if gap_time is None else gap_time
    length_threshold = 15 if length_threshold is None else length_threshold
    do_event_detection = True if do_event_detection is None else do_event_detection

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


def estimate_rate_with_binary_search(all_preds_to_compare, all_obs_to_compare, n_samples=None):
    """
    Estimates emission rate using binary search algorithm with bootstrap sampling.
    This is more efficient than grid search for finding optimal scaling factor.

    Parameters:
    -----------
    all_preds_to_compare : list or array-like
        Predicted concentration values from simulation
    all_obs_to_compare : list or array-like
        Observed concentration values from measurements
    n_samples : int, optional (default=None)
        Number of bootstrap samples to use for uncertainty estimation

    Returns:
    --------
    tuple(float, float, float)
        - Rate estimate in kg/hr
        - Lower bound (5th percentile) in kg/hr
        - Upper bound (95th percentile) in kg/hr
        Returns (np.nan, np.nan, np.nan) if estimation fails
    """
    # Set default value if None is provided
    n_samples = 1000 if n_samples is None else n_samples

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
    error_lower = np.quantile(valid_q_vals, 0.05) * 3600  # 5th percentile for lower bound
    error_upper = np.quantile(valid_q_vals, 0.95) * 3600  # 95th percentile for lower bound

    return rate_est, error_lower, error_upper