"""
Impact of Spike Sorting Quality on Intracortical BCI Decoding

Authors: Sai Dore and Kye Xiong

This script contains the within-session and cross-session analysis
used for the final BCI project. 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.preprocessing as spre
import spikeinterface.full as si
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import h5py
import os
import glob
import shutil

# =========================
# SESSION SETTINGS
# Change only these lines for each new session
# =========================
#Session names used are 1: indy_20160915_01; 2: indy_20160930_05; 3: indy_20161025_04
session_name = "indy_20161025_04" 
file_path = f"/content/{session_name}.mat"
raw_file_path = f"/content/{session_name}.nwb"
binary_path = f"/content/{session_name}_raw.dat"
sorting_output_folder = f"/content/kilosort4_output_{session_name}"

# Upload file only if it is not already loaded
print("Using file:", file_path)

def read_matlab_array(f, ref, transpose=False):
    """
    Load a MATLAB v7.3 HDF5 array from a reference and return it as a NumPy array.
    """
    obj = f[ref]
    # MATLAB v7.3 empty arrays can otherwise look like small numeric arrays
    if "MATLAB_empty" in obj.attrs:
        empty_flag = np.asarray(obj.attrs["MATLAB_empty"]).astype(bool).any()
        if empty_flag:
            return np.array([])
    arr = np.asarray(obj)
    if transpose and arr.ndim == 2:
        arr = arr.T  # Transpose 2D arrays when needed
    return np.asarray(arr).squeeze()

with h5py.File(file_path, "r") as f:
    # Load behavioral timestamps and position measurements
    t = np.asarray(f["t"]).squeeze()
    finger_pos = np.asarray(f["finger_pos"]).T
    cursor_pos = np.asarray(f["cursor_pos"]).T
    target_pos = np.asarray(f["target_pos"]).T
    # Load MATLAB spike cell array as refs to HDF5 object
    spikes_ref = f["spikes"]
    spikes = np.empty(spikes_ref.shape, dtype=object)
    # Resolve each spike reference and store its spike timestamps
    for i in range(spikes_ref.shape[0]):
        for j in range(spikes_ref.shape[1]):
            spikes[i, j] = read_matlab_array(f,spikes_ref[i, j])
    
    wf_ref = f["wf"]  # Waveform cell array
    waveforms = np.empty(wf_ref.shape, dtype=object)
    # Resolve each waveform reference
    for i in range(wf_ref.shape[0]):
        for j in range(wf_ref.shape[1]):
            waveforms[i, j] = read_matlab_array(f,wf_ref[i, j], transpose=True)

# MATLAB/HDF5 orientation is reversed so need to transpose 
spikes = spikes.T
waveforms = waveforms.T

print("Time shape:", t.shape)
print("Finger position shape:", finger_pos.shape)
print("Cursor position shape:", cursor_pos.shape)
print("Target position shape:", target_pos.shape)
print("Spike array shape:", spikes.shape)
print("Waveform array shape:", waveforms.shape)
# Plot the recorded X and Y cursor position over time
plt.figure(figsize=(12, 4))
plt.plot(t, cursor_pos[:, 0], label="X")
plt.plot(t, cursor_pos[:, 1], label="Y")
plt.xlabel("Time (s)")
plt.ylabel("Position (mm)")
plt.title("Cursor Position")
plt.legend()
plt.show()

dt = np.mean(np.diff(t))  # Estimate average time between behavioral samples
# X and Y cursor velocity from the position signals and combine into 2 col array
velocity_x = np.gradient(cursor_pos[:, 0], dt)
velocity_y = np.gradient(cursor_pos[:, 1], dt)
velocity = np.column_stack((velocity_x, velocity_y))
print(velocity.shape)
# Plot the calculated X and Y cursor velocity
plt.figure(figsize=(12, 4))
plt.plot(t, velocity[:, 0], label="X velocity")
plt.plot(t, velocity[:, 1], label="Y velocity")
plt.xlabel("Time (s)")
plt.ylabel("Velocity")
plt.title("Cursor Velocity")
plt.legend()
plt.show()

# Inspect spike entries from the first few electrodes
print("Spike matrix shape:", spikes.shape)
for channel in range(min(5, spikes.shape[0])):
    for unit in range(spikes.shape[1]):
        spike_times = spikes[channel, unit]
        if isinstance(spike_times, np.ndarray):  # Only process entries stored as NumPy arrays
            if spike_times.ndim > 0:  # Check if spike entries contain 1+ timestamps
                print(f"Channel {channel}, Unit {unit}: "
                    f"{len(spike_times)} spikes")
            else:  # Handle 0D array scalar case safely
                print(f"Channel {channel}, Unit {unit}: Scalar value ({spike_times.item()})")

# Store valid sorted-unit spike trains and basic unit metadata
sorted_units = []
sorted_unit_info = []

for channel in range(spikes.shape[0]):
    # Units 1-4 are sorted units and Unit 0 is the unsorted/hash unit
    for unit in range(1, spikes.shape[1]):
        spike_times = np.asarray(spikes[channel, unit]).flatten()
        spike_times = spike_times[np.isfinite(spike_times)]  # Keep only real timestamps
        # Keep timestamps inside this recording
        spike_times = spike_times[(spike_times >= t[0]) &(spike_times <= t[-1])]
        # Keep only units that contain at least one valid spike
        if len(spike_times) > 0:
            sorted_units.append(spike_times)
            sorted_unit_info.append({"channel": channel, "unit": unit, "n_spikes": len(spike_times)})

# Print basic statistics about the sorted units that were loaded
print("Number of real sorted units:", len(sorted_units))
unit_counts = np.array([x["n_spikes"] for x in sorted_unit_info])
print("Minimum spikes:", unit_counts.min())
print("Median spikes:", np.median(unit_counts))
print("Maximum spikes:", unit_counts.max())

bin_size = 0.050  # 50 ms bins for neural activity and behavioral velocity
start_time = t[0] # Start time of behavioral recording
end_time = t[-1]  # End time
# Create edges of each bin and calculate centers  
bin_edges = np.arange(start_time, end_time + bin_size, bin_size)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
spike_counts = np.zeros((len(bin_centers), len(sorted_units)))
# Count the number of spikes from each unit in every 50 ms bin
for unit_idx, spike_times in enumerate(sorted_units):
    counts, _ = np.histogram(spike_times,bins=bin_edges)
    spike_counts[:, unit_idx] = counts

# Convert spike counts into firing rates in spikes per second
firing_rates = spike_counts / bin_size
print("Firing rate matrix:", firing_rates.shape)
# Create interpolation functions for X and Y cursor velocity which allows eval at the center of each neural time bin
vx_interp = interp1d(t, velocity[:, 0], bounds_error=False, fill_value="extrapolate")
vy_interp = interp1d(t, velocity[:, 1], bounds_error=False, fill_value="extrapolate")
binned_velocity = np.column_stack([vx_interp(bin_centers), vy_interp(bin_centers)])
print("Neural features:", firing_rates.shape)
print("Velocity targets:", binned_velocity.shape)

# Use firing rates as decoder inputs and cursor velocity as the target
X = firing_rates
y = binned_velocity
# Use the first 80% of the recording for training and the final 20% for testing
split = int(0.8 * len(X))
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# Ridge regression model as a simple linear decoding baseline
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate decoding performance separately for X and Y velocity and overall prediction error
r2_x = r2_score(y_test[:, 0], y_pred[:, 0])
r2_y = r2_score(y_test[:, 1], y_pred[:, 1])
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("R2 X:", r2_x)
print("R2 Y:", r2_y)
print("RMSE:", rmse)

# Plot a portion of the true and predicted X velocity for visual comparison
plt.figure(figsize=(12, 4))
plt.plot(y_test[:500, 0], label="Actual")
plt.plot(y_pred[:500, 0], label="Predicted")
plt.xlabel("50 ms bins")
plt.ylabel("X velocity")
plt.title("Actual vs Predicted Hand Velocity")
plt.legend()
plt.show()

# Build one combined spike train for each electrode
threshold_channels = []
for channel in range(spikes.shape[0]):
    channel_spikes = []
    # Collect spike timestamps from every unit on this electrode
    for unit in range(spikes.shape[1]):
        spike_times = np.asarray(spikes[channel, unit]).flatten()
        spike_times = spike_times[np.isfinite(spike_times)]  # Remove invalid timestamp values
        # Keep only spikes occurring during the behavioral recording
        spike_times = spike_times[(spike_times >= t[0]) & (spike_times <= t[-1])] 
        if len(spike_times) > 0:
            channel_spikes.append(spike_times)
    # Combine all spike activity from the electrode into one ordered spike train
    if len(channel_spikes) > 0:
        combined = np.concatenate(channel_spikes)
        combined = np.sort(combined)
        threshold_channels.append(combined)
print("Electrodes with threshold crossings:",len(threshold_channels))

# Matrix to store threshold-crossing counts for each electrode and time bin
threshold_counts = np.zeros((len(bin_centers), len(threshold_channels)))
# Count threshold-crossing events in each 50 ms bin
for channel_idx, spike_times in enumerate(threshold_channels):
    counts, _ = np.histogram(spike_times,bins=bin_edges)
    threshold_counts[:, channel_idx] = counts
threshold_rates = threshold_counts / bin_size  # Convert threshold-crossing counts into firing rates
print("Threshold-crossing firing-rate matrix:",threshold_rates.shape)

X_threshold = threshold_rates
y = binned_velocity
# Apply the same chronological 80/20 train-test split
split = int(0.8 * len(X_threshold))
X_train_tc = X_threshold[:split]
X_test_tc = X_threshold[split:]
y_train_tc = y[:split]
y_test_tc = y[split:]

# Train a Ridge regression model using the threshold-crossing features
ridge_tc = Ridge(alpha=1.0)
ridge_tc.fit(X_train_tc,y_train_tc)
y_pred_tc = ridge_tc.predict(X_test_tc)
# Calculate R2 separately for X and Y velocity and overall RMSE
r2_x_tc = r2_score(y_test_tc[:, 0], y_pred_tc[:, 0])
r2_y_tc = r2_score(y_test_tc[:, 1], y_pred_tc[:, 1])
rmse_tc = np.sqrt(mean_squared_error(y_test_tc,y_pred_tc))
print("THRESHOLD CROSSINGS")
print("R2 X:", r2_x_tc)
print("R2 Y:", r2_y_tc)
print("RMSE:", rmse_tc)

def train_kalman_transition(y_train):
    """
    Pre-trains the state transition model (which is independent of neural features).
    """
    y_mean = y_train.mean(axis=0)  # Compute the mean velocity across the training data.
    Yc = y_train - y_mean  # Center the velocity data around zero
    Y_prev = Yc[:-1] 
    Y_next = Yc[1:]
    A = np.linalg.lstsq(Y_prev, Y_next, rcond=None)[0].T  # Predicts the next velocity state from the previous one
    state_residuals = (Y_next - Y_prev @ A.T)  # Prediction error of the learned transition model
    Q = np.cov(state_residuals, rowvar=False)   # Estimate process noise covariance from the transition residuals
    Q += np.eye(Q.shape[0]) * 1e-6
    return A, Q, y_mean

def train_kalman_observation(X_train, y_train):
    """
    Trains the observation model (which depends on the neural features X_train).
    """
    # Compute the mean neural activity and mean velocity and center around 0
    x_mean = X_train.mean(axis=0)
    y_mean = y_train.mean(axis=0)
    Xc = X_train - x_mean
    Yc = y_train - y_mean

    H = np.linalg.lstsq(Yc, Xc, rcond=None)[0].T  # Maps velocity states to neural observations
    observation_residuals = (Xc - Yc @ H.T)  # Error between observed and predicted neural activity
    R = np.cov(observation_residuals, rowvar=False)  # Observation noise covariance
    R += np.eye(R.shape[0]) * 1e-3
    return H, R, x_mean

def run_kalman_filter(X_test, x_mean, y_mean, A, Q, H, R):
    """
    Runs an optimized Kalman filter sweep on the test set.
    """
    X_test_centered = X_test - x_mean
    state = np.zeros(2)  # 2D velocity state
    P = np.eye(2)  # State covariance matrix
    predictions = np.zeros((len(X_test), 2))  # Predicted X/Y velocity at each time bin
    I = np.eye(2)
    HT = H.T
    
    # Process each neural observation sequentially
    for t_idx, observation in enumerate(X_test_centered):
        # Predict the next velocity state from the previous state and its uncertainty
        state_pred = A @ state
        P_pred = A @ P @ A.T + Q

        innovation_cov = H @ P_pred @ HT + R # Expected uncertainty in the neural observation
        K = np.linalg.solve(innovation_cov.T, H @ P_pred).T  # Kalman gain
        # Update velocity estimate and its uncertainty
        innovation = observation - H @ state_pred
        state = state_pred + K @ innovation
        P = (I - K @ H) @ P_pred
        predictions[t_idx] = state + y_mean

    return predictions

def kalman_decode(X_train, y_train, X_test):
    """
    Original API wrapper that binds everything together.
    """
    A, Q, y_mean = train_kalman_transition(y_train)
    H, R, x_mean = train_kalman_observation(X_train, y_train)
    return run_kalman_filter(X_test, x_mean, y_mean, A, Q, H, R)

# Evaluate Kalman decoder
kalman_sorted = kalman_decode(X_train, y_train, X_test)
kalman_sorted_r2_x = r2_score(y_test[:, 0], kalman_sorted[:, 0])
kalman_sorted_r2_y = r2_score(y_test[:, 1], kalman_sorted[:, 1])
kalman_sorted_rmse = np.sqrt(mean_squared_error(y_test, kalman_sorted))
print("Optimized Kalman Verification - SORTED UNITS")
print("R2 X:", kalman_sorted_r2_x)
print("R2 Y:", kalman_sorted_r2_y)

def evaluate_kalman_fast(X_train, X_test, y_test, x_mean, y_mean, A, Q, H, R):
    """
    Lightning fast evaluation utilizing pre-computed transition model and shapes.
    """
    pred = run_kalman_filter(X_test, x_mean, y_mean, A, Q, H, R)
    r2_x = r2_score(y_test[:, 0], pred[:, 0])
    r2_y = r2_score(y_test[:, 1], pred[:, 1])
    return {"R2_X": r2_x, "R2_Y": r2_y, "R2_mean": (r2_x + r2_y) / 2}

# Use the first 80% of the recording for training and the final 20% for testing
split = int(0.8 * len(firing_rates))
X_train_full = firing_rates[:split]
X_test_full = firing_rates[split:]
y_train_full = binned_velocity[:split]
y_test_full = binned_velocity[split:]

# Precompute baseline parameters and evaluate full-feature decoder
A_const, Q_const, y_mean_const = train_kalman_transition(y_train_full)
H_full, R_full, x_mean_full = train_kalman_observation(X_train_full, y_train_full)
baseline = evaluate_kalman_fast(X_train_full, X_test_full, y_test_full, x_mean_full, y_mean_const, A_const, Q_const, H_full, R_full)
baseline_r2 = baseline["R2_mean"]

# Parallel/Optimized Leave-One-Out Evaluation
unit_contributions = []
num_features = firing_rates.shape[1]

import time
t0 = time.time()

for unit_idx in range(num_features):
    # Keep all neural features except the current unit
    keep_indices = [i for i in range(num_features) if i != unit_idx]
    X_tr_sub = X_train_full[:, keep_indices]
    X_te_sub = X_test_full[:, keep_indices]
    # Retrain observation model only and evaluate after removing current unit
    H_sub, R_sub, x_mean_sub = train_kalman_observation(X_tr_sub, y_train_full)
    res = evaluate_kalman_fast(X_tr_sub, X_te_sub, y_test_full, x_mean_sub, y_mean_const, A_const, Q_const, H_sub, R_sub)
    unit_contributions.append(baseline_r2 - res["R2_mean"])

unit_contributions = np.array(unit_contributions)
t1 = time.time()  # Ending time of ablation analysis

print(f"Completed leave-one-out analysis for all {num_features} units in {t1 - t0:.2f} seconds!")
print("Baseline Mean R2:", baseline_r2)
print("Max contribution:", np.max(unit_contributions))
print("Min contribution:", np.min(unit_contributions))

# Plot how much each sorted unit contributes to decoding performance
plt.figure(figsize=(10, 4))
# Each bar represents the change in mean R2 when that unit is removed
plt.bar(np.arange(len(unit_contributions)),unit_contributions)
plt.axhline(0, linestyle="--")
plt.xlabel("Sorted Unit")
plt.ylabel("Drop in Mean R2 After Removal")
plt.title("Individual Unit Contribution to Kalman Decoding")
plt.show()

# Convert the stored unit information into a DataFrame and add each unit's decoding contribution
unit_table = pd.DataFrame(sorted_unit_info)
unit_table["decoding_contribution"] = unit_contributions
# Sort units from most helpful to least helpful
unit_table = unit_table.sort_values("decoding_contribution", ascending=False).reset_index(drop=True)
unit_table.head(10)

recording_duration = t[-1] - t[0]  # Total duration of the recording
unit_table["firing_rate_hz"] = (unit_table["n_spikes"] / recording_duration)  # Average firing rate for each sorted unit
unit_table.head()

# Check whether firing rate is related to decoding contribution
corr = unit_table[["firing_rate_hz", "decoding_contribution"]].corr()
print(corr)

# Plot firing rate against decoding contribution
plt.figure(figsize=(6, 5))
plt.scatter(unit_table["firing_rate_hz"], unit_table["decoding_contribution"])
plt.xlabel("Firing Rate (Hz)")
plt.ylabel("Decoding Contribution")
plt.title("Firing Rate vs Decoding Contribution")
plt.show()

# Evaluate any neural feature matrix using the Kalman decoder
def evaluate_kalman(X, y, split_ratio=0.8):
    """
    Evaluates a neural feature matrix using a chronological split and Kalman decoding.
    """
    # Split data chronologically into training and testing sets
    split = int(split_ratio * len(X))
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]
    pred = kalman_decode(X_train, y_train, X_test)  # Run the Kalman decoder

    # Calculate decoding metrics
    r2_x = r2_score(y_test[:, 0], pred[:, 0])
    r2_y = r2_score(y_test[:, 1], pred[:, 1])
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    return {"R2_X": r2_x, "R2_Y": r2_y, "RMSE": rmse, "R2_mean": (r2_x + r2_y) / 2}

# Create Condition D using random subsets of sorted units matched to number of units that pass quality thresholds
def matched_random_condition(firing_rates, n_units, n_repeats=20, seed=42):
    """
    Evaluates random subsets of sorted units matched to a specified feature count.
    """
    rng = np.random.default_rng(seed) # Reproducible random number generator
    results = []

    # Generate and evaluate multiple random subsets
    for repeat in range(n_repeats):
        # Randomly select the requested number of units
        selected = rng.choice(firing_rates.shape[1], size=n_units, replace=False)
        # Build the neural feature matrix using only those units and evaluate this random subset using the Kalman decoder
        X_random = firing_rates[:, selected]
        result = evaluate_kalman(X_random,binned_velocity)
        results.append(result)

    return pd.DataFrame(results)

# Initial example of Condition D before applying the final quality-passing unit count
n_quality_units = 83
random_results = matched_random_condition(firing_rates,n_units=n_quality_units)
random_results.mean()

# Verify that the raw NWB recording is available
if not os.path.exists(raw_file_path):
    raise FileNotFoundError(f"File not found: {raw_file_path}")
print("Using raw file:", raw_file_path)
file_size_bytes = os.path.getsize(raw_file_path)
print(f"Physical file size on disk: {file_size_bytes / (1024**2):.2f} MB")

try:
    # Open the NWB file and inspect the broadband recording
    with h5py.File(raw_file_path, "r") as f:
        broadband = f["/acquisition/timeseries/broadband/data"]
        timestamps = f["/acquisition/timeseries/broadband/timestamps"]
        print("Broadband shape:", broadband.shape)
        print("Timestamp shape:", timestamps.shape)

        # Read the voltage conversion factor
        conversion = broadband.attrs.get("conversion", 1.0)
        print("Voltage conversion factor:", conversion)

        # Estimate sampling frequency from timestamps
        first_times = np.asarray(timestamps[:1000]).squeeze()
        sampling_frequency = 1 / np.mean(np.diff(first_times))
        print("Sampling frequency:", sampling_frequency, "Hz")
        print("Number of channels:", broadband.shape[1])
        
except OSError as e:
    print("\n[ERROR] Failed to open the NWB file due to truncation/corruption.")
    print("Details:", str(e))
    print("Please ensure the file is completely uploaded. Expected size is likely ~1.05 GB (1,045,508,634 bytes), but found only", f"{file_size_bytes:,} bytes.")

# Full timestamp array so Kilosort spike samples can be mapped back to the same absolute time axis as the behavioral data
with h5py.File(raw_file_path, "r") as f:
    raw_timestamps = np.asarray(f["/acquisition/timeseries/broadband/timestamps"]).squeeze()
print("Raw timestamp range:", raw_timestamps[0], "to", raw_timestamps[-1])

# Load a short raw signal segment for visual inspection
with h5py.File(raw_file_path, "r") as f:
    broadband = f["/acquisition/timeseries/broadband/data"]
    conversion = broadband.attrs.get("conversion", 1.0)
    n_samples = int(sampling_frequency * 2)  # Load only 2 seconds from the first channel
    raw_segment = np.asarray(broadband[:n_samples, 0]).squeeze() * conversion
print("Loaded samples:", len(raw_segment))

# Plot the short broadband segment to inspect the raw neural signal
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(raw_segment)) / sampling_frequency, raw_segment)
plt.xlabel("Time (s)")
plt.ylabel("Voltage")
plt.title("Raw Broadband Neural Signal - Channel 1")
plt.show()

sampling_frequency = 24414.0625  # Use the exact sampling frequency of the raw neural recording
# Create a temporary binary file only if it does not already exist
if not os.path.exists(binary_path):
    print("Creating binary file...")

    with h5py.File(raw_file_path, "r") as f:
        broadband = f["/acquisition/timeseries/broadband/data"]
        chunk_size = 100000
        # Write the large NWB recording to disk in smaller chunks
        with open(binary_path, "wb") as out:
            for start in range(0, broadband.shape[0], chunk_size):
                stop = min(start + chunk_size, broadband.shape[0])
                # Read a manageable chunk instead of loading the whole recording and write to binary file
                chunk = np.asarray(broadband[start:stop, :]).astype("float32")
                chunk.tofile(out)
                if start % 1000000 == 0:
                    print(f"Written {start}/{broadband.shape[0]} samples")

    print("Binary file created.")
else:
    print("Binary file already exists.")

# Create a SpikeInterface recording object from the binary file
recording = si.read_binary(file_paths=binary_path, sampling_frequency=sampling_frequency, num_channels=96, dtype="float32")

# Set channel locations required by Kilosort, uses a simple vertical 2D layout with 20 micron spacing
locations = np.zeros((96, 2))
locations[:, 1] = np.arange(96) * 20
recording.set_channel_locations(locations)

# Bandpass filter the broadband signal to isolate spike-related frequencies
recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
print(recording_filtered)

# Run Kilosort4 on the filtered neural recording
sorting = si.run_sorter(
    sorter_name="kilosort4",
    recording=recording_filtered,
    folder=sorting_output_folder,
    remove_existing_folder=True,
    verbose=True
)
print(sorting)
print("Number of sorted units:", len(sorting.unit_ids))

# Create a SortingAnalyzer connecting the recording and sorted units
analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording_filtered, format="memory")
# Compute the information needed for the quality metrics
analyzer.compute("random_spikes")
analyzer.compute("waveforms")
analyzer.compute("templates")
analyzer.compute("noise_levels")
analyzer.compute("spike_amplitudes")

# Compute all quality metrics needed and get the completed quality-metric table
analyzer.compute("quality_metrics", metric_names=["snr", "isi_violation", "amplitude_cutoff", "amplitude_cv"])
quality_metrics = analyzer.get_extension("quality_metrics").get_data()

# Preview the quality metrics used in the analysis
print(quality_metrics[["snr", "isi_violations_ratio", "amplitude_cutoff", "amplitude_cv_median", "amplitude_cv_range"]].head())
print()

# Display summary statistics and create a copy of the quality metrics table
print(quality_metrics[["snr", "isi_violations_ratio", "amplitude_cutoff", "amplitude_cv_median", "amplitude_cv_range"]].describe())
quality_table = quality_metrics.copy()

# Define units that pass our quality criteria
quality_table["passes_quality"] = (
    (quality_table["snr"] >= 3.0) &
    (quality_table["isi_violations_ratio"] <= 0.5) &
    (quality_table["amplitude_cutoff"] <= 0.10) &
    (quality_table["amplitude_cv_median"].notna())
)

# Get the IDs of units that pass the quality criteria
quality_unit_ids = quality_table.index[quality_table["passes_quality"]].to_numpy()
print("Total Kilosort units:", len(quality_table))
print("Quality-passing units:", len(quality_unit_ids))
print("Percent passing:", 100 * len(quality_unit_ids) / len(quality_table))

# Show the quality metrics for the passing units
print(quality_table.loc[quality_unit_ids,
    ["snr", "isi_violations_ratio", "amplitude_cutoff", "amplitude_cv_median"]].describe())

# Create the same 50 ms bins used in the existing decoding pipeline
ks_spike_counts = np.zeros((len(bin_centers), len(sorting.unit_ids)))

# Convert each unit's spike sample indices to absolute timestamps and count the spikes in each 50 ms bin
for unit_idx, unit_id in enumerate(sorting.unit_ids):
    spike_samples = sorting.get_unit_spike_train(unit_id=unit_id)
    spike_times = raw_timestamps[spike_samples]  # Map zero-based sample indices to the NWB timestamp array
    # Count spikes in each 50 ms bin
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    ks_spike_counts[:, unit_idx] = counts

# Convert spike counts into firing rates
ks_firing_rates = ks_spike_counts / bin_size
print("All Kilosort units:", ks_firing_rates.shape)

# CONDITION A: all Kilosort sorted units
X_A = ks_firing_rates

# Find the column corresponding to each quality-passing unit and find the firing-rate columns belonging to quality-passing units
unit_id_to_col = {unit_id: i for i, unit_id in enumerate(sorting.unit_ids)}
quality_cols = [unit_id_to_col[unit_id] for unit_id in quality_unit_ids]

# CONDITION B: only units that pass quality thresholds
X_B = ks_firing_rates[:, quality_cols]
print("Condition A units:", X_A.shape[1])
print("Condition B units:", X_B.shape[1])

# Evaluate A and B using the same Kalman decoder
result_A = evaluate_kalman(X_A, binned_velocity)
result_B = evaluate_kalman(X_B, binned_velocity)

# CONDITION D: random subsets matched to Condition B's unit count
random_results = matched_random_condition(ks_firing_rates, n_units=X_B.shape[1], n_repeats=20)
result_D = random_results.mean()

# Display comparison
comparison = pd.DataFrame({
    "Condition": ["A: All sorted units", "B: Quality-passing units", "D: Matched random subset"],
    "R2 X": [result_A["R2_X"], result_B["R2_X"], result_D["R2_X"]],
    "R2 Y": [result_A["R2_Y"], result_B["R2_Y"], result_D["R2_Y"]],
    "Mean R2": [result_A["R2_mean"], result_B["R2_mean"], result_D["R2_mean"]],
    "RMSE": [result_A["RMSE"], result_B["RMSE"], result_D["RMSE"]]
})

print(comparison)

# Get baseline Kalman performance using all Kilosort units
baseline = evaluate_kalman(ks_firing_rates,binned_velocity)
baseline_r2 = baseline["R2_mean"]
ks_unit_contributions = []  # Store the decoding contribution of each sorted unit

# Remove one unit at a time and measure the drop in decoding performance.
for unit_idx in range(ks_firing_rates.shape[1]):
    X_removed = np.delete(ks_firing_rates, unit_idx, axis=1)
    result = evaluate_kalman(X_removed, binned_velocity)
    # Positive value means the unit helped decoding
    contribution = (baseline_r2 - result["R2_mean"])
    ks_unit_contributions.append(contribution)
    if (unit_idx + 1) % 10 == 0:
        print(f"Done {unit_idx + 1}/"f"{ks_firing_rates.shape[1]} units")

ks_unit_contributions = np.array(ks_unit_contributions)
print("Finished")
print("Baseline Mean R2:", baseline_r2)
print("Largest contribution:",ks_unit_contributions.max())
print("Most harmful contribution:",  ks_unit_contributions.min())

# Add decoding contribution to the quality table
quality_table["decoding_contribution"] = (ks_unit_contributions)
# Keep the metrics needed for the main analysis
analysis_table = quality_table[["snr", "isi_violations_ratio", "amplitude_cutoff", "amplitude_cv_median", "decoding_contribution"]].copy()
# Measure how strongly each quality metric relates to decoding usefulness
correlations = analysis_table.corr()["decoding_contribution"].sort_values(ascending=False)
print(correlations)

# Plot each quality metric against unit decoding contribution
metrics = ["snr", "isi_violations_ratio", "amplitude_cutoff", "amplitude_cv_median"]
for metric in metrics:
    plt.figure(figsize=(6, 4))
    plt.scatter(analysis_table[metric], analysis_table["decoding_contribution"])
    plt.xlabel(metric)
    plt.ylabel("Decoding Contribution")
    plt.title(f"{metric} vs Decoding Contribution")
    plt.show()

# Import SpikeInterface peak detection
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

# CONDITION C: detect unsorted threshold crossings independently on each electrode using the filtered broadband recording
peaks = detect_peaks(
    recording=recording_filtered,
    method="by_channel_torch",
    method_kwargs={"detect_threshold": 5.0,"exclude_sweep_ms": 1.0}
)
print("Total threshold crossings detected:", len(peaks))

# Create one threshold-crossing feature per electrode
n_channels = recording_filtered.get_num_channels()
threshold_counts_raw = np.zeros((len(bin_centers), n_channels))

# Bin threshold crossings separately for each electrode
for channel_idx in range(n_channels):
    channel_peaks = peaks[peaks["channel_index"] == channel_idx]
    # Convert raw sample indices to the same absolute clock as behavior
    spike_times = raw_timestamps[channel_peaks["sample_index"]]
    # Keep only the period where behavioral data exists
    spike_times = spike_times[(spike_times >= t[0]) & (spike_times <= t[-1])]

    # Count threshold crossings in the same 50 ms bins
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    threshold_counts_raw[:, channel_idx] = counts

# Convert counts to firing rates and evaluate Condition C with the same Kalman decoder
X_C = threshold_counts_raw / bin_size
result_C = evaluate_kalman(X_C, binned_velocity)
print("Condition C units/channels:", X_C.shape[1])
print("Condition C:", result_C)

# Calculate the average and variation across Condition D random subsets
D_mean = random_results.mean()
D_std = random_results.std()

# Build the final four-condition within-session results table
abcd_results = pd.DataFrame({
    "Condition": ["A: All sorted units", "B: Quality-passing units", "C: Threshold crossings", "D: Matched random subset"],
    "R2 X": [result_A["R2_X"], result_B["R2_X"], result_C["R2_X"], D_mean["R2_X"]],
    "R2 Y": [result_A["R2_Y"], result_B["R2_Y"], result_C["R2_Y"], D_mean["R2_Y"]],
    "Mean R2": [result_A["R2_mean"], result_B["R2_mean"], result_C["R2_mean"], D_mean["R2_mean"]],
    "RMSE": [result_A["RMSE"], result_B["RMSE"], result_C["RMSE"], D_mean["RMSE"]]
})
print(abcd_results)

# Compare quality-passing units against the random matched subsets
n_random_better = np.sum(random_results["R2_mean"] >= result_B["R2_mean"])
# Empirical randomization p-value
p_random = (n_random_better + 1) / (len(random_results) + 1)

print()
print("Condition D Mean R2:", D_mean["R2_mean"])
print("Condition D R2 standard deviation:", D_std["R2_mean"])
print("Condition B Mean R2:", result_B["R2_mean"])
print("Random subsets performing at least as well as B:", n_random_better, "out of",len(random_results))
print("Empirical p-value:", p_random)

# Plot the four main within-session conditions
plt.figure(figsize=(8, 5))
mean_r2_values = [result_A["R2_mean"], result_B["R2_mean"], result_C["R2_mean"], D_mean["R2_mean"]]
plt.bar(["A: All", "B: Quality", "C: Threshold", "D: Random"], mean_r2_values)
plt.xlabel("Neural Representation")
plt.ylabel("Mean R2")
plt.title("Within-Session BCI Decoding Performance")
plt.show()

# Store the main results from this session
session_result = {
    "session": session_name,
    "n_all_units": X_A.shape[1],
    "n_quality_units": X_B.shape[1],
    "A_mean_R2": result_A["R2_mean"],
    "B_mean_R2": result_B["R2_mean"],
    "C_mean_R2": result_C["R2_mean"],
    "D_mean_R2": D_mean["R2_mean"],
    "A_RMSE": result_A["RMSE"],
    "B_RMSE": result_B["RMSE"],
    "C_RMSE": result_C["RMSE"],
    "D_RMSE": D_mean["RMSE"]
}
# Put the results into a table
session_results = pd.DataFrame([session_result])
print(session_results)
shutil.make_archive(f"/content/kilosort4_output_{session_name}", "zip", f"/content/kilosort4_output_{session_name}")

"""## Cross-Session Decoding Analysis"""
# ============================================================
# RECOVER ONE SESSION AND EXPORT CROSS-SESSION DATA
# Run this once for each session.
# Only change session_name.
# ============================================================
import spikeinterface.full as si
from scipy.interpolate import interp1d
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

# ------------------------------------------------------------
# CHANGE ONLY THIS LINE FOR EACH SESSION
# ------------------------------------------------------------
session_name = "indy_20161025_04"

# Automatically find the matching files
mat_matches = glob.glob(f"/content/*{session_name}*.mat")
nwb_matches = glob.glob(f"/content/*{session_name}*.nwb")
zip_matches = glob.glob(f"/content/*kilosort4_output*{session_name}*.zip")

if len(mat_matches) != 1: raise ValueError(f"Expected 1 MAT file for {session_name}, found {mat_matches}")
if len(nwb_matches) != 1: raise ValueError(f"Expected 1 NWB file for {session_name}, found {nwb_matches}")
if len(zip_matches) != 1:raise ValueError(f"Expected 1 Kilosort ZIP for {session_name}, found {zip_matches}")

# Store the matched file paths for the selected session
mat_path = mat_matches[0]
nwb_path = nwb_matches[0]
zip_path = zip_matches[0]
print("MAT:", mat_path)
print("NWB:", nwb_path)
print("Kilosort ZIP:", zip_path)

# ============================================================
# 1. LOAD BEHAVIOR
# ============================================================

with h5py.File(mat_path, "r") as f:
    t = np.asarray(f["t"]).squeeze()
    cursor_pos = np.asarray(f["cursor_pos"]).T

# Calculate X/Y cursor velocity
dt = np.mean(np.diff(t))
velocity_x = np.gradient(cursor_pos[:, 0], dt)
velocity_y = np.gradient(cursor_pos[:, 1], dt)
velocity = np.column_stack((velocity_x, velocity_y))

# Use same 50 ms bins as within-session analysis
bin_size = 0.050
bin_edges = np.arange(t[0], t[-1] + bin_size, bin_size)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Interpolate cursor velocity into the 50 ms bins
vx_interp = interp1d(t, velocity[:, 0], bounds_error=False, fill_value="extrapolate")
vy_interp = interp1d(t, velocity[:, 1], bounds_error=False, fill_value="extrapolate")
binned_velocity = np.column_stack([vx_interp(bin_centers), vy_interp(bin_centers)])

# ============================================================
# 2. LOAD RAW NWB INFORMATION
# ============================================================

with h5py.File(nwb_path, "r") as f:
    broadband = f["/acquisition/timeseries/broadband/data"]
    # Load the absolute recording timestamps used to align spikes with behavior
    raw_timestamps = np.asarray(f["/acquisition/timeseries/broadband/timestamps"]).squeeze()
    # Use the sampling frequency associated with these recordings
    sampling_frequency = 24414.0625
    # Load the physical electrode coordinates from the NWB file
    locations = np.asarray(f["/general/extracellular_ephys/electrode_map"])[:, :2]
    n_channels = broadband.shape[1]
print("Sampling frequency:", sampling_frequency)
print("Channels:", n_channels)

# Convert electrode positions to micrometers if stored in meters
if np.max(np.abs(locations)) < 1:
    locations = locations * 1e6

# ============================================================
# 3. CREATE TEMPORARY BINARY RECORDING
# ============================================================

binary_path = (f"/content/{session_name}_cross_temp.dat")
# Create the temporary binary only if it does not already exist
if not os.path.exists(binary_path):
    print("Creating temporary binary file...")
    with h5py.File(nwb_path, "r") as f:
        broadband = f["/acquisition/timeseries/broadband/data"]
        chunk_size = 100000
        # Write the recording in chunks to avoid loading the entire NWB file into memory
        with open(binary_path, "wb") as out:
            for start in range(0, broadband.shape[0], chunk_size):
                stop = min(start + chunk_size,broadband.shape[0])
                chunk = np.asarray(broadband[start:stop, :]).astype("float32")
                chunk.tofile(out)
    print("Binary created.")
else:
    print("Using existing temporary binary.")

# Recreate the SpikeInterface recording object from the temporary binary
recording = si.read_binary(
    file_paths=binary_path,
    sampling_frequency=sampling_frequency,
    num_channels=n_channels,
    dtype="float32"
)
# Apply the real electrode geometry loaded from the NWB file
recording.set_channel_locations(locations)
# Apply the same 300-6000 Hz spike bandpass filter used previously
recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)

# ============================================================
# 4. LOAD EXISTING KILOSORT RESULTS
# ============================================================

# Extract the saved Kilosort results if they have not already been recovered
extract_folder = (f"/content/{session_name}_ks_recovered")
if not os.path.exists(extract_folder):
    shutil.unpack_archive(zip_path, extract_folder)

# Find the native Kilosort folder automatically
spike_time_files = glob.glob(
    os.path.join(extract_folder, "**", "spike_times.npy"),recursive=True)

if not spike_time_files:
    raise FileNotFoundError("Could not find spike_times.npy inside Kilosort ZIP.")

kilosort_native_folder = os.path.dirname(spike_time_files[0])
print("Loading existing Kilosort results from:",kilosort_native_folder)
sorting = si.read_kilosort(kilosort_native_folder)
print("Recovered Kilosort units:", len(sorting.unit_ids))

# ============================================================
# 5. RECOMPUTE QUALITY METRICS
# ============================================================
# Reconnect the recovered sorting to the filtered recording so waveform-based quality metrics can be recomputed
analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording_filtered, format="memory")
# Compute the intermediate quantities required for the quality metrics
analyzer.compute("random_spikes")
analyzer.compute("waveforms")
analyzer.compute("templates")
analyzer.compute("noise_levels")
analyzer.compute("spike_amplitudes")

# Recompute the same quality metrics used in the within-session analysis.
analyzer.compute("quality_metrics", metric_names=["snr", "isi_violation", "amplitude_cutoff", "amplitude_cv"])
quality_table = analyzer.get_extension("quality_metrics").get_data()
# Apply the same quality criteria used for Condition B
quality_table["passes_quality"] = (
    (quality_table["snr"] >= 3.0) &
    (quality_table["isi_violations_ratio"] <= 0.5) &
    (quality_table["amplitude_cutoff"] <= 0.10) &
    (quality_table["amplitude_cv_median"].notna())
)
# Store the IDs of units that pass all quality criteria
quality_unit_ids = quality_table.index[quality_table["passes_quality"]].to_numpy()
print("Quality-passing units:",len(quality_unit_ids))

# ============================================================
# 6. DETERMINE MAIN ELECTRODE OF EACH SORTED UNIT
# ============================================================
# Load the average spike templates for all sorted units
templates = analyzer.get_extension("templates").get_data()
# Templates shape: units x samples x channels
unit_main_channels = {}

# Assign each sorted unit to the electrode where its waveform has the largest absolute amplitude
for unit_index, unit_id in enumerate(sorting.unit_ids):
    template = templates[unit_index]
    # Find electrode with largest absolute waveform and use that as main channel
    channel_amplitudes = np.max(np.abs(template), axis=0)
    main_channel = int(np.argmax(channel_amplitudes))
    unit_main_channels[unit_id] = main_channel

# ============================================================
# 7. CONDITION A
# ALL SORTED UNITS POOLED BY ELECTRODE
# ============================================================

# Create one spike-count feature for each electrode
A_counts = np.zeros((len(bin_centers), n_channels))

# Pool every sorted unit onto its assigned main electrode
for unit_id in sorting.unit_ids:
    channel = unit_main_channels[unit_id]
    spike_samples = (sorting.get_unit_spike_train(unit_id=unit_id))
    spike_times = raw_timestamps[spike_samples]  # Convert Kilosort sample indices to absolute timestamps
    counts, _ = np.histogram(spike_times, bins=bin_edges)  # Count the unit's spikes in each 50 ms bin
    A_counts[:, channel] += counts  # Add the unit's activity to its main electrode
A_electrode = A_counts / bin_size

# ============================================================
# 8. CONDITION B
# QUALITY-PASSING SORTED UNITS BY ELECTRODE
# ============================================================

# Create the same 96-electrode representation using only units that passed the quality criteria
B_counts = np.zeros((len(bin_centers), n_channels))

for unit_id in quality_unit_ids:
    channel = unit_main_channels[unit_id]
    spike_samples = (sorting.get_unit_spike_train(unit_id=unit_id))
    spike_times = raw_timestamps[spike_samples]  # Convert sample indices to absolute timestamps
    counts, _ = np.histogram(spike_times,bins=bin_edges)  # Count spikes in the same 50 ms bins
    B_counts[:, channel] += counts  # Pool quality-passing units onto their main electrodes

B_electrode = B_counts / bin_size

# ============================================================
# 9. CONDITION C
# RAW THRESHOLD CROSSINGS
# ============================================================

# Detect threshold-crossing events directly from the filtered broadband recording without performing spike sorting
peaks = detect_peaks(
    recording=recording_filtered,
    method="by_channel_torch",
    method_kwargs={"detect_threshold": 5.0, "exclude_sweep_ms": 1.0}
)
# Create one threshold-crossing feature for each electrode
C_counts = np.zeros((len(bin_centers), n_channels))

for channel_idx in range(n_channels):
    # Keep only peaks detected on the current electrode and convert sample indices to absolute timestamps
    channel_peaks = peaks[peaks["channel_index"] ==channel_idx]
    spike_times = raw_timestamps[ channel_peaks["sample_index"]]  
    
    # Keep only events that occur during the behavioral recording
    spike_times = spike_times[(spike_times >= t[0]) &(spike_times <= t[-1])]
    counts, _ = np.histogram(spike_times, bins=bin_edges)  # Count threshold crossings in the same 50 ms bins
    C_counts[:,channel_idx] = counts
C_electrode = C_counts / bin_size

# ============================================================
# 10. SAVE SMALL CROSS-SESSION FILE
# ============================================================

print()
print("A shape:", A_electrode.shape)
print("B shape:", B_electrode.shape)
print("C shape:", C_electrode.shape)
print("Velocity shape:", binned_velocity.shape)

# Save the session's cross-session features in a compact NPZ file
export_path = (f"/content/" f"{session_name}_cross_session.npz")
np.savez_compressed(export_path, session=session_name, A=A_electrode, B=B_electrode, C=C_electrode, velocity=binned_velocity)
print()
print("SAVED:", export_path)

# ============================================================
# CROSS-SESSION DECODING ANALYSIS
# Run once after all cross_session.npz files are in /content
# ============================================================
from datetime import datetime

# ------------------------------------------------------------
# Kalman filter functions
# ------------------------------------------------------------

def train_kalman_transition(y_train):
    """
    Learns the velocity state-transition model from the training session.
    """
    # Center velocity data
    y_mean = y_train.mean(axis=0)
    Yc = y_train - y_mean
    # Learn how velocity changes from one 50 ms bin to the next
    Y_prev = Yc[:-1]
    Y_next = Yc[1:]
    A = np.linalg.lstsq(Y_prev, Y_next, rcond=None)[0].T
    # Estimate state-transition noise
    residuals = Y_next - Y_prev @ A.T
    Q = np.cov(residuals, rowvar=False)
    Q += np.eye(2) * 1e-6
    return A, Q, y_mean

def train_kalman_observation(X_train, y_train):
    """
    Learns the relationship between neural activity and movement velocity.
    """
    # Center neural activity and velocity
    x_mean = X_train.mean(axis=0)
    y_mean = y_train.mean(axis=0)
    Xc = X_train - x_mean
    Yc = y_train - y_mean
    # Learn the relationship between velocity and neural activity
    H = np.linalg.lstsq(Yc, Xc, rcond=None)[0].T
    # Estimate observation noise
    residuals = Xc - Yc @ H.T
    R = np.cov(residuals, rowvar=False)
    R += np.eye(R.shape[0]) * 1e-3
    return H, R, x_mean

def run_kalman_filter(X_test, x_mean, y_mean, A, Q, H, R):
    """
    Runs the trained Kalman filter sequentially on test-session neural activity.
    """
    # Center test neural activity using the TRAINING session mean
    Xc = X_test - x_mean
    # Initialize the two-dimensional velocity state and covariance
    state = np.zeros(2)
    P = np.eye(2)
    # Store one X/Y velocity prediction for every 50 ms bin
    predictions = np.zeros((len(X_test), 2))
    I = np.eye(2)
    HT = H.T

    # Decode one 50 ms bin at a time
    for i, observation in enumerate(Xc):
        # Predict next velocity state
        state_pred = A @ state
        P_pred = A @ P @ A.T + Q

        # Update prediction using current neural observation
        innovation_cov = H @ P_pred @ HT + R
        K = np.linalg.solve(innovation_cov.T, H @ P_pred).T
        innovation = observation - H @ state_pred

        state = state_pred + K @ innovation
        P = (I - K @ H) @ P_pred
        predictions[i] = state + y_mean

    return predictions


def evaluate_cross_session(X_train, y_train, X_test, y_test):
    """
    Trains a Kalman decoder on one session and evaluates it on another session.
    """
    # Train the full decoder only on the earlier session
    A, Q, y_mean = train_kalman_transition(y_train)
    H, R, x_mean = train_kalman_observation(X_train, y_train)
    
    # Test the trained decoder on a different session
    pred = run_kalman_filter(X_test, x_mean, y_mean, A, Q, H, R)
    # Calculate cross-session decoding performance
    r2_x = r2_score(y_test[:, 0], pred[:, 0])
    r2_y = r2_score(y_test[:, 1], pred[:, 1])
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return {"R2_X": r2_x, "R2_Y": r2_y, "Mean_R2": (r2_x + r2_y) / 2, "RMSE": rmse}

# ------------------------------------------------------------
# Load all saved session files
# ------------------------------------------------------------

# Find all previously exported cross-session feature files
session_files = sorted(glob.glob("/content/*_cross_session.npz"))
print("Cross-session files found:", len(session_files))
# At least two sessions are required for a cross-session comparison
if len(session_files) < 2:
    raise ValueError("Need at least two cross_session.npz files.")

sessions = []
# Load each session's saved A, B, C, and velocity arrays
for path in session_files:
    data = np.load(path, allow_pickle=True)
    # Extract the recording date from the session name
    name = str(data["session"])
    date_text = name.split("_")[1]
    date = datetime.strptime(date_text, "%Y%m%d")

    sessions.append({"name": name, "date": date, "A": data["A"], "B": data["B"], "C": data["C"], "velocity": data["velocity"]})

# Put sessions in chronological order
sessions = sorted(sessions, key=lambda s: s["date"])

# Check that every condition has 96 electrode features
for s in sessions:
    print(s["name"], "A:", s["A"].shape, "B:", s["B"].shape, "C:", s["C"].shape, "Velocity:", s["velocity"].shape)
    if s["A"].shape[1] != 96 or s["B"].shape[1] != 96 or s["C"].shape[1] != 96:
        raise ValueError(f"{s['name']} does not have 96 electrode features.")

# ------------------------------------------------------------
# Run every earlier-session -> later-session comparison
# ------------------------------------------------------------

results = []  # Store one result for each session pair and neural representation

for train_idx in range(len(sessions)):
    for test_idx in range(train_idx + 1, len(sessions)):
        train = sessions[train_idx]
        test = sessions[test_idx]
        # Number of days separating the training and testing recordings
        days_apart = (test["date"] - train["date"]).days

        # Test all three neural representations
        for condition in ["A", "B", "C"]:
            result = evaluate_cross_session(train[condition], train["velocity"], test[condition], test["velocity"])
            results.append({
                "Train Session": train["name"],
                "Test Session": test["name"],
                "Days Apart": days_apart,
                "Condition": condition,
                "R2 X": result["R2_X"],
                "R2 Y": result["R2_Y"],
                "Mean R2": result["Mean_R2"],
                "RMSE": result["RMSE"]
            })
cross_results = pd.DataFrame(results)
print("\nCROSS-SESSION RESULTS")
print(cross_results)

# ------------------------------------------------------------
# Average performance for each condition
# ------------------------------------------------------------

# Summarize Mean R2 across all train-to-test session pairs
cross_summary = cross_results.groupby("Condition")["Mean R2"].agg(["mean", "std", "count"])
print("\nAVERAGE CROSS-SESSION PERFORMANCE")
print(cross_summary)

# ------------------------------------------------------------
# Plot performance versus days between sessions
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
# Plot one line for each neural representation
for condition in ["A", "B", "C"]:
    subset = cross_results[cross_results["Condition"] == condition].sort_values("Days Apart")
    plt.plot(subset["Days Apart"], subset["Mean R2"], marker="o", label=condition)

plt.axhline(0, linestyle="--")
plt.xlabel("Days Between Training and Test Session")
plt.ylabel("Cross-Session Mean R²")
plt.title("Cross-Session Decoding Stability")
plt.legend(title="A = All Sorted, B = Quality, C = Threshold")
plt.show()