import numpy as np
from scipy.signal import lfilter
from scipy.signal.windows import gaussian

from lfads_torch.external_utils.utils import load_format_data, subselect_trials


def fact_to_conj_conds(tid, color, shape):
    """Convert task factors to conjunction conditions (0-15)."""
    conj_conds = np.zeros((tid.shape[0]))
    cnt = 0
    for tid_i in range(1, 5):
        for color_i in range(1, 3):
            for shape_i in range(1, 3):
                mask = (tid == tid_i) & (color == color_i) & (shape == shape_i)
                conj_conds[mask] = cnt
                cnt += 1
    return conj_conds


def load_session(
    data_path: str,
    session_id: str,
    system: str = "ripple",
    session_half: str = "full",
):
    """
    Load session data with optional half-session selection.

    Parameters
    ----------
    data_path : str
        Path to raw data directory
    session_id : str
        Session identifier
    system : str
        Recording system type
    session_half : str
        Which portion of session to load:
        - "full": all blocks (1-8)
        - "first": first half only (blocks 1-4) - used for training LOO session
        - "second": second half only (blocks 5-8) - used for testing LOO session

    Returns
    -------
    tuple
        (spikes, conds, data) or None if no data
    """
    bin_width_sec = 0.02
    data = load_format_data(data_path, session_id, system=system, full_trial=True)
    if data is None:
        print(f"    Session {session_id}: NO DATA - skipping")
        return None

    # remove stim trials
    data = subselect_trials(data, non_stim_trials_only=True)

    # Apply half-session selection
    if session_half == "first":
        data = subselect_trials(data, blockids=range(1, 5))
    elif session_half == "second":
        data = subselect_trials(data, blockids=range(5, 9))
    elif session_half != "full":
        raise ValueError(
            f"Invalid session_half: {session_half}. Must be 'full', 'first', 'second'"
        )

    spikes = data["fr"] * bin_width_sec
    tid_session = data["tid"]
    color_session = data["color"]
    shape_session = data["shape"]
    conds = fact_to_conj_conds(tid_session, color_session, shape_session)
    return spikes, conds, data


def remove_nan_trials(spikes: dict, conds: dict):
    """Remove trials with missing values."""
    print("\n  Removing NaN trials:")
    total_removed = 0

    for session in list(spikes.keys()):
        n_before = len(spikes[session])
        valid_trials = ~np.isnan(spikes[session]).any(axis=(1, 2))
        n_removed = n_before - np.sum(valid_trials)
        total_removed += n_removed

        spikes[session] = spikes[session][valid_trials]
        conds[session] = conds[session][valid_trials]

        if n_removed > 0:
            print(
                f"    Session {session}: removed {n_removed} NaN trials "
                f"({n_before} -> {len(spikes[session])})"
            )

    if total_removed == 0:
        print("    No NaN trials found in any session")
    else:
        print(f"    Total removed: {total_removed} trials")

    return spikes, conds


def smooth_spikes_session(
    spikes: np.ndarray, bin_width_sec: float = 0.02, std_sec: float = 0.02
):
    std = std_sec / bin_width_sec
    window = gaussian(int(std * 3 * 2), std)
    window /= window.sum()
    invalid_len = len(window) - 1

    print("\n  Gaussian smoothing parameters:")
    print(f"    Bin width: {bin_width_sec*1000:.1f} ms")
    print(f"    Std: {std_sec*1000:.1f} ms ({std:.1f} bins)")
    print(f"    Window length: {len(window)} bins")
    print(f"    Trimming first {invalid_len} bins (edge effects)")

    original_time = spikes.shape[1]
    smth_spikes = lfilter(window, 1, spikes, axis=1)[:, invalid_len:, :]
    new_time = smth_spikes.shape[1]
    print(f"    Time bins {original_time} -> {new_time}")
    return smth_spikes


def merge_train_valid(train_data, valid_data, train_inds, valid_inds):
    n_samples = len(train_data) + len(valid_data)
    merged_data = np.full((n_samples, *train_data.shape[1:]), np.nan)
    merged_data[train_inds] = train_data
    merged_data[valid_inds] = valid_data
    return merged_data
