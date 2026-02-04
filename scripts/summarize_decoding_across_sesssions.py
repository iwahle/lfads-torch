import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(session_ids: list):
    # load csv decoding results for each session
    decoding_accs = []
    for loo_idx, session_id in enumerate(session_ids):
        try:
            csv_path = (
                f"/usr/people/iwahle/lfads-torch/figures/{date}/"
                + f"loo_{loo_idx}/session_{session_id}_decoding_results.csv"
            )
            df = pd.read_csv(csv_path)
            decoding_accs.append(df["accuracy"].values[0])
        except Exception:
            print(f"No decoding results found for session {session_id}")
            decoding_accs.append(np.nan)
            continue
    print(decoding_accs)
    decoding_accs = np.array(decoding_accs)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(session_ids)), decoding_accs)
    ax.set_xticks(range(len(session_ids)))
    ax.set_xticklabels(session_ids)
    ax.set_xlabel("Session ID")
    ax.set_ylabel("Decoding Accuracy")
    ax.set_title(
        "Train on k-1 sessions, learn readin/out weights on first half of "
        + "kth session, test clf on second half of kth session"
    )
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    plt.show()
    save_path = os.path.join(
        "/usr/people/iwahle/lfads-torch/figures",
        date,
        "decoding_accuracy_across_sessions.png",
    )
    plt.savefig(save_path)

    print(f"Saved decoding accuracy plot to {save_path}")


if __name__ == "__main__":
    session_ids = list(range(21, 33))
    date = "260201"
    main(session_ids)
