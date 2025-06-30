import sys

import numpy as np

sys.path.append(
    "G:\Mi unidad\WORKING_MEMORY\EXPERIMENTS\ELECTROPHYSIOLOGY\ANALYSIS\functions"
)


def compute_window(data, runningwindow, option):
    """
    Computes a rolling average with a length of runningwindow samples.
    """
    performance = []
    end = False
    for i in range(len(data)):
        if data["trial"].iloc[i] <= runningwindow:
            # Store the first index of that session
            if end == False:
                start = i
                end = True
            performance.append(round(np.mean(data[option].iloc[start : i + 1]), 2))
        else:
            end = False
            performance.append(
                round(np.mean(data[option].iloc[i - runningwindow : i]), 2)
            )
    return performance


def compute_window_centered(data, runningwindow, option):
    """
    Computes a rolling average with a length of runningwindow samples.
    """
    performance = []
    start_on = False
    for i in range(len(data)):
        if data["trial"].iloc[i] <= int(runningwindow / 2):
            # Store the first index of that session for the first initial trials
            if start_on == False:
                start = i
                start_on = True
            performance.append(
                round(np.mean(data[option].iloc[start : i + int(runningwindow / 2)]), 2)
            )
        elif i < (len(data) - runningwindow):
            if data["trial"].iloc[i] > data["trial"].iloc[i + runningwindow]:
                # Store the last values for the end of the session
                if end == True:
                    end_value = i + runningwindow - 1
                    end = False
                performance.append(round(np.mean(data[option].iloc[i:end_value]), 2))

            else:  # Rest of the session
                start_on = False
                end = True
                performance.append(
                    round(
                        np.mean(
                            data[option].iloc[
                                i - int(runningwindow / 2) : i + int(runningwindow / 2)
                            ]
                        ),
                        2,
                    )
                )

        else:
            performance.append(round(np.mean(data[option].iloc[i : len(data)]), 2))
    return performance
