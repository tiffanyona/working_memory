
import os
from pathlib import Path
import re
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
pd.options.mode.chained_assignment = None  # Ignore SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import random

from matplotlib.animation import FFMpegWriter
from matplotlib import animation

class Extractor:
    def __init__(self, *, subject, session, h5, csv, video, timestamp, random_state=123, **kws):
        """
        Initializes the Extractor with paths and settings, loads and parses all required data,
        and determines whether the session should be skipped based on log likelihood.
        """
        
        self.subject = subject
        self.session = session
        self.csv = csv
        self.h5 = h5
        self.video = Path(video)
        self.timestamp = Path(timestamp)
        self.rs = random_state
        self.rng = np.random.default_rng(random_state)
        self.parse_tvec()
        self.parse_csv()
        self.load_h5_bodyparts()
        # self.extract_frame()
        
        # Check average log likelihood and decide whether to continue
        avg_ll = self.average_log_likelihood()
        threshold = -1.5  # adjust threshold as needed
        if avg_ll is None or avg_ll < threshold:
            print(f"Skipping {self.subject} {self.session} due to low average log likelihood ({avg_ll})")
            self.skip = True
        else:
            self.skip = False
            self.align_frames_with_behavior()
            
    def parse_tvec(self):
        """
        Loads the timestamp vector (tvec) from a .npz file.
        """
        try:
            with np.load(self.timestamp) as d:
                self.tvec = d['arr_0']
        except TypeError:
            self.tvec = np.load(self.timestamp)
            
    def parse_csv(self):
        """
        Parses the behavioral CSV file, aligns time with frame indices, and selects a random trial
        and a representative frame within that trial.
        """
        
        tvec = self.tvec
        
        self.df = pd.read_csv(self.csv, sep=';', skiprows=6).rename(columns={
            "TYPE": "type",
            "PC-TIME": "pc_time",
            "BPOD-INITIAL-TIME": "bpod_initial_time",
            "BPOD-FINAL-TIME": "bpod_final_time",
            "MSG": "msg",
            "+INFO": "info_",
        }).astype({
            "type": "category",
            "bpod_initial_time": "float32",
            "bpod_final_time": "float32",
        })
        
        self.df["pc_time"] = pd.to_datetime(self.df["pc_time"])

        self.df.loc[self.df.msg == 'DELAY', 'trial_idx'] = np.arange((self.df.msg == 'New trial').sum())
        self.df.trial_idx.fillna(method='ffill', inplace=True)
        self.df.trial_idx.fillna(-1, inplace=True)
        self.df.trial_idx = self.df.trial_idx.astype(int)

        self.df['frame_idx'] = -1
        types_to_sort = ['TRIAL', 'EVENT', 'TRANSITION', 'SOFTCODE', 'stdout', 'END-TRIAL','VAL']
        mask = self.df.type.isin(types_to_sort)
        self.df.loc[mask, 'frame_idx'] = np.searchsorted(tvec, self.df.loc[mask, 'pc_time'].values)

        base = pd.DataFrame({"trial_idx": np.sort(self.df.trial_idx.dropna().unique())})

        s = self.df.query("type == 'TRANSITION' and msg == 'ResponseWindow'")[['frame_idx', 'trial_idx']] \
            .groupby('trial_idx')['frame_idx'].min().reset_index().set_axis(['trial_idx', "s"], axis=1)

        e = self.df.loc[
            (self.df.type == 'TRANSITION') & (self.df.msg.isin(['Reward', 'Punish', 'Miss'])),
            ['frame_idx', 'trial_idx']
        ].groupby('trial_idx')['frame_idx'].max().reset_index().set_axis(['trial_idx', "e"], axis=1)

        tempos = base.merge(s, on="trial_idx", how="left").merge(e, on="trial_idx", how="left").dropna()

        chosen_trial = self.rng.choice(tempos.trial_idx.values)
        self.chosen_trial = chosen_trial
        start_frame = int(tempos.loc[tempos.trial_idx == chosen_trial, 's'].item())
        start_time = tvec[start_frame]
        target_time = start_time + np.timedelta64(250, 'ms')
        target_frame = np.searchsorted(tvec, target_time)
        if target_frame >= len(tvec):
            target_frame = len(tvec) - 1
        self.chosen_frame = int(target_frame)

    def load_h5_bodyparts(self):
        """
        Loads DeepLabCut tracking data from an HDF5 file.
        """
        self.dlc_df = pd.read_hdf(self.h5)
        
    def average_log_likelihood(self):
        """
        Computes the average log-likelihood across all body parts and frames.
        Returns:
            float or None: Log-likelihood value or None if not available.
        """
        
        scorer = self.dlc_df.columns.get_level_values(0)[0]
        bodyparts = self.dlc_df.columns.get_level_values(1).unique()

        likelihoods = []
        for bp in bodyparts:
            try:
                likelihoods.append(self.dlc_df[(scorer, bp, 'likelihood')].values)
            except KeyError:
                pass

        if len(likelihoods) == 0:
            print("No likelihood columns found in DLC data.")
            return None
        
        all_likelihoods = np.concatenate(likelihoods)
        clipped = np.clip(all_likelihoods, 1e-10, 1.0)
        avg_log_likelihood = np.mean(np.log(clipped))
        print(f"Average log likelihood for {self.subject} {self.session}: {avg_log_likelihood}")
        return avg_log_likelihood

    def extract_frame(self):
        outdir = Path("./extra_frames") / self.video.stem
        outdir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(self.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.chosen_frame)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {self.chosen_frame} from {self.video}")
            return

        try:
            coords = self.dlc_df.loc[self.chosen_frame]
            scorer = coords.index.get_level_values(0)[0]
            bodyparts = coords.index.get_level_values(1).unique()
            for bp in bodyparts:
                x = coords[(scorer, bp, 'x')]
                y = coords[(scorer, bp, 'y')]
                if not pd.isna(x) and not pd.isna(y):
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(frame, bp, (int(x) + 5, int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as ex:
            print(f"Could not overlay body parts: {ex}")

        filename = f"{self.subject}_{self.session}_trial{self.chosen_trial}_frame{self.chosen_frame:06}.png"
        save_path = outdir / filename
        cv2.imwrite(str(save_path), frame)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f"{self.subject} {self.session} trial {self.chosen_trial} frame {self.chosen_frame}")
        plt.axis('off')
        plt.show()

    def align_frames_with_behavior(self):
        if not hasattr(self, 'tvec') or not hasattr(self, 'df') or not hasattr(self, 'dlc_df'):
            raise ValueError("Need to run parse_tvec, parse_csv, and load_h5_bodyparts first.")
        
        self.df['pc_time'] = pd.to_datetime(self.df['pc_time'])
        self.df = self.df.sort_values('pc_time')

        bpod_times = self.df['pc_time'].values
        msgs = self.df['msg'].values

        indices = np.searchsorted(bpod_times, self.tvec, side='right') - 1
        indices = np.clip(indices, 0, len(bpod_times) - 1)

        frame_df = pd.DataFrame({
            'frame': np.arange(len(self.tvec)),
            'frame_time': self.tvec
        })

        scorer = self.dlc_df.columns.get_level_values(0)[0]
        bodyparts = self.dlc_df.columns.get_level_values(1).unique()

        for bp in bodyparts:
            frame_df[f'{bp}_x'] = self.dlc_df[(scorer, bp, 'x')]
            frame_df[f'{bp}_y'] = self.dlc_df[(scorer, bp, 'y')]
            frame_df[f'{bp}_log'] = self.dlc_df[(scorer, bp, 'likelihood')]
            
        self.frame_behavior_df = frame_df
        
    def merge_behavior_and_pose(self):
        """
        Merges pose tracking and behavioral logs into a single DataFrame.
        Adds processed metadata: outcome, side, delays, hand positions, etc.
        
        Returns:
            pd.DataFrame: Merged frame-behavior DataFrame.
        """
        df1 = self.frame_behavior_df
        df2 = self.df.rename(columns={'frame_idx': 'frame', 'trial_idx':'trial'})
        merged = pd.merge(df1, df2, on='frame', how='left')
        merged['session'] = self.session
        merged['subject'] = self.subject
        
        merged = merged.loc[merged['type'] != 'SOFTCODE']  # Remove SOFTCODE type if it exists

        merged['msg'] = merged['msg'].ffill()
        merged['trial'] = merged['trial'].ffill()

        # Create a new column extracting the number from rows that contain 'Side:'
        merged['reward_side'] = merged['msg'].str.extract(r'Side:\s*(\d+)')  # extracts the number
        merged['reward_side'] = merged['reward_side'].ffill()
        merged['reward_side'] = merged['reward_side'].astype('float')  # convert to float or int if you prefer
        
        #Fixing issue with the trial element happening before side assigment.
        majority = merged.groupby('trial').agg(
        majority_reward_side=('reward_side', lambda x: x.value_counts().idxmax() if not x.dropna().empty else np.nan)
        )
        merged.drop(columns=['reward_side'], inplace=True, errors='ignore')
        merged = pd.merge(merged, majority, on='trial', how='left')
        merged.rename(columns={'majority_reward_side': 'reward_side'}, inplace=True)

        # Create a new column extracting the number from rows that contain 'Side:'
        merged['delay'] = np.where(merged['msg'] == 'DELAY', merged['info_'], np.nan)  # extracts the number
        # Forward fill the side column
        merged['delay'] = merged['delay'].ffill()
        merged['delay'] = merged['delay'].astype('float')  # convert to float or int if you prefer

        outcome = merged.loc[(merged.msg == 'Reward')|(merged.msg == 'TimeOut')|(merged.msg == 'Miss')|(merged.msg == 'AW')][['msg', 'trial']]
        outcome['outcome'] = outcome['msg'].map({'Reward': 1, 'TimeOut': 0})
        outcome['valid'] = outcome['msg'].map({'Reward': 1, 'TimeOut': 1, 'Miss': 0})
        outcome['AW'] = outcome['msg'].map({'AW': 1})
        outcome = outcome.groupby('trial').agg({'outcome': 'first', 'valid':'mean', 'AW':'first'}).reset_index()

        merged = pd.merge(merged, outcome[['outcome', 'valid', 'AW','trial']], on='trial', how='left')

        merged['vector_answer'] = np.where(merged['outcome'] == 1, merged['reward_side'], 1 - merged['reward_side'])

        aw_trial = float(merged.loc[merged.msg == 'VAR_AW']['info_'].iloc[0])
        merged['AW'] = np.where(merged['trial'] <= aw_trial, 1, 0)
        
        selected_cols = ['RH1_x', 'RH2_x', 'RH3_x',
                'RH4_x', 'RH5_x','RWrist_x']
        merged['mean_R_x'] = merged[selected_cols].mean(axis=1)

        selected_cols = ['RH1_y', 'RH2_y', 'RH3_y',
                'RH4_y', 'RH5_y','RWrist_y']
        merged['mean_R_y'] = merged[selected_cols].mean(axis=1)

        selected_cols = ['LH1_x', 'LH2_x', 'LH3_x',
                'LH4_x', 'LH5_x','LWrist_x']
        merged['mean_L_x'] = merged[selected_cols].mean(axis=1)

        selected_cols = ['LH1_y', 'LH2_y', 'LH3_y',
                'LH4_y', 'LH5_y','LWrist_y']
        merged['mean_L_y'] = merged[selected_cols].mean(axis=1)
        
        selected_cols = ['LH1_log', 'LH2_log', 'LH3_log',
                'LH4_log', 'LH5_log','LWrist_log']
        merged['mean_L_log'] = merged[selected_cols].mean(axis=1)
        
        selected_cols = ['RH1_log', 'RH2_log', 'RH3_log',
                'RH4_log', 'RH5_log','RWrist_log']
        merged['mean_R_log'] = merged[selected_cols].mean(axis=1)
        
        # For each trial, mark the first occurrence of "New trial"
        first_idx = merged.groupby('trial').head(1).index

        merged.loc[first_idx, 'msg'] = 'New trial'
        merged.loc[first_idx, 'bpod_initial_time'] = 0

        # Interpolate bpod times using the frame rate
        merged.set_index('frame_time', inplace=True)
        merged['bpod_time_interp'] = merged['bpod_initial_time'].interpolate(method="time")
        merged.reset_index(inplace=True)

        # Create aligment timestamps for each trial
        for event in ['ResponseWindow', 'StimulusTrigger', 'Motor_out', 'Motor_in', 'Delay']:
            # Compute mean only from ResponseWindow rows
            means = (
                merged.loc[merged.msg == event]
                    .groupby(["trial"])["bpod_time_interp"]
                    .first()
                    .rename(event)
        )

            # Map back to all rows of merged
            merged[event] = merged.set_index(["trial"]).index.map(means)

        merged['Delay'] = merged['Delay'].fillna(merged["Motor_in"])

        for event in ['ResponseWindow', 'StimulusTrigger', 'Motor_out', 'Motor_in', 'Delay']:
            # Compute mean only from ResponseWindow rows
            merged['a_' + event] = merged['bpod_time_interp'] - merged[event]
            
        return merged
   
   
from matplotlib.animation import FFMpegWriter

def create_video_trial(merged: pd.DataFrame, chosen_trial: int = 1, variable: str = "TipTongue"):
    """
    Generates a video of the chosen trial with overlaid tracking for a given variable (e.g., TipTongue).
    
    Args:
        merged (pd.DataFrame): Merged pose + behavior dataframe.
        chosen_trial (int): Trial number to visualize.
        variable (str): Name of the tracked body part to plot (e.g., 'TipTongue').
    
    Saves:
        MP4 file showing the tracked variable and its motion trail across video frames.
    """
    
    save_path = Path(r"C:\Users\tiffany.ona\Documents\working_memory\code\video_analysis\videos")
    session_path = save_path / str(merged.session.unique()[0])

    base_path = Path(r"E:\wm_video_analysis")
    video_path = base_path / merged.subject.unique()[0] /str(merged.session.unique()[0] + ".mp4")
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Your tracking data
    df_test = merged.loc[(merged.trial == chosen_trial)]
    x = df_test[variable + "_x"]
    y = df_test[variable + "_y"]
    likelihoods = df_test[variable + "_log"]
    video_frames = df_test["frame"]  # <- This should map to actual video frame numbers

    # Set up plot
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((frame_height, frame_width, 3), dtype=np.uint8), origin='upper')  # placeholder
    scat = ax.scatter([], [], s=50, c='red')
    trail, = ax.plot([], [], 'yellow', alpha=0.6)

    # Sliding trail buffer
    trail_x, trail_y = [], []
    trail_window = 2 * 20  # 2 seconds × 20 FPS = 40 frames

    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)  # top-left origin to match video
    ax.set_aspect('equal')
    ax.set_title("Tongue Tracking Over Video")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    # --- Animation update ---
    def update(frame_idx):
        video_frame = video_frames.iloc[frame_idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
        ret, frame = cap.read()

        if not ret:
            print(f"Frame {video_frame} not read properly.")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im.set_data(frame_rgb)

        this_likelihood = likelihoods.iloc[frame_idx]

        if this_likelihood > 0.65:
            this_x = x.iloc[frame_idx]
            this_y = y.iloc[frame_idx]
            scat.set_offsets([[this_x, this_y]])

            trail_x.append(this_x)
            trail_y.append(this_y)
        else:
            # Hide the point if likelihood is low
            scat.set_offsets([[np.nan, np.nan]])

            # Optionally: do not add to trail or add NaN to keep gaps
            trail_x.append(np.nan)
            trail_y.append(np.nan)

        # Trim trail to last 2 seconds
        if len(trail_x) > trail_window:
            trail_x.pop(0)
            trail_y.pop(0)

        # Clean trail from NaNs before plotting
        clean_x = [pt for pt in trail_x if not np.isnan(pt)]
        clean_y = [pt for pt in trail_y if not np.isnan(pt)]
        trail.set_data(clean_x, clean_y)

        return im, scat, trail
    
    ani = animation.FuncAnimation(fig, update, frames=len(x), interval=50, blit=False)
    session_path = save_path / str(merged.session.unique()[0])
    
    writer = FFMpegWriter(
        fps=33,
        codec='libx264',        # H.264 = good compression
        bitrate=2000,           # lower = more compression
        extra_args=['-pix_fmt', 'yuv420p']  # for compatibility
    )
    ani.save(str(save_path) + "/" + str(merged.session.unique()[0]) + "_trial" + str(chosen_trial) + ".mp4", writer=writer)

    cap.release()
    plt.show()
    
def create_video_trial_multi(merged: pd.DataFrame, video_path: Path, chosen_trial: int = 1, variables: list = ["TipTongue"]):
    """
    Generates a video of the chosen trial with overlaid tracking for multiple variables.
    
    Args:
        merged (pd.DataFrame): Merged pose + behavior dataframe.
        chosen_trial (int): Trial number to visualize.
        variables (list): List of body part names to track (e.g., ['TipTongue', 'UpperLip']).
    
    Saves:
        MP4 file showing tracked points and trails over video frames.
    """
    
    save_path = Path(r"C:\Users\tiffany.ona\Documents\working_memory\code\video_analysis\videos")
    session_path = save_path / str(merged.session.unique()[0])
    # video_path = base_path / merged.subject.unique()[0] / str(merged.session.unique()[0] + ".mp4")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    df_trial = merged.loc[(merged.trial == chosen_trial)]

    # Preload data
    data = {}
    for var in variables:
        data[var] = {
            "x": df_trial[f"{var}_x"].reset_index(drop=True),
            "y": df_trial[f"{var}_y"].reset_index(drop=True),
            "log": df_trial[f"{var}_log"].reset_index(drop=True),
            "trail_x": [],
            "trail_y": []
        }

    video_frames = df_trial["frame"].reset_index(drop=True)

    # --- Plot Setup ---
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((frame_height, frame_width, 3), dtype=np.uint8), origin='upper')
    scatters = {var: ax.scatter([], [], s=50, label=var) for var in variables}
    trails = {var: ax.plot([], [], alpha=0.6)[0] for var in variables}
    trail_window = 40  # ~2 sec at 20 FPS

    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)
    ax.set_aspect('equal')
    ax.set_title("Tracking Over Video")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.legend(loc='upper right')

    def update(frame_idx):
        video_frame = video_frames.iloc[frame_idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {video_frame} not read properly.")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im.set_data(frame_rgb)

        for var in variables:
            this_log = data[var]["log"].iloc[frame_idx]
            if this_log > 0.75:
                x = data[var]["x"].iloc[frame_idx]
                y = data[var]["y"].iloc[frame_idx]
                scatters[var].set_offsets([[x, y]])
                data[var]["trail_x"].append(x)
                data[var]["trail_y"].append(y)
            else:
                scatters[var].set_offsets(np.empty((0, 2)))
                data[var]["trail_x"].append(np.nan)
                data[var]["trail_y"].append(np.nan)

            if len(data[var]["trail_x"]) > trail_window:
                data[var]["trail_x"].pop(0)
                data[var]["trail_y"].pop(0)

            clean_x = [pt for pt in data[var]["trail_x"] if not np.isnan(pt)]
            clean_y = [pt for pt in data[var]["trail_y"] if not np.isnan(pt)]
            trails[var].set_data(clean_x, clean_y)

        return [im] + list(scatters.values()) + list(trails.values())

    ani = animation.FuncAnimation(fig, update, frames=len(video_frames), interval=50, blit=False)

    video_out = save_path / f"{merged.session.unique()[0]}_trial{chosen_trial}_multi.mp4"
    writer = FFMpegWriter(fps=33, codec='libx264', bitrate=2000, extra_args=['-pix_fmt', 'yuv420p'])
    ani.save(str(video_out), writer=writer)

    cap.release()
    plt.close()
