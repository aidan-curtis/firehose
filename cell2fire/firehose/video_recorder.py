import os
from datetime import datetime

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym_env import FireEnv


class FirehoseVideoRecorder:
    """Wrapper around gym VideoRecorder to record firehose environment."""

    def __init__(self, env: FireEnv, args, disable_video: bool = False):
        if disable_video:
            self.video_recorder = None
        else:
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Make video directory if it doesn't exist
            if not os.path.exists("videos"):
                os.mkdir("videos")

            video_fname = f"videos/{args.algo}-{args.map}-{date_str}.mp4"
            self.video_recorder = VideoRecorder(env, video_fname, enabled=True)

    def capture_frame(self):
        if self.video_recorder is not None:
            self.video_recorder.capture_frame()

    def close(self):
        if self.video_recorder is not None:
            self.video_recorder.close()
            os.remove(self.video_recorder.metadata_path)
