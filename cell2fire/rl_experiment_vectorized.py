import argparse
import json
import os
from datetime import datetime

from evaluate_model import (
    MAP_TO_EXTRA_KWARGS,
    MAP_TO_IGNITION_POINTS,
    SB3_ALGO_TO_MODEL_CLASS,
)
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from cell2fire.gym_env import FireEnv
from firehose.config import set_training_enabled
from firehose.models import PaddedNatureCNN
from firehose.utils import TrainerEncoder


Model = object

set_training_enabled(True)


class Trainer:
    def __init__(self, args):
        """ Process args to setup trainer """
        self.args = args
        self.total_timesteps = args.train_steps
        self.checkpoint_save_freq = int(self.total_timesteps / 300)

        # Steps before sim and per action
        self.steps_before_sim = MAP_TO_EXTRA_KWARGS[args.map]["steps_before_sim"]
        self.steps_per_action = MAP_TO_EXTRA_KWARGS[args.map]["steps_per_action"]

        # Determine observation type for the architecture and model kwargs
        if args.architecture == "CnnPolicy":
            self.model_kwargs = {"features_extractor_class": PaddedNatureCNN}
            self.observation_type = "forest_rgb"
        else:
            # MlpPolicy needs just forest -1,0,1 observations
            assert args.architecture == "MlpPolicy"
            self.model_kwargs = {}
            self.observation_type = "forest"

        # Unique ID for this training run, used by tensorboard
        self.unique_id = (
            f"algo={args.algo}"
            f"__ignition_type={args.ignition_type}"
            f"__map={args.map}"
            f"__architecture={args.architecture}"
            f"__action_space={args.action_space}"
            f"__seed={args.seed}"
            f"__acr={args.action_diameter}"
            f"__gamma={args.gamma}"
            f"__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        set_random_seed(args.seed)

        # Where we save the model checkpoints and self dump JSON
        self.train_log_dir = os.path.join("train_logs", self.unique_id)

        # Set the log directory for tensorboard
        self.tf_logdir = os.path.join(args.tf_logdir, self.unique_id)

        # Output directory of cell2fire itself with all the Forest CSVs.
        # We don't really want to save these for future use
        self.out_dir = (
            # On supercloud TMPDIR is the slurm job directory
            os.environ["TMPDIR"]
            if "TMPDIR" in os.environ.keys()
            else os.path.dirname(os.path.realpath(__file__))
        )

        # Print debug info
        print("Args:", json.dumps(args.__dict__, indent=2))
        print("Saving checkpoints to", self.train_log_dir)
        print("Checkpoint frequency:", self.checkpoint_save_freq)
        print("cell2fire output directory:", self.out_dir)
        print("Tensorboard logdir:", self.tf_logdir)
        print("========================================")

        print("Model kwargs:", self.model_kwargs)
        print("Observation type:", self.observation_type)
        print("Total train timesteps:", self.total_timesteps)

        self._dump_to_json()

    def _dump_to_json(self):
        """ Dump self to JSON"""
        if os.path.exists(self.train_log_dir):
            raise RuntimeError(
                f"Train log directory {self.train_log_dir} already exists"
            )

        os.makedirs(self.train_log_dir)
        log_fname = os.path.join(self.train_log_dir, "train_args.json")

        # Write params to JSON
        with open(log_fname, "w") as f:
            json.dump(self.__dict__, f, cls=TrainerEncoder, indent=2)
        print("Wrote train params to", log_fname)

    def _get_env(self):
        """ Get the environment based on the ignition type """
        args = self.args
        if args.ignition_type == "fixed":
            ig_points = MAP_TO_IGNITION_POINTS[args.map]
            single_env = lambda: FireEnv(
                ignition_points=ig_points,
                action_type=args.action_space,
                action_diameter=args.action_diameter,
                fire_map=args.map,
                output_dir=self.out_dir,
                observation_type=self.observation_type,
                steps_before_sim=self.steps_before_sim,
                steps_per_action=self.steps_per_action,
            )
            return single_env
        elif args.ignition_type == "random":
            single_env = lambda: FireEnv(
                action_type=args.action_space,
                action_diameter=args.action_diameter,
                fire_map=args.map,
                observation_type=self.observation_type,
                output_dir=self.out_dir,
                steps_before_sim=self.steps_before_sim,
                steps_per_action=self.steps_per_action,
            )
            return single_env
        else:
            raise NotImplementedError

    def _get_model(self, env) -> Model:
        args = self.args

        if args.resume_from:
            # Check if we need to reload from checkpoint
            if not os.path.exists(args.resume_from):
                raise ValueError(f"Checkpoint {args.resume_from} does not exist")
            # TODO: figure out passing gamma, model kwargs, etc. in clean way
            model = SB3_ALGO_TO_MODEL_CLASS[args.algo].load(args.resume_from)
            old_tf_logdir = model.tensorboard_log
            model.tensorboard_log = self.tf_logdir
            print(
                "Warning! Loading checkpoint from disk. Some args may not be used (e.g. gamma)"
            )
            print(
                f"Overrode tensorboard log dir from {old_tf_logdir} to {self.tf_logdir}"
            )
        elif args.algo in {"ppo", "a2c", "trpo", "ppo-maskable"}:
            # If no reload specified then just create a new model
            model_cls = SB3_ALGO_TO_MODEL_CLASS[args.algo]
            model = model_cls(
                args.architecture,
                env,
                verbose=1,
                tensorboard_log=self.tf_logdir,
                gamma=args.gamma,
                policy_kwargs=self.model_kwargs,
            )
        elif args.algo == "dqn":
            # DQN doesn't support gamma so handle separately
            model = DQN(
                args.architecture, env, verbose=1, tensorboard_log=self.tf_logdir
            )
        else:
            raise NotImplementedError

        # Workaround for logits having invalid values for masked PPO
        if args.algo == "ppo-maskable":
            model.set_env(env, force_reset=True)

        return model

    def evaluate(self):
        # FIXME: should we bother? Can just use evaluate_model.py script
        raise NotImplementedError

    def train(self):
        args = self.args

        # Need to use use SubprocVecEnv so its parallelized. DummyVecEnv is sequential on a single core
        single_env = self._get_env()
        if args.num_processes == 1:
            # Mainly for debugging purposes
            env = single_env()
        else:
            env = make_vec_env(
                single_env, n_envs=args.num_processes, vec_env_cls=SubprocVecEnv
            )

        # Get the model and setup checkpointing
        model = self._get_model(env)
        checkpoint_callback = CheckpointCallback(
            save_freq=self.checkpoint_save_freq, save_path=self.train_log_dir
        )

        # Train the model
        try:
            model.learn(
                total_timesteps=self.total_timesteps, callback=[checkpoint_callback]
            )
        except Exception as e:
            model.save(os.path.join(self.train_log_dir, f"{args.algo}_final.zip"))
            raise e

        # Save the final model
        model.save(os.path.join(self.train_log_dir, f"{args.algo}_final.zip"))
        env.close()
        print("Done!")


def train(args):
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-al",
        "--algo",
        default="a2c",
        help="Specifies the RL algorithm to use",
        choices=set(SB3_ALGO_TO_MODEL_CLASS.keys()),
    )
    parser.add_argument(
        "-m",
        "--map",
        default="Sub20x20",
        help="Specifies the map to run the environment in",
    )
    parser.add_argument(
        "-ar",
        "--architecture",
        default="MlpPolicy",
        help="Specifies whether to use an MLP or CNN as the neural architecture for the agent",
    )
    parser.add_argument(
        "-i",
        "--ignition_type",
        default="random",
        help="Specifies whether to use a random or fixed fire ignitinon point",
        choices={"fixed", "random"},
    )
    parser.add_argument(
        "-as",
        "--action_space",
        default="flat",
        help="Action space type",
        choices=FireEnv.ACTION_TYPES,
    )
    parser.add_argument("-g", "--gamma", default=0.9, type=float, help="Agent gamma")
    parser.add_argument(
        "-acr", "--action_diameter", default=1, type=int, help="Action diameter"
    )
    parser.add_argument(
        "-t",
        "--train_steps",
        default=5_000_000,
        type=int,
        help="Number of training steps",
    )
    parser.add_argument("-s", "--seed", default=0, type=int, help="RL seed")
    parser.add_argument(
        "-n",
        "--num-processes",
        default=16,
        type=int,
        help="Number of parallel processes",
    )
    parser.add_argument(
        "--resume-from", type=str, help="Resume training from checkpoint"
    )
    parser.add_argument(
        "-l",
        "--tf_logdir",
        default=f"/home/gridsan/{os.environ['USER']}/firehosetmp",
        help="Logdir for Tensorboard",
    )
    # parser.add_argument(
    #     "-l", "--tf_logdir", default="/tmp/firehose", help="Logdir for Tensorboard"
    # )
    train(args=parser.parse_args())
