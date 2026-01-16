from __future__ import annotations

import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Mapping

import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder


def shard_num_envs(*, total: int, world_size: int) -> int:
  if world_size <= 0:
    raise ValueError("world_size must be >= 1")
  if total % world_size != 0:
    raise ValueError(f"num_envs ({total}) must be divisible by world_size ({world_size})")
  return total // world_size


def ensure_offline_safe_logging(
  agent_cfg: RslRlOnPolicyRunnerCfg, *, env: Mapping[str, str] = os.environ
) -> None:
  """Avoid requiring W&B connectivity for offline motion training.

  If the config requests `wandb` logging but W&B isn't configured, we switch to
  `tensorboard` so offline training can run without W&B setup.
  """
  def _wandb_is_configured() -> bool:
    if env.get("WANDB_API_KEY"):
      return True

    # `wandb login` commonly writes credentials to `~/.netrc`; older installs may
    # not export `WANDB_API_KEY`. Detect that case so we don't silently disable
    # W&B logging even though the user is authenticated.
    try:
      from wandb.sdk.lib.auth import read_netrc_auth

      host = env.get("WANDB_API_HOST", "api.wandb.ai")
      # `wandb.sdk.lib.auth.read_netrc_auth()` expects a URL with a scheme
      # (it uses `urlsplit(host).netloc`). `wandb login` stores the key under
      # `machine api.wandb.ai`, so normalize accordingly.
      if "://" not in host:
        host = "https://" + host
      return read_netrc_auth(host=host) is not None
    except Exception:
      return False

  if agent_cfg.logger != "wandb":
    return

  disabled = env.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes")
  if disabled:
    agent_cfg.logger = "tensorboard"
    return

  mode = env.get("WANDB_MODE", "").lower()
  if mode in ("offline", "dryrun"):
    return

  if _wandb_is_configured():
    return

  print(
    "[INFO] W&B logging requested, but no W&B credentials detected "
    "(set `WANDB_API_KEY` or run `wandb login`). Falling back to TensorBoard."
  )
  agent_cfg.logger = "tensorboard"


def apply_offline_motion_file(env_cfg: ManagerBasedRlEnvCfg, motion_file: str) -> None:
  is_tracking_task = (
    env_cfg.commands is not None
    and "motion" in env_cfg.commands
    and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
  )
  if not is_tracking_task:
    raise ValueError("Task does not appear to be a motion-tracking task.")

  assert env_cfg.commands is not None
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = motion_file


def apply_offline_motion_pack(
  env_cfg: ManagerBasedRlEnvCfg,
  *,
  pack_dir: str,
  split: str,
) -> None:
  from tracker.tasks.tracking.config.patch import apply_motion_pack

  apply_motion_pack(env_cfg, pack_dir=Path(pack_dir), split=split)


def _make_log_dir(agent_cfg: RslRlOnPolicyRunnerCfg) -> Path:
  log_root_path = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if agent_cfg.run_name:
    log_dir_name += f"_{agent_cfg.run_name}"
  return log_root_path / log_dir_name


def run_train_offline(
  task_id: str,
  env_cfg: ManagerBasedRlEnvCfg,
  agent_cfg: RslRlOnPolicyRunnerCfg,
  *,
  motion_file: str | None = None,
  motion_pack_dir: str | None = None,
  motion_split: str = "train",
  log_dir: Path,
  wandb_run_path: str | None = None,
  torchrunx_log_dir: str | None = None,
  enable_nan_guard: bool = False,
  video: bool = False,
  video_length: int = 200,
  video_interval: int = 2000,
) -> None:
  del torchrunx_log_dir

  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = agent_cfg.seed
    rank = 0
    world_size = 1
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    os.environ["MUJOCO_GL"] = "egl"
    device = f"cuda:{local_rank}"
    seed = agent_cfg.seed + local_rank

  configure_torch_backends()

  if motion_pack_dir is not None:
    apply_offline_motion_pack(env_cfg, pack_dir=motion_pack_dir, split=motion_split)
  elif motion_file is not None:
    apply_offline_motion_file(env_cfg, motion_file)
  else:
    raise ValueError("Offline training requires either motion_file or motion_pack_dir.")

  agent_cfg.seed = seed
  env_cfg.seed = seed

  if enable_nan_guard:
    env_cfg.sim.nan_guard.enabled = True
    if rank == 0:
      print(f"[INFO] NaN guard enabled, output dir: {env_cfg.sim.nan_guard.output_dir}")

  if world_size > 1:
    env_cfg.scene.num_envs = shard_num_envs(total=env_cfg.scene.num_envs, world_size=world_size)

  if rank == 0:
    print(f"[INFO] Offline training with: device={device}, seed={seed}, world_size={world_size}")
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array" if video else None)

  # Resume handling (optional) mirrors mjlab behavior.
  resume_path: Path | None = None
  if agent_cfg.resume:
    log_root_path = log_dir.parent
    if wandb_run_path is not None:
      resume_path, was_cached = get_wandb_checkpoint_path(log_root_path, Path(wandb_run_path))
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

  # Video recording (rank 0 only).
  if video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % video_interval == 0,
      video_length=video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or OnPolicyRunner
  runner_kwargs = {}
  if runner_cls is not OnPolicyRunner:
    runner_kwargs["registry_name"] = None

  runner = runner_cls(env, asdict(agent_cfg), str(log_dir), device, **runner_kwargs)

  add_wandb_tags(agent_cfg.wandb_tags)
  if rank == 0:
    runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    if rank == 0:
      print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
  env.close()


def launch_training_offline(
  task_id: str,
  env_cfg: ManagerBasedRlEnvCfg,
  agent_cfg: RslRlOnPolicyRunnerCfg,
  *,
  motion_file: str | None = None,
  motion_pack_dir: str | None = None,
  motion_split: str = "train",
  gpu_ids: list[int] | str | None = None,
  wandb_run_path: str | None = None,
  torchrunx_log_dir: str | None = None,
  enable_nan_guard: bool = False,
  video: bool = False,
  video_length: int = 200,
  video_interval: int = 2000,
) -> None:
  log_dir = _make_log_dir(agent_cfg)

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(gpu_ids)

  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"

  ensure_offline_safe_logging(agent_cfg)

  if num_gpus <= 1:
    run_train_offline(
      task_id,
      env_cfg,
      agent_cfg,
      motion_file=motion_file,
      motion_pack_dir=motion_pack_dir,
      motion_split=motion_split,
      log_dir=log_dir,
      wandb_run_path=wandb_run_path,
      torchrunx_log_dir=torchrunx_log_dir,
      enable_nan_guard=enable_nan_guard,
      video=video,
      video_length=video_length,
      video_interval=video_interval,
    )
    return

  import torchrunx

  logging.basicConfig(level=logging.INFO)

  if "TORCHRUNX_LOG_DIR" not in os.environ:
    if torchrunx_log_dir is not None:
      os.environ["TORCHRUNX_LOG_DIR"] = torchrunx_log_dir
    else:
      os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

  print(f"[INFO] Launching offline training with {num_gpus} GPUs", flush=True)
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=num_gpus,
    backend=None,
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
  ).run(
    run_train_offline,
    task_id,
    env_cfg,
    agent_cfg,
    motion_file=motion_file,
    motion_pack_dir=motion_pack_dir,
    motion_split=motion_split,
    log_dir=log_dir,
    wandb_run_path=wandb_run_path,
    torchrunx_log_dir=torchrunx_log_dir,
    enable_nan_guard=enable_nan_guard,
    video=video,
    video_length=video_length,
    video_interval=video_interval,
  )
