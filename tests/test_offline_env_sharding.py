import pytest


def test_shard_num_envs_requires_divisible():
  from tracker.integrations.mjlab.offline_train import shard_num_envs

  with pytest.raises(ValueError):
    shard_num_envs(total=3, world_size=2)


def test_shard_num_envs_even_split():
  from tracker.integrations.mjlab.offline_train import shard_num_envs

  assert shard_num_envs(total=4096, world_size=2) == 2048

