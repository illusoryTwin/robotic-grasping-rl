"""Agent configurations for UR10 lift task."""

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class UR10LiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for UR10 lift task."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "ur10_lift"
    run_name = ""
    resume = False
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TwoStageGraspPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for two-stage grasp task.

    Optimized for sparse reward with single RL step per episode.
    """

    num_steps_per_env = 1  # Single RL step per episode (pose prediction)
    max_iterations = 10000
    save_interval = 100
    experiment_name = "two_stage_grasp"
    run_name = ""
    resume = False
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # Lower noise for pose prediction
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Higher entropy for exploration with sparse reward
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


__all__ = ["UR10LiftPPORunnerCfg", "TwoStageGraspPPORunnerCfg"]
