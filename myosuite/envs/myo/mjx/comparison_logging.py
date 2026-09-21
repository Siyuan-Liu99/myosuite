"""Shared, evaluation-only comparison metrics for SAC and FastSAC."""


def comparison_metrics(values, env_steps):
    """Keep diagnostics while exposing the same headline metrics for both runs.

    Do not carry evaluation values into training-only records: this would make
    stale reward/success values look like new evaluations.
    """
    values = {**values, "env_steps": int(env_steps)}
    for source, target in (
        ("eval/episode_reward", "metrics/reward"),
        ("eval/episode_success", "metrics/success_rate"),
        ("eval/episode_solved_per_step", "metrics/solved_step_fraction"),
        ("eval/episode_numerical_failure", "metrics/numerical_failure_rate"),
    ):
        if source in values:
            values[target] = values[source]
    return values


def configure_wandb_metrics(run):
    run.define_metric("env_steps")
    run.define_metric("metrics/*", step_metric="env_steps")
