"""Shared scalar training/evaluation histories for SAC and FastSAC."""


def comparison_metrics(values, env_steps):
    """Keep training episode curves separate from periodic policy evaluation."""
    values = {**values, "env_steps": int(env_steps)}
    for source_prefix, target_prefix in (
        ("training", "metrics"),
        ("eval", "eval_metrics"),
    ):
        for source, target in (
            ("reward", "reward"),
            ("success", "success_rate"),
            ("solved_per_step", "solved_step_fraction"),
            ("numerical_failure", "numerical_failure_rate"),
        ):
            key = f"{source_prefix}/episode_{source}"
            if key in values:
                values[f"{target_prefix}/{target}"] = values[key]
    return values


def configure_wandb_metrics(run):
    run.define_metric("env_steps", hidden=True)
    # These are numeric history series, not summary-only/bar/table plots.
    for prefix in ("metrics", "eval_metrics"):
        for name in (
            "reward",
            "success_rate",
            "solved_step_fraction",
            "numerical_failure_rate",
        ):
            run.define_metric(
                f"{prefix}/{name}", step_metric="env_steps", summary="none"
            )
