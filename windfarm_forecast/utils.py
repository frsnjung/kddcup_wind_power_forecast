def setup_mlflow(experiment_name="windfarm_power_prediction"):
    """Setup MLflow tracking and experiment.

    Args:
        experiment_name (str): Name of the MLflow experiment

    Returns:
        str: ID of the created or existing experiment
    """
    from pathlib import Path

    import mlflow

    # Set MLflow tracking URI to use a local directory at project root
    project_root = Path(__file__).parent.parent
    mlflow.set_tracking_uri(f"file:{project_root}/mlruns")

    # Create or get the experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # Set the experiment as active
    mlflow.set_experiment(experiment_name)
    print(f"Active experiment: {mlflow.get_experiment(experiment_id).name}")

    return experiment_id
