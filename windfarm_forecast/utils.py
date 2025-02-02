from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px


def setup_mlflow(experiment_name="windfarm_power_prediction"):
    """Setup MLflow tracking and experiment.

    Args:
        experiment_name (str): Name of the MLflow experiment

    Returns:
        str: ID of the created or existing experiment
    """
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


def plot_predictions(y_true: pd.Series, y_pred: np.ndarray | list, title: str = "Predictions vs Actual Values") -> None:
    """
    Creates an interactive line plot comparing predicted vs actual wind farm power output values.

    Args:
        y_true (pd.Series): True power output values with datetime index
        y_pred (array-like): Predicted power output values corresponding to y_true
        title (str, optional): Title for the plot. Defaults to 'Predictions vs Actual Values'

    Returns:
        None.
    """
    plot_df = pd.concat(
        [
            pd.DataFrame({"Time": y_true.index, "Power Output (kW)": y_true.values, "Type": "Actual"}),
            pd.DataFrame({"Time": y_true.index, "Power Output (kW)": y_pred, "Type": "Predicted"}),
        ]
    )

    fig = px.line(
        plot_df,
        x="Time",
        y="Power Output (kW)",
        color="Type",
        title=f"Wind Farm Power Output: {title}",
        color_discrete_map={"Actual": "#2E86AB", "Predicted": "#D64933"},
    )

    fig.update_layout(
        title={"y": 0.95, "x": 0.5, "xanchor": "center", "yanchor": "top", "font": dict(size=20)},
        xaxis_title={"text": "Time of Measurement (10-minute intervals)", "font": dict(size=14)},
        yaxis_title={"text": "Power Output (kW)", "font": dict(size=14)},
        legend=dict(title="Values", bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1),
        plot_bgcolor="#f6f6f6",
        paper_bgcolor="#f6f6f6",
        width=1800,
        height=600,
        hovermode="x unified",
    )

    fig.update_traces(line=dict(width=2), selector=dict(name="Actual"))
    fig.update_traces(line=dict(width=2, dash="dash"), selector=dict(name="Predicted"))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)", tickangle=45)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")

    fig.show()
