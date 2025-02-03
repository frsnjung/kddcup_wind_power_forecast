import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))


def load_data():
    """Load the predictions data"""
    file_path = project_root / "data/modified/predictions/predictions.parquet"
    df = pd.read_parquet(file_path)
    # Ensure the index is datetime and drop any NaT values
    df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=["actual"])  # Drop rows with NaN values
    return df


def create_plot(data, model_col, title="Wind Farm Power Output Predictions"):
    """
    Creates a Plotly figure comparing predicted vs actual wind farm power output values.
    """
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({"Time": data.index, "Actual": data["actual"], "Predicted": data[model_col]})

    # Create the plot
    fig = px.line(
        plot_df,
        x="Time",
        y=["Actual", "Predicted"],
        title=title,
        labels={"value": "Power Output (MW)", "variable": "Type"},
    )

    # Update layout
    fig.update_layout(
        title={"y": 0.95, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        xaxis_title=dict(
            text="Time of Measurement (10-minute intervals)", font=dict(size=14, color="black", family="Arial Bold")
        ),
        yaxis_title=dict(text="Power Output (MW)", font=dict(size=14, color="black", family="Arial Bold")),
        legend=dict(title="Type"),
        plot_bgcolor="#f6f6f6",
        paper_bgcolor="#f6f6f6",
        font=dict(size=12, color="#000000"),
    )

    # Update line colors and styles with increased transparency
    fig.update_traces(
        line=dict(color="rgba(46, 134, 171, 0.4)", width=2),  # Blue with 0.4 opacity
        selector=dict(name="Actual"),
    )
    fig.update_traces(
        line=dict(color="rgba(214, 73, 51, 0.4)", width=2, dash="dash"),  # Red with 0.4 opacity
        selector=dict(name="Predicted"),
    )

    # Make axes and grid more visible
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(0,0,0,0.1)",
        showline=True,
        linewidth=2,
        linecolor="black",
        ticks="outside",
        tickfont=dict(size=12, color="black", family="Arial Bold"),  # Enhance tick labels
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(0,0,0,0.1)",
        showline=True,
        linewidth=2,
        linecolor="black",
        ticks="outside",
        tickfont=dict(size=12, color="black", family="Arial Bold"),  # Enhance tick labels
    )

    return fig


def calculate_metrics(data, prediction_col):
    """Calculate metrics for the selected prediction"""
    mae = (data["actual"] - data[prediction_col]).abs().mean()
    rmse = ((data["actual"] - data[prediction_col]) ** 2).mean() ** 0.5
    return mae, rmse


def main():
    st.set_page_config(page_title="Wind Farm Predictions", layout="wide")

    st.title("Wind Farm Power Output Predictions")

    try:
        # Load data
        data = load_data()

        # Dataset selector
        dataset = st.selectbox(
            "Select Dataset",
            ["Training", "Validation"],
            key="dataset_selector",
            format_func=lambda x: "Training Set" if x == "Training" else "Validation Set",
        )

        # Filter data based on selected dataset
        dataset_value = "train" if dataset == "Training" else "val"
        mask = data["set"] == dataset_value
        filtered_data = data[mask].copy()

        # Model selector
        model = st.selectbox("Select Model", ["Linear Regression", "XGBoost"], key="model_selector")

        prediction_col = "pred_linear_regression" if model == "Linear Regression" else "pred_xgboost"

        # Date range selector
        min_date = filtered_data.index.min().date()
        max_date = filtered_data.index.max().date()

        date_range = st.date_input(
            "Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
        )

        if len(date_range) == 2:
            start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            # Add time to make it inclusive of the full days
            start_date = start_date.replace(hour=0, minute=0)
            end_date = end_date.replace(hour=23, minute=59)

            # Filter data based on selected dates
            mask = (filtered_data.index >= start_date) & (filtered_data.index <= end_date)
            display_data = filtered_data[mask]

            # Create and display plot
            fig = create_plot(display_data, prediction_col, f"{dataset} Set: Wind Farm Power Output - {model} Model")
            st.plotly_chart(fig, use_container_width=True)

            # Calculate and display metrics
            mae, rmse = calculate_metrics(display_data, prediction_col)

            # Display metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Absolute Error", f"{mae:.2f} kW")
            with col2:
                st.metric("Root Mean Square Error", f"{rmse:.2f} kW")

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure the predictions.parquet file is in the correct location and format.")


if __name__ == "__main__":
    main()
