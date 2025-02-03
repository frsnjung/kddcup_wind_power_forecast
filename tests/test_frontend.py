import sys
from pathlib import Path

import pandas as pd
import pytest

from windfarm_forecast.frontend.app import main

# Add the project root to the path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))


@pytest.fixture
def mock_data():
    timestamps = pd.date_range(start="2020-05-01 00:10:00", periods=6, freq="10min")
    data = {
        "actual": [47, 45, 47, 42, 39, 41],
        "pred_linear_regression": [59, 58, 57, 56, 54, 55],
        "set": ["train", "train", "train", "train", "train", "train"],
        "pred_xgboost": [46, 45, 47, 43, 36, 38],
    }
    return pd.DataFrame(data, index=timestamps)


class MockStreamlit:
    def __init__(self, mock_data):
        self.sidebar_items = {}
        self.items = {}
        self.plotly_chart_called = False
        self.mock_data = mock_data

    def set_page_config(self, **kwargs):
        pass

    def title(self, text):
        pass

    def selectbox(self, label, options, key=None, format_func=None):
        return options[0]

    def date_input(self, label, value=None, min_value=None, max_value=None):
        return [self.mock_data.index.min(), self.mock_data.index.max()]

    def plotly_chart(self, fig, use_container_width=None):
        self.plotly_chart_called = True
        pass

    def columns(self, n):
        return [MockCol() for _ in range(n)]

    def error(self, text):
        raise Exception(text)

    def info(self, text):
        pass


class MockCol:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def metric(self, label, value):
        pass


def test_app_loads_without_error(monkeypatch, mock_data):
    """Test that the Streamlit app initializes and runs without raising any exceptions.

    This test verifies the basic functionality of the app by:
    1. Mocking all Streamlit components
    2. Providing mock wind farm data
    3. Running the main app function

    Args:
        monkeypatch: pytest fixture for modifying objects during testing
        mock_data: pytest fixture providing sample DataFrame with wind farm data

    Raises:
        pytest.fail: If the app raises any exceptions during execution
    """

    # Mock the load_data function to return our test data
    def mock_load_data():
        return mock_data

    monkeypatch.setattr("windfarm_forecast.frontend.app.load_data", mock_load_data)

    # Create mock streamlit instance with mock_data
    mock_st = MockStreamlit(mock_data)

    # Apply the mocks
    monkeypatch.setattr("streamlit.set_page_config", mock_st.set_page_config)
    monkeypatch.setattr("streamlit.title", mock_st.title)
    monkeypatch.setattr("streamlit.selectbox", mock_st.selectbox)
    monkeypatch.setattr("streamlit.date_input", mock_st.date_input)
    monkeypatch.setattr("streamlit.plotly_chart", mock_st.plotly_chart)
    monkeypatch.setattr("streamlit.columns", mock_st.columns)
    monkeypatch.setattr("streamlit.error", mock_st.error)
    monkeypatch.setattr("streamlit.info", mock_st.info)

    # Run the app - if no exception is raised, the test passes
    try:
        main()
    except Exception as e:
        pytest.fail(f"App failed to start: {str(e)}")


def test_plot_is_shown(monkeypatch, mock_data):
    """Test that the app successfully creates and displays a visualization plot.

    This test ensures that the app's visualization functionality is working by:
    1. Mocking all Streamlit components
    2. Providing mock wind farm data
    3. Verifying that plotly_chart is called to display the visualization

    Args:
        monkeypatch: pytest fixture for modifying objects during testing
        mock_data: pytest fixture providing sample DataFrame with wind farm data

    Raises:
        AssertionError: If no plot is displayed during app execution
    """

    # Mock the load_data function to return our test data
    def mock_load_data():
        return mock_data

    monkeypatch.setattr("windfarm_forecast.frontend.app.load_data", mock_load_data)

    # Create mock streamlit instance with mock_data
    mock_st = MockStreamlit(mock_data)

    # Apply the mocks
    monkeypatch.setattr("streamlit.set_page_config", mock_st.set_page_config)
    monkeypatch.setattr("streamlit.title", mock_st.title)
    monkeypatch.setattr("streamlit.selectbox", mock_st.selectbox)
    monkeypatch.setattr("streamlit.date_input", mock_st.date_input)
    monkeypatch.setattr("streamlit.plotly_chart", mock_st.plotly_chart)
    monkeypatch.setattr("streamlit.columns", mock_st.columns)
    monkeypatch.setattr("streamlit.error", mock_st.error)
    monkeypatch.setattr("streamlit.info", mock_st.info)

    # Run the app
    main()

    # Assert that plotly_chart was called
    assert mock_st.plotly_chart_called, "No plot was displayed in the app"
