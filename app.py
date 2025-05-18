# shiny run --reload app.py
# visit http://127.0.0.1:8000 in your web browser
  

# app.py
from shiny import App, ui, render_plot
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

import shiny_data

# load Dataset - here its Iris
iris_bunch = load_iris(as_frame=True)
iris_df    = iris_bunch.frame
# the sepal‐length column is named "sepal length (cm)"

# the UI
app_ui = ui.page_fluid(
    ui.h2("Iris Sepal Length Histogram"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_slider("bins", "Number of bins:", min=1, max=50, value=20),
        ),
        # main content goes here:
        ui.output_plot("distPlot"),
    )
)

# server logic
def server(input, output, session):
    @output
    @render_plot
    def distPlot():
        x = iris_df["sepal length (cm)"]
        bins = np.linspace(x.min(), x.max(), input.bins() + 1)
        fig, ax = plt.subplots()
        ax.hist(
            x,
            bins=bins,
            color="darkgray",
            edgecolor="white",
        )
        ax.set_xlabel("Sepal length (cm)")
        ax.set_title("Distribution of Iris Sepal Lengths")
        return fig

# create the app
app = App(app_ui, server)

if __name__ == "__main__":
    from shiny import run_app
    run_app(app, launch_browser=True)

