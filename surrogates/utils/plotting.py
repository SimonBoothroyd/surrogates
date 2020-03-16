import arviz
import corner
import numpy
from matplotlib import pyplot


def plot_trace(trace, parameter_labels, show=False):
    """Use `Arviz` to plot a trace of the variable parameters,
    alongside a histogram of their distribution.

    Parameters
    ----------
    trace: numpy.ndarray
        The parameter trace with shape=(n_steps, n_variable_parameters)
    parameter_labels: list of str
        The names of each parameter in the trace.
    show: bool
        If true, the plot will be shown.

    Returns
    -------
    matplotlib.pyplot.Figure
        The plotted figure.
    """

    data = arviz.convert_to_inference_data(trace)

    axes = arviz.plot_trace(data)
    figure = axes[0][0].figure

    if show:
        figure.show()

    return figure


def plot_corner(trace, parameter_labels, show=False):
    """Use `corner` to plot a corner plot of the parameter
    distributions.

    Parameters
    ----------
    trace: dict of str and numpy.ndarray
        The parameter trace with shape=(n_steps,)
    parameter_labels: list of str
        The names of each parameter in the trace.
    show: bool
        If true, the plot will be shown.

    Returns
    -------
    matplotlib.pyplot.Figure
        The plotted figure.
    """

    trace_array = numpy.concatenate(
        [trace[x].reshape(-1, 1) for x in parameter_labels], axis=1
    )
    figure = corner.corner(trace_array, labels=parameter_labels, color="#17becf")

    if show:
        figure.show()

    return figure


def plot_log_p(log_p, show=False, label="$log p$"):
    """Plot the log p trace.

    Parameters
    ----------
    log_p: numpy.ndarray
        The log p trace with shape=(n_steps, 1)
    show: bool
        If true, the plot will be shown.
    label: str
        The y-axis label to use.

    Returns
    -------
    matplotlib.pyplot.Figure
        The plotted figure.
    """
    figure, axes = pyplot.subplots(1, 1, figsize=(5, 5), dpi=200)

    axes.plot(log_p, color="#17becf")
    axes.set_xlabel("steps")
    axes.set_ylabel(f"{label}")

    if show:
        figure.show()

    return figure
