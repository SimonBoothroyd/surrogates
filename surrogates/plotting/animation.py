import functools
import logging
import os
import pickle
from collections import defaultdict
from glob import glob
from typing import Dict, Optional, Tuple

import numpy
from matplotlib import animation, pyplot
from matplotlib.lines import Line2D

from surrogates.drivers.compute import SurrogateDriverSnapshot
from surrogates.utils.numpy import parameter_dict_to_array

logger = logging.getLogger(__name__)


class AnimatedCornerPlot:
    def __init__(self, snapshot_directory: str) -> None:
        """

        Parameters
        ----------
        snapshot_directory: str
            The directory containing the snapshots.
        """

        # Find the snapshot files.
        self._snapshot_file_names = glob(os.path.join(snapshot_directory, "*"))
        self._snapshot_file_names.sort(key=lambda x: int(os.path.basename(x)))

        # Load in the parameters.
        self._all_parameters = {}

        self._min_parameters = {}
        self._max_parameters = {}

        self._parameter_labels = []

        self._load_all_parameters()

        # Create variables to store the data used to train
        # the surrogate model and build the convex hull.
        self._simulation_parameters = {}
        self._reweighted_parameters = {}

    def _load_all_parameters(self) -> None:
        """Loads all of the model parameters from each of the
        snapshots.
        """

        # Extract the names of the parameters
        with open(self._snapshot_file_names[0], "rb") as file:
            initial_snapshot_data: SurrogateDriverSnapshot = pickle.load(file)

        property_types = [*initial_snapshot_data.current_parameters]

        parameter_labels = {
            x: [*initial_snapshot_data.current_parameters[x]] for x in property_types
        }

        # Extract all of the parameters.
        all_parameters = defaultdict(list)

        for index, snapshot_file_name in enumerate(self._snapshot_file_names):

            with open(snapshot_file_name, "rb") as file:
                snapshot_data: SurrogateDriverSnapshot = pickle.load(file)

            for label in snapshot_data.current_parameters:
                all_parameters[label].append(snapshot_data.current_parameters[label])

        # Refactor the parameters into arrays.
        all_parameters = {
            x: numpy.concatenate(y).reshape(-1, 1) for x, y in all_parameters.items()
        }

        all_parameters = parameter_dict_to_array(all_parameters, [*all_parameters])

        # Determine the parameter bounds.
        self._min_parameters = numpy.min(all_parameters, axis=0)
        self._max_parameters = numpy.max(all_parameters, axis=0)

        self._parameter_labels = parameter_labels
        self._all_parameters = all_parameters

        self._drop_untrained_parameters()

    def _drop_untrained_parameters(self) -> None:
        """Removes any parameters which the surrogate model was not
        trained upon for the parameter arrays.
        """

        min_parameters = self._min_parameters
        max_parameters = self._max_parameters

        parameter_labels = self._parameter_labels
        all_parameters = self._all_parameters

        flat_parameter_indices = numpy.argwhere(
            numpy.isclose(min_parameters, max_parameters)
        )
        index_mask = numpy.ones(len(min_parameters), numpy.bool)
        index_mask[flat_parameter_indices] = 0

        parameter_labels = [
            x
            for i, x in enumerate(parameter_labels)
            if i not in flat_parameter_indices
        ]

        min_parameters = min_parameters[index_mask]
        max_parameters = max_parameters[index_mask]

        all_parameters = all_parameters[:, index_mask]

        self._min_parameters = min_parameters
        self._max_parameters = max_parameters

        self._parameter_labels = parameter_labels
        self._all_parameters = all_parameters

    def _setup_axes(
        self, axes: numpy.ndarray
    ) -> Tuple[
        Dict[Tuple[int, int], Line2D],
        Dict[Tuple[int, int], Line2D],
        Dict[Tuple[int, int], Line2D],
        Dict[Tuple[int, int], Line2D],
    ]:
        """Set up the figure axes, settings limits, labels and
        creating the line brushes.

        Parameters
        ----------
        axes: numpy.ndarray
            The axes to set up.

        Returns
        -------
        Dict[Tuple[int, int], Line2D]
            The brushes used to draw the convex hull.
        Dict[Tuple[int, int], Line2D]
            The brushes used to draw the reweighted points.
        Dict[Tuple[int, int], Line2D]
            The brushes used to draw the simulated points.
        Dict[Tuple[int, int], Line2D]
            The brushes used to draw the parameter trace.
        """

        trace_lines = {}
        hull_lines = {}

        simulation_points = {}
        reweighted_points = {}

        parameter_labels = self._parameter_labels

        min_parameters = self._min_parameters
        max_parameters = self._max_parameters

        n_axes = len(parameter_labels) - 2

        for x_index in range(len(parameter_labels) - 1):

            for y_index in range(x_index):

                axes[y_index, x_index].axis("off")

        for x_index in range(len(parameter_labels) - 1):

            for y_index in range(len(parameter_labels) - 1 - x_index):

                axis = axes[n_axes - y_index, x_index]

                axis.set_xlim(
                    (min_parameters[x_index] * 0.98, max_parameters[x_index] * 1.02)
                )
                axis.set_ylim(
                    (
                        min_parameters[-y_index - 1] * 0.98,
                        max_parameters[-y_index - 1] * 1.02,
                    )
                )

                if y_index == 0:
                    axis.set_xlabel(parameter_labels[x_index])
                if x_index == 0:
                    axis.set_ylabel(parameter_labels[-y_index - 1])

                # Create the brushes
                (simulation_point,) = axis.plot([], [], "o")
                (reweighted_point,) = axis.plot([], [], "x")
                (hull_line,) = axis.plot([], [], "r--", lw=2)
                (trace_line,) = axis.plot([], [], color="C0")

                trace_lines[x_index, y_index] = trace_line
                hull_lines[x_index, y_index] = hull_line

                simulation_points[x_index, y_index] = simulation_point
                reweighted_points[x_index, y_index] = reweighted_point

        return hull_lines, reweighted_points, simulation_points, trace_lines

    def _update_training_data(
        self, property_type: str, snapshot_data: SurrogateDriverSnapshot
    ) -> None:
        """Appends the current snapshots training data onto the
        previously loaded training data.

        Parameters
        ----------
        property_type: str
            The property type the model was trained upon.
        snapshot_data: SurrogateDriverSnapshot
            The current snapshot which contains the training data.
        """

        if len(snapshot_data.simulation_training_parameters) <= 0:
            return

        simulation_parameters = snapshot_data.simulation_training_parameters[
            property_type
        ]
        reweighted_parameters = snapshot_data.reweighted_training_parameters[
            property_type
        ]

        simulation_parameters = parameter_dict_to_array(
            {
                x: y
                for x, y in simulation_parameters.items()
                if x in self._parameter_labels
            },
            self._parameter_labels,
        )
        reweighted_parameters = parameter_dict_to_array(
            {
                x: y
                for x, y in reweighted_parameters.items()
                if x in self._parameter_labels
            },
            self._parameter_labels,
        )

        if property_type not in self._simulation_parameters:

            self._simulation_parameters[property_type] = simulation_parameters
            self._reweighted_parameters[property_type] = reweighted_parameters

        else:

            self._simulation_parameters[property_type] = numpy.concatenate(
                [self._simulation_parameters[property_type], simulation_parameters],
                axis=0,
            )
            self._reweighted_parameters[property_type] = numpy.concatenate(
                [self._reweighted_parameters[property_type], reweighted_parameters],
                axis=0,
            )

    def _animate(
        self,
        index: int,
        property_type: str,
        n_rolling_parameters: int,
        trace_lines: Dict[Tuple[int, int], Line2D],
        hull_lines: Dict[Tuple[int, int], Line2D],
        reweighted_points: Dict[Tuple[int, int], Line2D],
        simulation_points: Dict[Tuple[int, int], Line2D],
    ) -> Tuple[Line2D, Line2D, Line2D, Line2D]:
        """Renders a single frame of the animation

        Parameters
        ----------
        index: int
            The index of the frame to render.
        property_type: str
            The property type of interest.
        n_rolling_parameters: int
            The number of previous trace points to plot.
        trace_lines: Dict[Tuple[int, int], Line2D]
            The brushes used to draw the parameter trace.
        hull_lines: Dict[Tuple[int, int], Line2D]
            The brushes used to draw the convex hull.
        reweighted_points: Dict[Tuple[int, int], Line2D]
            The brushes used to draw the reweighted points.
        simulation_points: Dict[Tuple[int, int], Line2D]
            The brushes used to draw the simulated points.

        Returns
        -------
        Tuple[Line2D, Line2D, Line2D, Line2D]
            The brushes used during the rendering.
        """

        parameter_labels = self._parameter_labels

        with open(self._snapshot_file_names[index], "rb") as file:
            snapshot_data: SurrogateDriverSnapshot = pickle.load(file)

        # Update the convex hull if it has changed.
        self._update_training_data(property_type, snapshot_data)

        all_model_parameters = self._all_parameters

        # all_training_parameters = numpy.concatenate(
        #     [
        #         self._simulation_parameters[property_type],
        #         self._reweighted_parameters[property_type],
        #     ],
        #     axis=0,
        # )

        for x_index in range(len(parameter_labels) - 1):

            for y_index in range(len(parameter_labels) - 1 - x_index):

                x = all_model_parameters[
                    max(0, index - n_rolling_parameters) : index + 1, x_index
                ]
                y = all_model_parameters[
                    max(0, index - n_rolling_parameters) : index + 1, -y_index - 1
                ]

                # convex_hull = ConvexHull(all_training_parameters[:, [x_index, -y_index - 1]])
                # convex_vertices = convex_hull.vertices.tolist()

                trace_lines[x_index, y_index].set_data(numpy.ravel(x), numpy.ravel(y))

                # hull_lines[x_index, y_index].set_data(
                #     all_training_parameters[convex_vertices + [convex_vertices[0]], 0],
                #     all_training_parameters[convex_vertices + [convex_vertices[0]], 1],
                # )

                simulation_points[x_index, y_index].set_data(
                    self._simulation_parameters[property_type][:, x_index],
                    self._simulation_parameters[property_type][:, -y_index - 1],
                )
                reweighted_points[x_index, y_index].set_data(
                    self._reweighted_parameters[property_type][:, x_index],
                    self._reweighted_parameters[property_type][:, -y_index - 1],
                )

        logger.info(f"Finished animating frame {index}")

        return (
            *trace_lines.values(),
            *hull_lines.values(),
            *simulation_points.values(),
            *reweighted_points.values(),
        )

    def plot(
        self,
        property_type: str,
        n_rolling_parameters: int,
        file_name: Optional[str] = None,
        figure_size: Tuple[float, float] = (5.0, 5.0),
    ):
        """Produce an animated plot of a previously ran optimization.

        Parameters
        ----------
        property_type: str
            The property type of interest.
        n_rolling_parameters:
            The number of model parameters to show
            at any one time.
        file_name: str, optional
            The file name to save the animation to. If
            None, the `property_type` will be used to
            choose the file name.
        figure_size: tuple of float, float
            The size of the figure to produce.
        """

        # Create the figure and axis
        n_axes = len(self._parameter_labels) - 1

        figure, axes = pyplot.subplots(
            nrows=n_axes,
            ncols=n_axes,
            sharex="col",
            sharey="row",
            figsize=figure_size,
            squeeze=False,
        )

        (
            hull_lines,
            reweighted_points,
            simulation_points,
            trace_lines,
        ) = self._setup_axes(axes)

        figure.tight_layout()

        animate_function = functools.partial(
            self._animate,
            property_type=property_type,
            n_rolling_parameters=n_rolling_parameters,
            trace_lines=trace_lines,
            hull_lines=hull_lines,
            reweighted_points=reweighted_points,
            simulation_points=simulation_points,
        )

        n_snapshots = len(self._all_parameters)
        # n_snapshots = 10

        plot_animation = animation.FuncAnimation(
            figure, animate_function, frames=n_snapshots - 1, interval=1, blit=True
        )

        if file_name is None:
            file_name = f"{property_type}.gif"

        plot_animation.save(file_name, writer="imagemagick")
