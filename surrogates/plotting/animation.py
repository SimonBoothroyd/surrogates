import functools
import logging
import os
import pickle
from collections import defaultdict
from glob import glob

import numpy
from matplotlib import animation, pyplot

from surrogates.drivers.compute import SurrogateDriverSnapshot
from surrogates.utils.numpy import parameter_dict_to_array

logger = logging.getLogger(__name__)


class AnimatedCornerPlot:
    def __init__(self, snapshot_directory):
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

        self._parameter_labels = {}

        self._load_all_parameters()

        # Create variables to store the data used to train
        # the surrogate model and build the convex hull.
        self._simulation_parameters = {}
        self._reweighted_parameters = {}

    def _load_all_parameters(self):

        # Extract the names of the parameters
        with open(self._snapshot_file_names[0], "rb") as file:
            initial_snapshot_data: SurrogateDriverSnapshot = pickle.load(file)

        property_types = [*initial_snapshot_data.current_parameters]

        parameter_labels = {
            x: [*initial_snapshot_data.current_parameters[x]] for x in property_types
        }

        # Extract all of the parameters.
        all_parameters = {x: defaultdict(list) for x in property_types}

        for index, snapshot_file_name in enumerate(self._snapshot_file_names):

            with open(snapshot_file_name, "rb") as file:
                snapshot_data: SurrogateDriverSnapshot = pickle.load(file)

            for property_type, parameters in snapshot_data.current_parameters.items():

                for label in parameters:
                    all_parameters[property_type][label].append(parameters[label])

        for property_type in all_parameters:

            # Refactor the parameters into arrays.
            all_parameters[property_type] = {
                x: numpy.array(y) for x, y in all_parameters[property_type].items()
            }

            all_parameters[property_type] = parameter_dict_to_array(
                all_parameters[property_type], parameter_labels[property_type]
            )

            # Determine the parameter bounds.
            self._min_parameters[property_type] = numpy.min(
                all_parameters[property_type], axis=0
            )
            self._max_parameters[property_type] = numpy.max(
                all_parameters[property_type], axis=0
            )

        self._parameter_labels = parameter_labels
        self._all_parameters = all_parameters

        self._drop_untrained_parameters()

    def _drop_untrained_parameters(self):

        for property_type in self._min_parameters:

            min_parameters = self._min_parameters[property_type]
            max_parameters = self._max_parameters[property_type]

            parameter_labels = self._parameter_labels[property_type]
            all_parameters = self._all_parameters[property_type]

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

            self._min_parameters[property_type] = min_parameters
            self._max_parameters[property_type] = max_parameters

            self._parameter_labels[property_type] = parameter_labels
            self._all_parameters[property_type] = all_parameters

    def _setup_axes(self, axes, property_type):

        trace_lines = {}
        hull_lines = {}

        simulation_points = {}
        reweighted_points = {}

        parameter_labels = self._parameter_labels[property_type]

        min_parameters = self._min_parameters[property_type]
        max_parameters = self._max_parameters[property_type]

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

    def _update_training_data(self, property_type, snapshot_data):

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
                if x in self._parameter_labels[property_type]
            },
            self._parameter_labels[property_type],
        )
        reweighted_parameters = parameter_dict_to_array(
            {
                x: y
                for x, y in reweighted_parameters.items()
                if x in self._parameter_labels[property_type]
            },
            self._parameter_labels[property_type],
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
        index,
        property_type,
        n_rolling_parameters,
        trace_lines,
        hull_lines,
        reweighted_points,
        simulation_points,
    ):

        parameter_labels = self._parameter_labels[property_type]

        with open(self._snapshot_file_names[index], "rb") as file:
            snapshot_data: SurrogateDriverSnapshot = pickle.load(file)

        # Update the convex hull if it has changed.
        self._update_training_data(property_type, snapshot_data)

        all_model_parameters = self._all_parameters[property_type]

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
        property_type,
        n_rolling_parameters,
        file_name=None,
        figure_size=(5.0, 5.0),
    ):

        # Create the figure and axis
        n_axes = len(self._parameter_labels[property_type]) - 1

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
        ) = self._setup_axes(axes, property_type)

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

        n_snapshots = len(self._all_parameters[property_type])
        # n_snapshots = 10

        plot_animation = animation.FuncAnimation(
            figure, animate_function, frames=n_snapshots - 1, interval=1, blit=True
        )

        if file_name is None:
            file_name = f"{property_type}.gif"

        plot_animation.save(file_name, writer="imagemagick")
