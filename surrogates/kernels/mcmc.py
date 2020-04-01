"""
Code to perform MCMC simulations on model parameters.
"""
import json
import os
from typing import Dict, Optional, Tuple

import numpy
import torch
from tqdm import tqdm

from surrogates.kernels.samplers import Metropolis, Sampler
from surrogates.models import BayesianModel


class MCMCSimulation:
    """ Builds an object that samples the posterior of
    a specified model.
    """

    @property
    def trace(self) -> Dict[str, numpy.ndarray]:
        """dict of str and numpy.ndarray: A trajectory of the model parameters over the
        course of the simulation with shape=(n_steps, n_trainable_parameters)."""
        return {
            x: y if len(y) == 0 else numpy.concatenate(y)
            for x, y in self._trace.items()
        }

    @property
    def log_p_trace(self) -> numpy.ndarray:
        """numpy.ndarray: A trajectory of the value of log p over the course
        of the simulation with shape=(n_steps,)."""
        return numpy.asarray(self._log_p_trace)

    def __init__(
        self,
        model: BayesianModel,
        initial_parameters: Dict[str, numpy.ndarray],
        sampler: Optional[Sampler] = None,
        random_seed: Optional[int] = None,
    ):
        """Initializes the basic state of the simulator object.

        Parameters
        ----------
        model: BayesianModel
            The model whose posterior should be sampled.
        initial_parameters: dict of str and numpy.ndarray
            The initial parameters to seed the simulation with.
        sampler: Sampler, optional
            The sampler to use for in-model proposals. If None,
            a default `MetropolisSampler` will be used.
        random_seed: int, optional
            The random seed to use.
        """

        self._model = model

        # Make sure the parameters are valid.
        self._validate_parameters(initial_parameters)
        self._initial_values = initial_parameters

        # Make sure we have a sampler set
        if sampler is None:

            proposal_sizes = {
                x: 0.01 if y <= 0.0 else y / 100.0
                for x, y in initial_parameters.items()
            }
            sampler = Metropolis(self._model, proposal_sizes)

        sampler.log_p_function = self._evaluate_log_p
        self._sampler = sampler

        # Set a random seed
        if random_seed is None:
            random_seed = torch.randint(1000000, (1,)).item()

        self._random_seed = random_seed

        torch.manual_seed(self._random_seed)
        numpy.random.seed(self._random_seed)

        # Set up any bookkeeping arrays
        self._has_stepped = False

        # Set up the trace arrays.
        self._trace = {label: [] for label in model.trainable_parameters}
        self._log_p_trace = []

    def _validate_parameters(self, initial_parameters: Dict[str, numpy.ndarray]):

        initial_log_p, initial_gradients = self._evaluate_log_p(initial_parameters)

        if numpy.isnan(initial_log_p) or numpy.isinf(initial_log_p):
            raise ValueError(f"The initial log p is NaN / inf - {initial_log_p}")

        if any(numpy.isnan(x) or numpy.isinf(x) for x in initial_gradients.values()):

            raise ValueError(
                f"The initial gradient of log p is NaN / inf - {initial_gradients}"
            )

    def propagate(
        self, steps: int, warm_up: bool = False, progress_bar: bool = True
    ) -> Tuple[Dict[str, numpy.ndarray], numpy.ndarray]:
        """Propagate the simulation forward by the specified number of
        `steps`. If these are flagged as `warm_up` steps, all data generated
        will be discarded and the in-model sampler will attempt to tune itself.

        Parameters
        ----------
        steps: int
            The number of steps to take.
        warm_up: bool
            Whether the treat these steps as 'warm-up' or
            'equilibration' steps.
        progress_bar: bool or tqdm.tqdm
            If False, no progress bar is printed to the terminal. If True,
            a default progress bar is printed to the terminal. If an existing
            `tqdm` progress bar, this will be used instead if the default.

        Returns
        -------
        dict of str and numpy.ndarray
            The final model parameters.
        numpy.ndarray
            The final value of log p
        """

        if not warm_up:
            self._has_stepped = True

        # Make sure we don't equilibrate after having already performed
        # some production steps.
        if self._has_stepped and warm_up:
            raise ValueError("The warm-up phase must come before the production phase.")

        if progress_bar is True:
            progress_bar = tqdm(total=steps + 1)

        # Initialize the starting values.
        current_parameters = {x: y.copy() for x, y in self._initial_values.items()}
        current_log_p, _ = self._evaluate_log_p(current_parameters)

        for i in range(steps):

            # Propagate the simulation one step forward.
            current_parameters, current_log_p, acceptance = self._step(
                current_parameters, current_log_p, warm_up,
            )

            # Update the bookkeeping.
            if not warm_up:

                for parameter_label in current_parameters:
                    self._trace[parameter_label].append(
                        current_parameters[parameter_label]
                    )

                self._log_p_trace.append(current_log_p)

            if progress_bar is not None and progress_bar is not False:
                progress_bar.update()

        if warm_up:
            self._sampler.reset_counters()

        self._initial_values = current_parameters

        return current_parameters, current_log_p

    def _step(
        self,
        current_parameters: Dict[str, numpy.ndarray],
        current_log_p: numpy.ndarray,
        adapt_moves: bool = False,
    ) -> Tuple[Dict[str, numpy.ndarray], numpy.ndarray, bool]:
        """Propagates the simulation forward a single step.

        Parameters
        ----------
        current_parameters: dict of str and numpy.ndarray
            The current model parameters.
        current_log_p: numpy.ndarray
            The current value of log p.
        adapt_moves: bool
            If True, the in-model sampler will be allowed to tune itself.

        Returns
        -------
        dict of str and numpy.ndarray
            The new model parameters.
        numpy.ndarray
            The new value of log p.
        bool
            Whether this move was accepted or not.
        """

        proposed_parameters, proposed_log_p, acceptance = self._sampler.step(
            current_parameters, current_log_p, adapt_moves
        )

        return proposed_parameters, proposed_log_p, acceptance

    def _evaluate_log_p(self, parameters: Dict[str, numpy.ndarray]) -> Tuple[numpy.ndarray, Dict[str, numpy.ndarray]]:
        """Evaluates the (possibly un-normalized) target distribution
        for the given set of parameters.

        Parameters
        ----------
        parameters: dict of str and numpy.ndarray
            The parameters to evaluate at.

        Returns
        -------
        numpy.ndarray
            The evaluated log p (x).
        dict of str and numpy.ndarray
            The gradient of log p with respect to the parameters.
        """
        return self._model.evaluate(parameters)

    def save_results(self, directory_path: str = ""):
        """Saves the results of this simulation to disk.

        Returns
        -------
        directory_path: str
            The directory to save the results into.
        """

        # Make sure the output directory exists
        if len(directory_path) > 0:
            os.makedirs(directory_path, exist_ok=True)

        # Save the traces
        self._save_traces(directory_path)

        # Save the move statistics
        self._save_statistics(directory_path)

    def _save_statistics(self, directory_path: str):
        """Save statistics about the simulation.

        Parameters
        ----------
        directory_path: str
            The directory to save the results into.
        """
        results = {
            "random_seed": self._random_seed,
            "sampler_statistics": self._sampler.get_statistics_dictionary(),
        }

        filename = os.path.join(directory_path, "statistics.json")

        with open(filename, "w") as file:
            json.dump(results, file, sort_keys=True, indent=4, separators=(",", ": "))

    def _save_traces(self, directory_path: str):
        """Saves the raw traces, as well as plots of the traces
        to the output directory.

        Parameters
        ----------
        directory_path: str
            The directory to save the results into.
        """

        if len(directory_path) > 0:
            os.makedirs(directory_path, exist_ok=True)

        numpy.save(os.path.join(directory_path, "trace.npy"), self._trace)
        numpy.save(os.path.join(directory_path, "log_p.npy"), self._log_p_trace)

    def run(
        self,
        warm_up_steps: int,
        steps,
        progress_bar: bool = True,
        output_directory: str = "",
    ) -> Tuple[Dict[str, numpy.ndarray], numpy.ndarray]:
        """A convenience function to run a production simulation
        after an initial warm-up simulation, and save the output to
        a given directory.

        Parameters
        ----------
        warm_up_steps: int
            The number of warm-up steps to take. During this time all
            move proposals will be tuned.
        steps: int
            The number of steps which the simulation should run for.
        progress_bar: bool
            If False, no progress bar is printed to the terminal. If True,
            a default progress bar is printed to the terminal.
        output_directory: str
            The path to save the simulation results in.

        Returns
        -------
        dict of str and numpy.ndarray
            A trajectory of the model parameters over the course
            of the simulation with shape=(n_steps, n_trainable_parameters+1),
            where the 'first parameter is the model index'.
        numpy.ndarray:
            A trajectory of the value of log p over the course
            of the simulation with shape=(n_steps,).
        """

        print("==============================")
        print("Warm-up simulation:")
        self.propagate(warm_up_steps, True, progress_bar)
        print("==============================")

        print("==============================")
        print("Production simulation:")
        self.propagate(steps, False, progress_bar)
        print("==============================")

        self.save_results(output_directory)

        return self.trace, self.log_p_trace
