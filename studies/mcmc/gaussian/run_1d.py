import numpy
from matplotlib import pyplot

from surrogates.kernels import MCMCSimulation
from surrogates.kernels.samplers.hmc import Hamiltonian
from surrogates.models.simple import UnconditionedModel
from surrogates.utils.distributions import Normal
from surrogates.utils.file import change_directory
from surrogates.utils.plotting import plot_log_p, plot_trace


def main():

    std_dev = 500.0

    priors = {"a": Normal(numpy.array([0.0]), numpy.array([std_dev]))}
    model = UnconditionedModel(priors)

    # Construct and run the simulation object.
    initial_parameters = {"a": numpy.array([0.0])}

    # Setup the sampler
    sampler = Hamiltonian(
        model,
        momentum_scales={"a": numpy.array([1.0 / std_dev])},
        step_size=1.0,
        n_steps=10
    )

    # Run the simulation.
    with change_directory("1d"):

        simulation = MCMCSimulation(
            model, initial_parameters, sampler=sampler, random_seed=42
        )
        simulation.run(2000, 10000)

        # Plot the output.
        trace_figure = plot_trace(simulation.trace, show=False)
        trace_figure.savefig("trace.png")
        pyplot.close(trace_figure)

        log_p_figure = plot_log_p(simulation.log_p_trace, show=False)
        log_p_figure.savefig("log_p.png")
        pyplot.close(log_p_figure)

    print(f"std estimated={numpy.std(simulation.trace['a'])} real={std_dev}")
    print(f"mean estimated={numpy.mean(simulation.trace['a'])} real={0.0}")


if __name__ == "__main__":
    main()
