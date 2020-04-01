import numpy
from matplotlib import pyplot

from surrogates.kernels import MCMCSimulation
from surrogates.kernels.samplers.hmc import Hamiltonian
from surrogates.models.simple import UnconditionedModel
from surrogates.utils.distributions import Normal
from surrogates.utils.file import change_directory
from surrogates.utils.plotting import plot_log_p, plot_trace, plot_corner


def main():

    std_devs = {
        "a": 0.05,
        "b": 50.0,
        "c": 5000.0
    }

    priors = {
        "a": Normal(numpy.array([0.0]), numpy.array([std_devs["a"]])),
        "b": Normal(numpy.array([100.0]), numpy.array([std_devs["b"]])),
        "c": Normal(numpy.array([0.0]), numpy.array([std_devs["c"]])),
    }
    model = UnconditionedModel(priors)

    # Construct and run the simulation object.
    initial_parameters = {
        "a": numpy.array([0.0]),
        "b": numpy.array([0.0]),
        "c": numpy.array([0.0]),
    }

    # Setup the sampler
    sampler = Hamiltonian(
        model,
        momentum_scales={
            "a": numpy.array([1.0 / std_devs["a"]]),
            "b": numpy.array([1.0 / std_devs["b"]]),
            "c": numpy.array([1.0 / std_devs["c"]]),
        },
        step_size=1.0,
        n_steps=10
    )

    # Run the simulation.
    with change_directory("3d_univariate"):

        simulation = MCMCSimulation(
            model, initial_parameters, sampler=sampler, random_seed=42
        )
        simulation.run(2000, 20000)

        # Plot the output.
        trace_figure = plot_trace(simulation.trace, show=False)
        trace_figure.savefig("trace.png")
        pyplot.close(trace_figure)

        corner_figure = plot_corner(
            simulation.trace, model.trainable_parameters, show=False
        )
        corner_figure.savefig("corner.png")
        pyplot.close(corner_figure)

        log_p_figure = plot_log_p(simulation.log_p_trace, show=False)
        log_p_figure.savefig("log_p.png")
        pyplot.close(log_p_figure)

    for label in std_devs:

        print(
            f"{label}: std estimated={numpy.std(simulation.trace[label])} "
            f"real={std_devs[label]}"
        )
        print(
            f"{label}: mean estimated={numpy.mean(simulation.trace[label])} real={0.0}"
        )


if __name__ == "__main__":
    main()
