import gpytorch
import torch

from surrogates.models import TrainableModel


class GaussianProcessModel(TrainableModel):
    """A model which evaluates a trained Gaussian Process based on a radial-basis
    function kernel. The Gaussian Process may be retrained with extra data on the
    fly.
    """

    class _ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):

            super(GaussianProcessModel._ExactGPModel, self).__init__(
                train_x, train_y, likelihood
            )

            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):

            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)

            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __init__(
        self,
        priors,
        fixed_parameters,
        condition_parameters,
        condition_data,
        learning_rate=0.1,
        train_iterations=200,
    ):

        super(GaussianProcessModel, self).__init__(
            priors, fixed_parameters, condition_parameters, condition_data,
        )

        self._models = {}
        self._likelihoods = {}

        self._learning_rate = learning_rate
        self._train_iterations = train_iterations

    def _validate_training_data(self, parameters, values, uncertainties):

        super(GaussianProcessModel, self)._validate_training_data(
            parameters, values, uncertainties
        )

        # Make sure we are training upon the same data if we have already trained
        # at least once.
        if len(self._models) == 0:
            return

        if not all(x in self._models for x in values) or not all(
            x in values for x in self._models
        ):

            raise ValueError(
                "The data types to train on do not match the data type the model has "
                "already been trained upon."
            )

    def add_training_data(self, parameters, values, uncertainties):

        super(GaussianProcessModel, self).add_training_data(
            parameters, values, uncertainties
        )

        self._retrain()

    def _retrain(self):
        """Re-train the models hyperparameters based on the currently available
        training data.

        Notes
        -----
        This function is based in the version found in a scikit-learn tutorial:
        https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html
        """

        self._models = {}
        self._likelihoods = {}

        for label, values in self._training_values.items():

            values = values.reshape((-1,))
            noise = self._training_uncertainties[label].reshape((-1,))

            # Initialize likelihood and model
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise, False)
            model = self._ExactGPModel(self._training_parameters, values, likelihood)

            # Find optimal model hyperparameters
            model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(
                [{"params": model.parameters()}], lr=self._learning_rate
            )

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for i in range(self._train_iterations):

                # Zero gradients from previous iteration
                optimizer.zero_grad()

                # Output from model
                output = model(self._training_parameters)

                # Calc loss and backprop gradients
                loss = -mll(output, values)
                loss.backward()

                print(
                    "Iter %d/%d - Loss: %.5f   lengthscale: %.5f   outputscale: %.5f"
                    % (
                        i + 1,
                        self._train_iterations,
                        loss.item(),
                        model.covar_module.base_kernel.lengthscale.item(),
                        model.covar_module.outputscale.item(),
                        # model.likelihood.noise.item(),
                    )
                )

                optimizer.step()

            # Get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()

            self._models[label] = model
            self._likelihoods[label] = likelihood

    def evaluate(self, properties, parameters):

        if len(self._models) == 0:
            raise ValueError("The model has not yet been trained upon any data.")

        parameters = torch.from_numpy(parameters)
        parameters = (parameters - self._parameter_shift) / self._parameter_scale

        values = {}
        uncertainties = {}

        for property_type in properties:

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                prediction = self._likelihoods[property_type](
                    self._models[property_type](parameters)
                )

                values[property_type] = (
                    prediction.mean * self._value_scales[property_type]
                    + self._value_shifts[property_type]
                ).numpy()
                uncertainties[property_type] = (
                    prediction.stddev * self._value_scales[property_type]
                ).numpy()

        return values, uncertainties
