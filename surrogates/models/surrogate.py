import logging

import gpytorch
import torch

from surrogates.models import SurrogateModel

logger = logging.getLogger(__name__)


class GaussianProcess(SurrogateModel):
    """A model which evaluates a trained Gaussian Process based on a radial-basis
    function kernel. The Gaussian Process may be retrained with extra data on the
    fly.
    """

    class _ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):

            super(GaussianProcess._ExactGPModel, self).__init__(
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
        parameter_labels,
        condition_parameters,
        condition_data,
        learning_rate=0.1,
        train_iterations=200,
    ):

        super(GaussianProcess, self).__init__(
            parameter_labels, condition_parameters, condition_data,
        )

        self._model = None
        self._likelihood = None

        self._learning_rate = learning_rate
        self._train_iterations = train_iterations

    def _retrain(self):

        values = self._training_values.reshape((-1,))
        noise = self._training_uncertainties.reshape((-1,))

        # Initialize likelihood and model
        self._likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise, False
        )
        self._model = self._ExactGPModel(
            self._training_parameters, values, self._likelihood
        )

        # Find optimal model hyperparameters
        self._model.train()
        self._likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            [{"params": self._model.parameters()}], lr=self._learning_rate
        )

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        for i in range(self._train_iterations):

            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Output from model
            output = self._model(self._training_parameters)

            # Calc loss and backprop gradients
            loss = -mll(output, values)
            loss.backward()

            logger.debug(
                "Iter %d/%d - Loss: %.5f   lengthscale: %.5f   outputscale: %.5f"
                % (
                    i + 1,
                    self._train_iterations,
                    loss.item(),
                    self._model.covar_module.base_kernel.lengthscale.item(),
                    self._model.covar_module.outputscale.item(),
                )
            )

            optimizer.step()

        # Get into evaluation (predictive posterior) mode
        self._model.eval()
        self._likelihood.eval()

    def evaluate(self, parameters):

        if self._model is None:
            raise ValueError("The model has not yet been trained upon any data.")

        parameters = self._parameter_dict_to_tensor(parameters)
        parameters = (parameters - self._parameter_shift) / self._parameter_scale

        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            prediction = self._likelihood(self._model(parameters))

            values = (prediction.mean * self._value_scale + self._value_shift).numpy()
            uncertainties = (prediction.stddev * self._value_scale).numpy()

        return values, uncertainties
