import torch
from tabe.models.abstractmodel import AbstractModel
from transformers import AutoModelForCausalLM
from tabe.utils.logger import logger
import timesfm

# See
# https://github.com/google-research/timesfm
# https://huggingface.co/google/timesfm-2.0-500m-pytorch

# Note that the five parameters are fixed to load the 500m model
# input_patch_len=32,
# output_patch_len=128,
# num_layers=50,
# model_dims=1280,
# use_positional_embedding=False,


class TimesFM(AbstractModel):
    MAX_CONTEXT_LEN = 2048 # limitation of TimesFM 2.0

    def __init__(self, configs, device=None, name='TimesFM'):
        super().__init__(configs, device, name)
        self.device = device
        self.checkpoint_path = 'google/timesfm-2.0-500m-pytorch'
        
        assert configs.seq_len % 32 == 0, 'seq_len must be a multiple of 32(input_patch_len)'
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend='gpu' if configs.use_gpu else 'cpu', # backend : one of "cpu", "gpu", case sensitive.
                per_core_batch_size=configs.batch_size, # 32,                
                horizon_len=1, # Can be set to anything, but recommend <= context length 

                num_layers=50, # fixed to load the 500m model
                use_positional_embedding=False, # fixed to load the 500m model

                # context_len can be set as the max context length of the model. 
                # Or, it needs to be a multiplier of input_patch_len, i.e. a multiplier of 32
                # context_len=self.MAX_CONTEXT_LEN, 
                context_len=configs.seq_len, 
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.checkpoint_path),
        )

    # TODO : Fine-tune
    def train(self):
        pass

    def test(self):
        raise NotImplementedError

    def _forward_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        assert batch_x.shape[1] <= self.MAX_CONTEXT_LEN, f'Exceeded TimesFM\'s Max context length!'

        # Given: batch_x.shape = (batch_len=1, seq_len, feature_dim)
        # Reshape batch_x from (batch_len, seq_len, feature_dim) to (batch_len, seq_len)
        batch_x = batch_x[:, :, -1] # target feature only

        #   def forecast(
        #       self,
        #       inputs: Sequence[Any],
        #       freq: Sequence[int] | None = None,
        #       window_size: int | None = None,
        #       forecast_context_len: int | None = None,
        #       return_forecast_on_context: bool = False,
        #       normalize: bool = False,
        #   ) -> tuple[np.ndarray, np.ndarray]:
        #     """Forecasts on a list of time series.

        #     Args:
        #       inputs: list of time series forecast contexts. Each context time series
        #         should be in a format convertible to JTensor by `jnp.array`.
        #       freq: frequency of each context time series. 0 for high frequency
        #         (default), 1 for medium, and 2 for low. Notice this is different from
        #         the `freq` required by `forecast_on_df`.
        #       window_size: window size of trend + residual decomposition. If None then
        #         we do not do decomposition.
        #       forecast_context_len: optional max context length.
        #       return_forecast_on_context: True to return the forecast on the context
        #         when available, i.e. after the first input patch.
        #       normalize: If True, then we normalize the inputs before forecasting and
        #         the outputs are then renormalized to the original scale.

        #     Returns:
        #     A tuple for np.array:
        #     - the mean forecast of size (# inputs, # forecast horizon),
        #     - the full forecast (mean + quantiles) of size
        #         (# inputs,  # forecast horizon, 1 + # quantiles).

        point_forecast, quantile_forecast = self.model.forecast(
            # batch_x.to(self.device).to(self.model.dtype),
            batch_x,
            forecast_context_len = batch_x.shape[1]
        )
        pred = point_forecast[0][0]

        # TODO : Training with the new data 

        return pred
