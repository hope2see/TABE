import os
import torch
from tabe.models.abstractmodel import AbstractModel
from tabe.utils.logger import logger
from transformers import AutoModelForCausalLM


from tabe.utils.mem_util import MemUtil
_mem_util = MemUtil(rss_mem=False, python_mem=False)


class TimeMoE(AbstractModel):
    MAX_CONTEXT_LEN = 4096 # limitation of TimeMoE architecture

    def __init__(self, configs, device, name='TimeMoE', ds_size='large'):
        super().__init__(configs, name)
        self.device = device
        model_path = \
            'Maple728/TimeMoE-50M' if ds_size == 'base' else  \
            'Maple728/TimeMoE-200M' if ds_size == 'large' else \
            'Maple728/Time-300B' # not available! 
            
        try:
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map=device,
                # attn_implementation='flash_attention_2',
                torch_dtype='auto',
            )
        except:
            logger.info('Got exception during creating TimeMoeForPrediction!, So use AutoModelForCausalLM instead.')
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                # attn_implementation='flash_attention_2',
                torch_dtype='auto',
                trust_remote_code=True,
            )

        logger.info(f'>>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}')

        self.model = model
        self.device = device
        self.model.eval()


    # NOTE
    # Pretrained model. So, it not necessary to train the model.
    # However, it would be better to fine-tune the model with the dataset.
    # TODO : Fine-tune
    def train(self):
        pass

    def test(self):
        raise NotImplementedError

    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, training: bool = False):
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1
        assert batch_x.shape[1] <= self.MAX_CONTEXT_LEN, f'Exceeded TimeMoE\'s Max context length!'

        # Shapes of given parameters
        #   : batch_x, batch_x_mark  = (batch_len=1, seq_len, feature_dim)
        #   : batch_y, batch_y_mark  = (batch_len=1, seq_len + pred_len, feature_dim)
        # TimeMoE expects inputs with the shape below (from source of TimeMoeModel clsss).
        #   : input_ids is the input of time series, its shape is [batch_size, seq_len, 'input_size']
        #  However, actually it does not support 'input_size' (feature dimension) in 
        #   timemoe.model.ts_generation_mixin.TSGenerationMixin._greedy_search(), which expects 
        #   input_ids.shape = [batch_size, cur_len]
        #  It means that TimeMoE supports only single variate forecasting, yet. 

        # Also, Note the comment in transformers.generation.utils.GenerationMixin.generate()
        # inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
        #     The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
        #     method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
        #     should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
        #     `input_ids`, `input_values`, `input_features`, or `pixel_values`.

        # Reshape batch_x from (batch_len, seq_len, feature_dim) to (batch_len, seq_len)
        batch_x = batch_x[:, :, -1] # target feature only
        
        outputs = self.model.generate(
            inputs=batch_x.to(self.device).to(self.model.dtype),
            max_new_tokens=1, # prediction_length
        )
        y_hat = outputs[0, -1].item()

        # calculate the actuall loss of next timestep
        y = batch_y[0, -1:, -1] 
        loss = self.criterion(torch.tensor(y_hat), y).item()

        if training: # TODO 
            pass

        return y_hat, loss


    def predict(self, seqs, context_len=None, pred_len=1):
        context_len = len(seqs) if context_len is None else context_len
        context_len = min(context_len, self.MAX_CONTEXT_LEN)
        seqs = seqs[-context_len:]

        # normalize seqs
        mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
        normed_seqs = (seqs - mean) / std

        output = self.model.generate(normed_seqs, max_new_tokens=pred_len)  
        normed_predictions = output[:, -pred_len:]  

        # inverse normalize
        preds = normed_predictions * std + mean
        return preds
