import torch
from tabe.models.abstractmodel import AbstractModel
from transformers import AutoModelForCausalLM
from tabe.utils.logger import logger
from tabe.utils.mem_util import MemUtil

_mem_util = MemUtil(rss_mem=False, python_mem=False)


class Timer(AbstractModel):
    MAX_CONTEXT_LEN = 4096 # limitation of TimeMoE architecture

    def __init__(self, configs, device, name='Timer'):
        super().__init__(configs, name)
        self.device = device
        checkpoint_path = 'thuml/timer-base-84m'
            
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map=device,
            torch_dtype='auto',
            trust_remote_code=True,
        )

        logger.info(f'>>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}')

        self.model = model
        self.device = device
        # self.model.eval()


    # TODO : Fine-tune
    def train(self):
        pass

    def test(self):
        raise NotImplementedError

    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, training: bool = False):
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1
        assert batch_x.shape[1] <= self.MAX_CONTEXT_LEN, f'Exceeded TimeMoE\'s Max context length!'

        # Reshape batch_x from (batch_len, seq_len, feature_dim) to (batch_len, seq_len)
        batch_x = batch_x[:, :, -1] # target feature only

        # NOTE 
        # input_token_len of default (or pretrained) config of Timer model is set to be 96 !! 
        # Input length must be at least Timer's config.input_token_len
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
