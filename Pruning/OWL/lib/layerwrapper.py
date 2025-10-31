import torch
import torch.nn as nn
import transformers
import einops

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.k
    """

    def __init__(self, layer, layer_id=0, layer_name="none", is_gpt=False, device="cuda"):
        
        self.layer = layer
        self.is_gpt = is_gpt
        if is_gpt:
            self.dev = self.layer.weight.device
            self.rows = layer.weight.data.shape[1]
            self.columns = layer.weight.data.shape[0]
        else:
            self.dev = device
            self.rows = layer.weight.data.shape[0]
            self.columns = layer.weight.data.shape[1]
        
        # important: self.columns has to be the same size as model dimension!
        self.scaler_row = torch.zeros((self.columns), device=self.dev).to(self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        self.inp = inp
        #print("nsamples", self.nsamples)
      
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        
        if isinstance(self.layer, nn.Linear)  or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            #print("wrapper input", inp.shape)
            #print("wrapper input", inp[:10, :10])

            inp = inp.t()
        #print("scaler row before addition", self.scaler_row[:10])

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp
        inp = inp.type(torch.float32).to(self.dev)
        norm = torch.norm(inp, p=2, dim=1)**2
        #print("norm", norm[:10])        
        self.scaler_row +=  torch.norm(inp, p=2, dim=1)**2 / self.nsamples
        #print("scaler_row", self.scaler_row[:10])