from dataset.IOI_template import *
import torch as t
from torch.utils.data import TensorDataset


class IOI_dataset():
    def __init__(
        self, 
        N:int=500, 
        patching_method:str="path", 
        tokenizer=None, 
        device:str="cuda", 
        seed:int=1234, 
        prepend_bos=False, 
        model_name="gpt2", 
        remove_target_token=False
        
        ) -> None:   
        
        print("using", model_name, "model in IOI dataset")

        if tokenizer is None:
            print("no tokenizer selected, choose gpt2 tokenizer per default")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        else:
            self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.N = N
        self.device = device
        self.clean_dataset = IOIDataset(
            model_name=model_name,
            prompt_type="mixed",
            N=self.N,
            tokenizer= self.tokenizer,
            prepend_bos=prepend_bos,
            seed=seed,
            device=self.device,
            remove_target_token=remove_target_token
            )
        
        self.clean_input = self.clean_dataset.sentences
        self.groups = self.clean_dataset.groups
        self.word_idx_dict = self.clean_dataset.word_idx
        if patching_method == "activation":
            print("activation patching")
            self.corrupted_dataset =  self.clean_dataset.gen_flipped_prompts("ABB->ABA, BAB->BAA", device=self.device)
            # flipp the prompts
            self.clean_tokens, self.corrupted_tokens, self.answer_tokens = self.process_dataset_for_activation_patching()
            self.target_idx =  t.stack((t.arange(self.clean_tokens.size(0)), self.word_idx_dict["end"].repeat_interleave(2)))
        else:
            
            # use abc distribution
            self.corrupted_dataset = self.clean_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ", device=self.device)
            self.clean_tokens, self.corrupted_tokens, self.answer_tokens = self.process_dataset_for_path_patching()
            self.target_idx =t.stack((t.arange(
                                self.clean_tokens.size(0)), 
                                self.word_idx_dict["target"]), dim=1
                                    )
            self.corrupted_input = self.corrupted_dataset.sentences
            
        self.max_len = self.clean_tokens.shape[1]
        self.attention_mask = self.clean_dataset.attention_mask
        
        self.start = self.word_idx_dict["starts"]
        self.end = self.word_idx_dict["ends"]
        
        if not t.all(self.clean_tokens[self.target_idx[:, 0], self.target_idx[:, 1] + 1] == self.answer_tokens[:, 0]):
            print("clean word idx", self.clean_dataset.word_idx)
            print("corrupted word idx", self.corrupted_dataset.word_idx)
            print(tokenizer.batch_decode(self.answer_tokens[:, 0]))
            print(tokenizer.batch_decode(self.clean_tokens[self.target_idx[:, 0], self.target_idx[:, 1] + 1]))
            raise Exception("Target idx does not align with the position of the IOI in at least one of the senteces.")
    
        self.correct_answers = self.answer_tokens[:, 0]
        self.wrong_answers = self.answer_tokens[:, 1]
        self.dataset = TensorDataset(
            self.clean_tokens,        # [N, seq_len], 
            self.corrupted_tokens,    # [N, seq_len]
            self.attention_mask,      # [N, seq_len]
            self.correct_answers,     # [N, 1]
            self.wrong_answers,       # [N, x]            
            self.target_idx,          # [N, 2]
        )
        
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx): 
        return {
            "clean_tokens": self.clean_tokens[idx],
            "corrupted_tokens": self.corrupted_tokens[idx],
            "mask": self.attention_mask[idx],
            "correct_answers": self.correct_answers[idx],
            "wrong_answers": self.wrong_answers[idx],
            "target_idx": self.target_idx[idx],
        }
        
    def process_dataset_for_activation_patching(self):
        clean_t = self.clean_dataset.toks
        corrupted_t = self.corrupted_dataset.toks
        
        if self.N < 25:
            # artificially balance the dataset, by including for each ABB prompt its reversed BAA (BAB -> ABA) prompt
            # Swap each adjacent pair to get corrupted tokens
            clean_tokens = t.stack((clean_t,corrupted_t), dim=1).view(2*len(clean_t), len(clean_t[0]))
            clean_tokens = self.remove_target_token(clean_tokens).to(self.device)
            corrupted_tokens = t.stack((corrupted_t, clean_t), dim=1).view(2*len(clean_t), len(clean_t[0]))
            corrupted_tokens = self.remove_target_token(corrupted_tokens).to(self.device)
            answer_tokens = t.zeros([len(clean_tokens), 2], dtype=t.int64).to(self.device)
            answer_tokens[:, 0] = t.stack((t.Tensor(self.clean_dataset.io_tokenIDs), t.Tensor(self.clean_dataset.s_tokenIDs)), dim=1).view(len(answer_tokens))
            answer_tokens[:, 1] = t.stack((t.Tensor(self.clean_dataset.s_tokenIDs), t.Tensor(self.clean_dataset.io_tokenIDs)), dim=1).view(len(answer_tokens))

        else:
            clean_tokens = self.remove_target_token(clean_t).to(self.device)
            corrupted_tokens = self.remove_target_token(corrupted_t).to(self.device)
            answer_tokens = t.zeros([len(clean_tokens), 2], dtype=t.int64).to(self.device)
            answer_tokens[:, 0] = t.Tensor(self.clean_dataset.io_tokenIDs)
            answer_tokens[:, 1] = t.Tensor(self.clean_dataset.s_tokenIDs)
        return clean_tokens, corrupted_tokens, answer_tokens
    
    def process_dataset_for_path_patching(self):
        clean_tokens = self.clean_dataset.toks.to(self.device)
        corrupted_tokens  = self.corrupted_dataset.toks.to(self.device)
        
        #clean_tokens = self.remove_target_token(clean_tokens)
        #corrupted_tokens = self.remove_target_token(corrupted_tokens)
        
        answer_tokens = t.zeros([len(clean_tokens), 2], dtype=t.int64).to(self.device)
        answer_tokens[:, 0] = t.Tensor(self.clean_dataset.io_tokenIDs) # correct labels
        answer_tokens[:, 1] = t.Tensor(self.clean_dataset.s_tokenIDs)  # wrong labels
        return clean_tokens, corrupted_tokens, answer_tokens      
    
    def remove_target_token(self, toks):
        # remove the target IO-token of the prompt
        batch, pos = toks.shape
        res = t.zeros((batch, pos-1), dtype=t.int64)
        bos_token = self.tokenizer.encode(self.tokenizer.bos_token)[0] 
        for phrase_idx in range(len(toks)):
            for next_tok in range(1, pos):
                if toks[phrase_idx, next_tok] == bos_token:
                    res[phrase_idx, next_tok-1] = bos_token
                else:
                    res[phrase_idx, next_tok-1] = toks[phrase_idx, next_tok-1] 
        return res