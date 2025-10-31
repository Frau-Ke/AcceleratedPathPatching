import torch as t
import requests

# Checkout https://arxiv.org/pdf/2309.16042 for more details 
class PairedFacts():
    def __init__(self, N, patching_method, tokenizer, device) -> None:
        self.N = N
        self.device = device
        response = requests.get('https://www.jsonkeeper.com/b/P1GL',)
        self.clean_dataset = response.json()
        indices = t.randint(0, len(self.clean_dataset), (N,))
        self.clean_tokens, self.corrupted_tokens, self.answer_tokens = self.process_dataset(indices, tokenizer)
        
        
    def process_dataset(self, indices, tokenizer):
    
        if self.N < 25:
            prompts = []
            answers = []
            for i in indices:
                prompts.append(self.clean_dataset[i]["pair"][0])  
                prompts.append(self.clean_dataset[i]["pair"][1])  
                answers.append(self.clean_dataset[i]["answer"])
                answers.append(list(reversed(self.clean_dataset[i]["answer"])))

            clean_tokens = t.Tensor(tokenizer(prompts, padding=True).input_ids).long()
            answer_tokens = t.Tensor(tokenizer(answers, padding=True).input_ids).long()
            
            indices = [i+1 if i % 2 == 0 else i-1 for i in range(len(clean_tokens))]
            corrupted_tokens = clean_tokens[indices]
        else:
            clean_t = []
            corrupted_t = []
            answers = []
            for i in indices:
                clean_t.append(self.clean_dataset[i]["pair"][0])  
                corrupted_t.append(self.clean_dataset[i]["pair"][1])  
                answers.append(self.clean_dataset[i]["answer"])

            clean_tokens = t.Tensor(tokenizer(clean_t, padding=True).input_ids).long().to(self.device)
            corrupted_t =  t.Tensor(tokenizer(corrupted_t, padding=True).input_ids).long().to(self.device)
            answer_tokens = t.Tensor(tokenizer(answers, padding=True).input_ids).long().to(self.device)
        return clean_tokens, corrupted_tokens, answer_tokens