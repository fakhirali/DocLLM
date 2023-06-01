#this file contains useful functions for the project
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM
import torch
import torch.nn.functional as F


#Agent class
#the agent is responsible for performing a task
#the agent will have some instructions in the form of a pre_prompt
#the agent will be fed a text input on it the agent will have to perform the task

class Agent:
    def __init__(self, model,tokenizer,  pre_prompt, break_words, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.pre_prompt = pre_prompt
        encoded_prompt = tokenizer.encode(pre_prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.forward(input_ids=encoded_prompt.to(device))
        self.prompt_key_vals = out.past_key_values
        self.break_words = break_words
        pass
        

    @torch.no_grad()
    def generate_response_greedy(self, input_text, max_length=2048,temp=0.7, verbose=False, name='<|ASSISTANT|>'):
        
        inputs = self.tokenizer.encode(input_text + '\n' + name, return_tensors="pt")
        response_ids = inputs
        length_prompt = len(response_ids[0])
        output = ''
        last_n = ''
        all_out = ''
        past_key_vals = self.prompt_key_vals
        for _ in (range(max_length)):
            out = self.model.forward(input_ids=response_ids.to(self.device), past_key_values=past_key_vals)
            next_token_id = torch.multinomial(F.softmax(out.logits[:, -1, :]/temp,  dim=-1), num_samples=1).to('cpu')
            past_key_vals = out.past_key_values
            response_ids = next_token_id
            output = self.tokenizer.decode([response_ids[0][-1]], skip_special_tokens=False)
            if verbose:
                print(output, end='')
            all_out += output
            br = False
            for b in self.break_words:
                if all_out.rstrip().endswith(b):
                    # print()
                    # print()
                    all_out = all_out.replace(b, '')
                    br = True
            if br:
                break
        decoded_output = self.tokenizer.decode(response_ids[0], skip_special_tokens=False)
        past_kv = past_key_vals
        next_id = response_ids
        return all_out


# torch.set_num_threads(8)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_name = 'EleutherAI/gpt-neo-125M'
# model = AutoModelForCausalLM.from_pretrained(model_name,
#                                              torch_dtype=torch.bfloat16,
#                                             low_cpu_mem_usage=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model.to(device)
# pass

# prompt = f'''
# You are a relevancy checker bot. Your job is to evaluate whether the given user input is a valid input in the
# current context. You are dealing with users of an e-commerce website. The users are here to buy things, ask for the price, rating, etc.
# Anything the user asks for other than what is relevant should be flagged as not relavant. You also provide a reason for your decision. Your decisions are either Valid or Invalid. Anything that is not related to products should be Invalid. Anything related to
# price, ratings, or reviews is relevant.

# Examples:

# <|USER|> What is the name of the most expensive product?
# <|BOT|> Reason: The user asked for details related to our products, specifically about the most expensive product, which is relevant to the e-commerce context as it helps them find high-priced items.
# Decision: relevant

# <|USER|> What is the capital of France?
# <|BOT|> Reason: The user asked about the capital of France, which is a geographical question and has no relation to the current context of an e-commerce website focused on selling products.
# Decision: not relevant

# <|USER|> Can you show me the top-rated products in the electronics category?
# <|BOT|> Reason: The user is inquiring about top-rated products within a specific category (electronics), which is relevant to the e-commerce context as it helps them find high-quality products and make informed decisions.
# Decision: relevant

# <|USER|> Who won the world cup?
# <|BOT|> Reason: The user asked about the winner of the world cup, which is a sports-related question and has no relation to the current context of an e-commerce website where the focus is on selling products.
# Decision: not relevant

# <|USER|> How is the weather?
# <|BOT|> Reason: The user asked how the weather is, which is a weather-related question and has no relation to the current context of an e-commerce website focused on selling products.
# Decision: not relevant

# <|USER|>'''


# guard_agent = Agent(model, tokenizer, prompt,break_words=['<|USER|>', '---------',
#                                                '<|BOT|>'], device=device)
# a = guard_agent.generate_response_greedy('How are you.', 
#                                          verbose=True, temp=0.3, name='<|BOT|>')