from llama_cpp import Llama

class LLM:
    def __init__(self):
        self.llm = Llama(
            model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
            chat_format="llama-2",
            n_gpu_layers=-1,
            n_ctx=4096
        )
        self.messages = [
            {"role": "system",
             "content": "You are an assistant who answers a user's question based on the information provided."},
        ]
    def get_response(self, question, information):
        query = f'''
        Question: {question}
        Information: {information}
        '''
        self.messages.append({"role": "user", "content": query})
        out = self.llm.create_chat_completion(messages=self.messages)
        response = out["choices"][-1]['message']["content"]
        return response

if __name__ == "__main__":
    llm = Llama(
        model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
        chat_format="llama-2",
        n_gpu_layers=-1,
        n_ctx=4096
    )
    messages = [
        {"role": "system",
         "content": "You are an assistant who answers a user's question based on the information provided."},
        ]



    q = '''
    Question: What are habits?

    Information:
    ” His work provides the perfect starting point for
    discussing how habits form in our own lives  It also provides answers
    to some fundamental questions like: What are habits? And why does
    the brain bother building them at all?
    WHY YOUR BRAIN BUILDS HABITS
    A habit is a behavior that has been repeated enough times to become
    automatic
    --------
    Once you have a full list, look at each behavior, and ask yourself, “Is
    this a good habit, a bad habit, or a neutral habit?” If it is a good habit,
    write “+” next to it  If it is a bad habit, write If it is a neutral habit,
    write “=”
    --------
    If you’re still having trouble determining how to rate a particular
    habit, here is a question I like to use: “Does this behavior help me
    become the type of person I wish to be? Does this habit cast a vote for
    or against my desired identity?” Habits that reinforce your desired
    identity are usually good  Habits that conflict with your desired
    identity are usually bad
    '''

    messages.append({"role":"user", "content":q})
    out = llm.create_chat_completion(messages=messages)
    print(out)
    response = out["choices"][-1]['message']["content"]
    print(response)
