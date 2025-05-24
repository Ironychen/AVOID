import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from zhipuai  import ZhipuAI
import random

key = "653d34becfd62e46c1a02e7642e43c02.eqLfDKBj4GJdnOgH" 

class GLMAgent:
    def __init__(self, name, system_prompt=None, short_mem_size=3):
        self.name = name
        self.short_mem_size = short_mem_size  # Number of recent dialogues to keep (hyperparameter)
        self.short_mem = []  # Keep the most recent short_mem_size dialogues
        self.long_mem = []   # Store memory vectors and texts
        self.encoder = SentenceTransformer('/usr/gao/gubincheng/LLM/all-MiniLM-L6-v2')
        self.client = ZhipuAI(api_key=key)
        self.system_prompt = system_prompt or "You are a helpful assistant"

    def build_prompt(self, conversation_history):
        # Construct recent dialogue context
        recent_dialogue = "\n".join(
            [f"{msg['speaker']}: {msg['content']}" 
             for msg in conversation_history[-self.short_mem_size:]]
        )
        
        # Long-term memory retrieval
        current_topic = conversation_history[-1]["content"]
        if self.long_mem:
            current_vec = self.encoder.encode(current_topic)
            scores = [np.dot(current_vec, mem["vector"]) for mem in self.long_mem]
            best_match = self.long_mem[np.argmax(scores)]["text"]
            memory_section = f"\n[Relevant Memory]\n{best_match}"
        else:
            memory_section = ""
        """
        Return Example:
        You are analyzing news from your state's perspective. Your task is to decide on a reaction (like, comment, or do nothing) based on the given news.
            [News Content]
            {news_content}

            [Dialogue History]
            Moderator: {news_content}
            Florida: I like this news because it highlights the importance of social security, a key issue in Florida.
            Pennsylvania: I don't agree with Florida's perspective; the news doesn't cover economic policies in industrial states well enough.
            Michigan: I think the news is relevant to Michigan, especially regarding union workers and healthcare concerns.

            [Relevant Memory]
            Q: How does Michigan feel about economic growth and healthcare news?
            A: Michigan's voters are deeply concerned about healthcare and jobs, and news on economic growth and healthcare policies is often met with mixed reactions.

            Please respond to Pennsylvania and add your perspective...
        """
            
        return (
            f"{self.system_prompt}\n"
            f"[Dialogue History]\n{recent_dialogue} "
            f"{memory_section}\n"
            f"Please respond to {conversation_history[-1]['speaker']} and add your perspective on the news content..."
        )

    def respond(self, conversation_history, news_content):
        prompt = self.build_prompt(conversation_history)
        
        
        response = self.client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=150
        ).choices[0].message.content

        # Simulate the agent's behavior: like, comment, or do nothing
        action = random.choice(['like', 'comment', 'donothing'])
        if action == 'like':
            response = f"I like this news: {news_content}"
        elif action == 'comment':
            response = f"My comment on this news: {news_content} - I think it's very interesting."
        else:
            response = "I don't have any comment on this news."
        
        # Update memory
        self.update_memory(news_content, action, response)
        return action, response

    def update_memory(self, news_content, action, response):
        # Maintain short-term memory
        self.short_mem.append((news_content, action, response))
        if len(self.short_mem) > self.short_mem_size:
            self.short_mem.pop(0)
            
        # Store long-term memory
        memory_text = f"News: {news_content}\nAction: {action}\nResponse: {response}"
        self.long_mem.append({
            "text": memory_text,
            "vector": self.encoder.encode(memory_text)
        })

def multi_agent_chat(agents, rounds=5, initial_news="Here's some important news!"):
    history = [{"speaker": "Moderator", "content": initial_news}]
    
    for i in range(rounds):
        current_agent = agents[i % len(agents)]
        print(f"\nCurrent Speaker: {current_agent.name}")
        
        # Get the news content from the previous agent's response
        news_content = history[-1]["content"]
        
        # Get the action and response of the current agent
        action, response = current_agent.respond(history, news_content)
        
        # Update the history with the current agent's action and response
        history.append({
            "speaker": current_agent.name,
            "content": f"Action: {action}, Response: {response}"
        })
        
        print(f"{current_agent.name}: Action: {action}, Response: {response}")
        print("-"*50)

if __name__ == "__main__":
    # Initialize agents with some context
    florida = GLMAgent("Florida", 
        "You represent Florida, a state with a significant elderly population. Your task is to interact with the news based on your perspective.",
        short_mem_size=2) 
    
    pennsylvania = GLMAgent("Pennsylvania", 
        "You represent Pennsylvania, an industrial state with crucial suburban voters. Your task is to interact with the news from your state's perspective.",
        short_mem_size=3)  
    
    michigan = GLMAgent("Michigan", 
        "You represent Michigan, a state with strong union influence. You need to react to the news considering Michigan's point of view.",
        short_mem_size=4)  
    
    # Conduct 5 rounds of dialogue where agents interact with the news
    multi_agent_chat(
        agents=[florida, pennsylvania, michigan],
        rounds=5,
        initial_news="The U.S. is witnessing unprecedented economic growth. What are your thoughts on this?"
    )
