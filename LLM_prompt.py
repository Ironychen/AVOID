import numpy as np
import json
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from zhipuai import ZhipuAI
import json
import base64
import re
import time
from openai import OpenAI

image_path = "/usr/gao/gubincheng/article_rep/Agent/data/politic_imgs/"
data_path = "/usr/gao/gubincheng/article_rep/Agent/data/pro/news_politic_286.json"

key = "653d34becfd62e46c1a02e7642e43c02.eqLfDKBj4GJdnOgH" 
client = ZhipuAI(api_key=key)

def build_content_prompt_politic(news_id=None, content=None, max_content_length=300):
    """
    Build a prompt for obtaining news summaries and opinions
    
    Parameters:
        news_id: News ID, if provided, will fetch news content and comments from data file
        content: Directly provided news content, used if news_id is None
        max_content_length: Maximum number of words to extract from news content
    
    Returns:
        A prompt string to be submitted to the LLM
    """
    # If news_id is provided, get news content and comments from data file
    if news_id:
        news_data = load_news_by_id(news_id)
        if news_data:
            news_content = news_data.get('content', '')
            # Limit the content length
            content_words = news_content.split()
            if len(content_words) > max_content_length:
                content = ' '.join(content_words[:max_content_length]) + '...'
            else:
                content = news_content
            
            # Get comments
            comments = news_data.get('comment', '')
            if comments and isinstance(comments, list):
                comments_text = '\n'.join([f"{c.get('name', 'Anonymous')}: {c.get('comment', '')}" for c in comments])
            elif comments and isinstance(comments, str):
                comments_text = comments
            else:
                comments_text = None
        else:
            content = content or "Could not find news content for the specified ID"
            comments_text = None
    else:
        # Use directly provided content
        comments_text = None

    prompt = f"""
    You will be presented with a news article and potentially related comments.
    
    News Article:
    {content}
    """
    
    if comments_text and comments_text.strip():
        prompt += f"""
        Related Comments:
        {comments_text}
        """
    
    prompt += """
    Please analyze this information following a chain of thought reasoning process:
    
    Step 1: First, identify the key facts presented in the news article.
    - What is the main event or topic?
    - Who are the key individuals or organizations involved?
    - When and where did this take place?
    - What are the claimed causes and effects?
    
    Step 2: Analyze any comments provided (if present).
    - What perspectives do the comments offer?
    - Do they support, contradict, or add nuance to the news article?
    - Are there any recurring themes or concerns in the comments?
    
    Step 3: Synthesize a concise summary based on your analysis.
    - Focus on the most important and verified information
    - Present the summary in a neutral, factual manner
    
    Step 4: Form your own opinion on the matter.
    - Consider the credibility of the sources
    - Evaluate the logical consistency of the claims
    - Balance different perspectives
    - Draw your own conclusions based on the information available
    
    After completing these steps, provide your final response in the following JSON format:
    {
        "summarize": "Your concise summary of the key points in the news article",
        "opinion": "Your thoughtful perspective on the article, considering both the content and any comments less than 80 words"
    }
    
    Ensure your reasoning is clear throughout your thought process, and make sure your final JSON output accurately reflects your analysis.
    """
    
    return prompt


def build_keyword_extraction_prompt(news_text):
    
    prompt = (
        "You are an expert in named entity and event extraction. "
        "From the following English news article, extract:\n"
        "1. Important People: Only real human names (e.g., politicians, CEOs, celebrities). "
        "Do NOT include organizations, places, or generic terms.\n"
        "2. Event Keywords: Only the most critical event-related keywords (up to 3). "
        "No full sentences or explanations.\n\n"
        "Output format:\n"
        "People: name1, name2, name3\n"
        "Events: keyword1, keyword2, keyword3\n\n"
        f"News:\n{news_text}"
    )
    return prompt


def build_content_prompt_gossip(news_id=None, content=None, max_content_length=300):
    """
    Build a prompt for obtaining news summaries and opinions
    
    Parameters:
        news_id: News ID, if provided, will fetch news content and comments from data file
        content: Directly provided news content, used if news_id is None
        max_content_length: Maximum number of words to extract from news content
    
    Returns:
        A prompt string to be submitted to the LLM
    """
    # If news_id is provided, get news content and comments from data file
    if news_id:
        news_data = load_news_by_id(news_id)
        if news_data:
            news_content = news_data.get('content', '')
            # Limit the content length
            content_words = news_content.split()
            if len(content_words) > max_content_length:
                content = ' '.join(content_words[:max_content_length]) + '...'
            else:
                content = news_content
            
            # Get comments
            comments = news_data.get('comment', '')
            if comments and isinstance(comments, list):
                comments_text = '\n'.join([f"{c.get('name', 'Anonymous')}: {c.get('comment', '')}" for c in comments])
            elif comments and isinstance(comments, str):
                comments_text = comments
            else:
                comments_text = None
        else:
            content = content or "Could not find news content for the specified ID"
            comments_text = None
    else:
        # Use directly provided content
        comments_text = None

    prompt = f"""
    You will be presented with a news article and potentially related comments.
    
    News Article:
    {content}
    """
    
    if comments_text and comments_text.strip():
        prompt += f"""
        Related Comments:
        {comments_text}
        """
    
    prompt += """
    Please analyze the following entertainment news using a chain-of-thought reasoning process:

    Step 1: Identify the core facts in the news article.
    - What is the main topic or event?
    - Which celebrities or entertainment entities are involved?
    - When and where did it allegedly take place?
    - What claims or rumors are being made, and what are the supposed effects or reactions?

    Step 2: Analyze the provided comment (if any).
    - Does the comment support, question, or reinterpret the news?
    - Is it emotionally charged, humorous, sarcastic, or neutral?
    - Does it reflect a common audience reaction or raise important skepticism?

    Step 3: Synthesize a concise summary.
    - Focus on the main point of the news and any claims that seem clearly stated or widely reported.
    - Avoid emotional or speculative phrasing—keep the tone neutral and fact-based.

    Step 4: Offer your own judgment.
    - Consider how credible or speculative the news seems.
    - Weigh the impact of public perception and how the comment influenced your view.
    - Use logical reasoning to form a concise opinion.

    After completing these steps, provide your final response in **JSON format**:

    {
        "summarize": "Your neutral summary of the key points in the entertainment news.",
        "opinion": "Your reasoned opinion (under 80 words), including your judgment on its credibility and how the comment affected your perception."
    }

    Make sure your chain of thought is clear, and ensure your final JSON accurately reflects your analysis.
    """
    return prompt
    


def build_image_prompt(news_id):
    """
    Build a prompt for obtaining image descriptions using a simplified chain of thought reasoning
    
    Parameters:
        news_id: News ID, if provided, will fetch the associated image
        image_path: Directly provided image path, used if news_id is None
    
    Returns:
        A prompt string to be submitted to the LLM
    """
    # If news_id is provided, get the associated image path
    if news_id:
        image_path = "/usr/gao/gubincheng/article_rep/Agent/data/gossip/pics_gossip/" + news_id + ".jpg"
        if not image_path:
            return "Could not find an image associated with the specified news ID."
    
    prompt = """
    You will be presented with an image associated with a news article. Please examine this image carefully.
    
    Image: [Image attached]
    
    Please analyze this image following a chain of thought reasoning process:
    
    Step 1: Describe the visual content objectively.
    - Identify the main subjects, people, objects, and setting in the image
    - Note any text, captions, or graphical elements visible in the image
    - Describe the actions or events depicted in the image
    
    Step 2: Analyze the context and composition.
    - Consider how the image is framed and what perspective it presents
    - Identify any emotional tone conveyed by the image
    - Determine how this image might relate to current events or news topics
    
    Step 3: Formulate your interpretation.
    - Evaluate what message or story this image appears to convey
    - Consider any potential biases or perspectives embedded in the image
    - Reflect on how different viewers might interpret this image
    
    After completing these steps, provide your final response in the following JSON format:
    {
        "description": "Your objective description of what is shown in the image",
        "interpretation": "Your analysis of the image's meaning and significance"
    }
    
    Keep your response concise but informative.
    """
    
    return prompt, image_path

def load_news_by_id(news_id):
    """
    Load news with a specific ID from the news data file
    
    Parameters:
        news_id: News ID
    
    Returns:
        Dictionary containing news content and comments, or None if not found
    """
    try:
        data_path = "/usr/gao/gubincheng/article_rep/Agent/data/gossip/gossip_news.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
            
        for news_item in news_data:
            if news_item.get('id') == news_id:
                return news_item
                
        return None
    except Exception as e:
        print(f"Error loading news data: {e}")
        return None

# def get_content_response(prompt):
#     client = OpenAI(api_key="sk-fd13d31db47b40259fdf269295c23f92", base_url="https://api.deepseek.com")
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[{"role": "system", "content": "You are a helpful assistant."},
#                   {"role": "user", "content": prompt}],
#         stream=False
#     )
#     return response.choices[0].message.content
def get_content_response(prompt):
    client = ZhipuAI(api_key=key)  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4-flash",  # 填写需要调用的模型名称
        messages=[{"role": "system", "content": "You are an active user on social media."},
                  {"role": "user", "content": prompt}],
        stream=False
    )
    return response.choices[0].message.content

def get_image_response(prompt, img_path):
    with open(img_path, 'rb') as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')

    client = ZhipuAI(api_key=key)  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4v-flash",  # 填写需要调用的模型名称
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_base
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content

def build_fake_news_strategy_prompt(comment_text, predicted_label, true_label, reasoning):

    prompt = f"""
    Your previous decision resulted in a negative outcome (punishment) because your prediction was '{predicted_label}', 
    but the true label is '{true_label}'. Your comment was: "{comment_text}"
    and your reasoning process was: {reasoning}

    This negative outcome indicates that your current strategy is flawed and will be penalized if repeated.
    To learn from this mistake and avoid similar errors in the future, please analyze your error from the following three perspectives:

    1. Entity Relationship Analysis:**  
    - Extract the key entities and events mentioned in the news.
    - Use your internal knowledge (do not perform online searches) to check for inconsistencies or missing links between these entities.
    - Determine whether such discrepancies misled your judgment.

    2. News Semantic Analysis:
    - Evaluate whether the news contains extreme, polarized, or sensationalized language.
    - Break down the keywords to identify if emotional or biased wording contributed to an incorrect interpretation.

    3. Advanced Semantic Information:  
    - Reflect on the proper attitude you should maintain when processing such events.
    - Analyze if your stance deviated from an optimal, balanced approach.
    - Consider if this deviation is a strong indicator of error.

    From these three aspects, select the one you believe most likely caused your error (i.e., led to the punishment). 
    Then, based on that analysis, propose a specific new corrective strategy—a policy update that you will adopt to avoid making the same mistake in the future.

    Finally, summarize your analysis and the corrective strategy in a single, coherent paragraph. Your summary should clearly outline the key insight and the updated approach you will follow next time.

    Return your answer as a single, coherent piece of text.
"""
    return prompt

def build_fake_news_strategy_img_prompt(comment_text, predicted_label, true_label, reasoning):

    prompt = f"""
    Your previous decision resulted in a negative outcome (punishment) because your prediction was '{predicted_label}', 
    but the true label is '{true_label}'. Your comment was: "{comment_text}"
    and your reasoning process was: {reasoning}

    This incorrect prediction indicates that you failed to correctly judge the credibility of the entertainment news in this case. 
    To improve, you must now analyze your mistake in a concrete, event-specific way.

    Please proceed as follows:

    1. Re-identify the Misjudged News Item:  
    - Re-express what the news was about: What happened, who was involved, and what claims were made?  
    - Clearly name the key entities or people involved (e.g., "Brad Pitt", "divorce", "secret dinner", etc).  
    - Specify what you initially thought and what the ground truth actually was.

    2. Locate the Error Source (Choose One or Two of the Following):  
    - Entity Relationship Flaw: Did you wrongly assume a relationship (e.g., romantic or conflict) based on co-occurrence of names or misinterpret a known public fact?  
    - Textual Bias: Was your judgment swayed by sensational words like “shocking”, “exclusive”, “confirmed”, etc., even though no credible source was cited?  
    - Image Deception: Did the image (e.g., a photo of two people talking) mislead you into thinking the event happened recently or meant something more than it did?  
    - Contextual Mismatch: Was the image old, unrelated, or taken from a different event? Was it misleadingly paired with the article?

    3. Generate a Corrective Strategy:  
    - Based on the error you made, describe **what rule you will now follow** in future similar news (e.g., “I will no longer trust headlines that use ‘confirmed’ without cited sources”, or “I will be more cautious of celebrity photos taken without a timestamp”).

    4. Final Summary:  
    - Summarize your analysis and your new policy in **one concise paragraph**. The summary must reference the **specific event**, **the cause of your error**, and the **change in strategy** you will implement.

    Return your answer as a single, coherent paragraph that demonstrates learning from this specific mistake.
    """
    return prompt

# comment_text = "neutral: The report presents an incident involving Maxine Waters and Laura Loomer, but without additional context or evidence, it is difficult to determine the authenticity of the news. Both parties have their own accounts, and without an independent investigation or official statement, it is uncertain how to weigh the credibility of either side."
# predicted_label = "neutral"
# true_label = "fake"
# reasoning = "Step 1: Analyzed the news for factual details and the presence of a clear narrative; Step 2: Noted the absence of additional context or supporting evidence; Step 3: Recognized the presence of conflicting statements from both parties without a resolution; Step 4: Concluded that without further information, it is neutral regarding the authenticity of the news."
# prompt1 = build_fake_news_strategy_prompt(comment_text, predicted_label, true_label, reasoning)
# print("prompt: ",prompt1)
# response = get_content_response(prompt1)
# print("response: ",response)

# prompt2 = f"summarize the response under 100 words {response}"
# res2 = get_content_response(prompt2)
# print("res2: ",res2)



# prompt = build_content_prompt(news_id="politifact14727")
# # print(prompt)
# response = get_content_response(prompt)
# print("content response:", response)

# # **使用正则表达式提取 JSON**
# json_pattern = re.search(r'\{[\s\S]*?\}', response)  # `[\s\S]*?` 匹配多行 JSON
# json_str = json_pattern.group(0)  # 获取匹配的 JSON 字符串
# data = json.loads(json_str)
# content_summarize = data.get("summarize")
# content_opinion = data.get("opinion")


# print("content summarize:", content_summarize)
# print("content opinion:", content_opinion)


# prompt1, img_path = build_image_prompt("politifact73")
# # print(prompt1)
# img_response = describe_image(prompt1, img_path)
# print("image response:", img_response)

# json_pattern_img = re.search(r'\{[\s\S]*?\}', response)  # `[\s\S]*?` 匹配多行 JSON
# json_img = json_pattern_img.group(0)  # 获取匹配的 JSON 字符串
# data_img = json.loads(json_img)
# image_description = data_img.get("description")
# image_inter = data_img.get("interpretation")

# **使用正则表达式提取 JSON**
# json_pattern_img = re.search(r'\{[\s\S]*?\}', img_response)  # `[\s\S]*?` 匹配 JSON 结构
# json_img = json_pattern_img.group(0)  # 获取匹配的 JSON 字符串
# data = json.loads(json_img)

# description_value = data.get("description")
# interpretation_value = data.get("interpretation")


# print(f"- description: {description_value}")
# print(f"- interpretation: {interpretation_value}")


# print("image_description:", image_description)
# print("image_inter:", image_inter)

# print("image description:", img_response)
