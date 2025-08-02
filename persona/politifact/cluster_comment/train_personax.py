import os
import json
import random
import pandas as pd
from typing import List
from openai import OpenAI
from tqdm import tqdm

# —— 配置区 —— #
INPUT_TSV    = '/usr/gao/gubincheng/article_rep/Agent/simplest_agent/cluster/cluster_comment/clu7/input_sampled.tsv'
OUTPUT_TSV   = 'cluster_personas6.tsv'
# 从环境变量读取 DeepSeek API Key
API_KEY      = "sk-98897711786641deb082803110753e1a"

MODEL_NAME   = 'deepseek-chat'
CHUNK_SIZE   = 12   # 每次迭代使用的评论数量
temperature   = 0.7

# 初始化 DeepSeek 客户端
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# Prompt 模板，包括维度指引
INIT_PROMPT = '''
### Task
Given the following {n} comments from one cluster, summarize what kind of person would write these comments.
Describe:
- Ideology (e.g., liberalism, conservatism, socialism, etc.)
- Big Five Traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
- Speaking Style (rational vs. emotional; aggressive vs. constructive; concise vs. detailed; humorous vs. serious; use of modifiers, etc.)
- Language Style: formality, rationality, aggressiveness, rhetoric_features
- Logical Rigour: presence of coherent reasoning, structured arguments, clarity of logic

Comments:
{comments}

### Output Format
Persona: '''

UPDATE_PROMPT = '''
### Task Description
You are an expert in personality analysis and discourse profiling. Your goal is to **update and refine the existing persona** based on additional comments from the same individual (cluster). The persona is characterized along these explicit dimensions:  
- Ideology (e.g., liberal, conservative, neutral, etc.)  
- Big Five Personality Traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)  
- Speaking/Writing Style (e.g., rational or emotional; constructive or confrontational)  
- Language Style (formality, rationality, aggressiveness, use of rhetoric, logical coherence, etc.)  
- Logical Rigour (clarity and structure of reasoning, presence of argument chains, etc.)

### Instructions
1. Review the following n new comments from the same cluster, analyzing each for:  
    - New behavioral, stylistic, or value-related signals that supplement, clarify, or shift the current persona.  
    - Any contrasts or emerging nuances compared to the original persona description.

2. For each dimension, clearly identify:  
    - Specific traits or shifts observed (with direct reference to comments where relevant).
    - Whether these signals reinforce, contradict, or nuance the existing persona.

3. Explain why these changes are significant:  
    - What do these shifts reveal about the individual's personality, communication style, or value priorities?
    - How do these observations refine, expand, or correct your prior understanding of the persona?

4. Pay special attention to: 
    - Changes in formality, rationality, aggressiveness, rhetorical techniques, and logical structuring.
    - Any recurring patterns, unique expressions, or contradictions that provide deeper insight into the persona.

Comments to analyze:
{comments}

Current Persona:
{persona}

### Output Format
Write your answer as follows:
Updated Persona:
[A detailed, comprehensive persona description, integrating and clearly explaining any updates or nuances across the specified dimensions. Reference specific observations and reasoning for each dimension where appropriate.]
'''


FINAL_SUMMARY_PROMPT = '''
### Task Description
Given the fully refined persona below, produce a **vivid, comprehensive, and highly individualized final persona profile**. Your summary should not only synthesize the main characteristics, but also highlight any unique or illustrative details that bring the persona to life.

**Your response must include the following sections, each with concrete examples or explicit observations wherever possible:**

1. Core Interests & Thematic Focus  
   - What topics, issues, or domains is this individual most concerned with?  
   - Highlight any recurring passions or specialized knowledge areas.

2. Ideological Leanings & Value Orientation 
   - Clearly state their place on relevant ideological spectra (e.g., liberal-conservative, individualist-collectivist, progressive-traditional, etc.), if discernible.  
   - Note specific stances, value signals, or nuanced positions evident in their discourse.

3. Big Five Personality Traits  
   - Describe each trait as it manifests in their comments (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism).  
   - Give short, concrete examples or typical behaviors for dominant traits.

4. Speaking & Language Style  
   - Characterize their communication (e.g., formal/informal, logical/emotional, direct/indirect, use of rhetoric or figurative language).  
   - Cite notable phrasing, habits, or stylistic quirks.

5. Logical Reasoning & Argumentative Approach 
   - Summarize their approach to building arguments or explanations (e.g., evidence-based, story-driven, analogical, Socratic questioning, etc.).  
   - Point out any patterns in logical rigor, use of examples, or argumentative strategies.

6. Notable Nuances & Illustrative Details  
   - Include any subtle, unique, or distinguishing features—such as contradictions, recurring expressions, pet peeves, humor, or preferred analogies—that add depth and individuality to the persona.

Refined Persona:
{persona}

### Output Format
Final Persona:
[Write a single, fluent, and detailed paragraph that synthesizes all the information above. Capture the individual's core interests, ideological leanings, dominant Big Five personality traits (without giving specific examples), speaking and language style, typical ways of expressing their positions, logical reasoning habits, and any distinctive nuances or illustrative details. Emphasize both how they communicate and how they tend to state or frame their views. The profile should read as a vivid, and memorable individual—not a generic summary.]
'''

# 加载数据
def load_comments(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep='\t', dtype=str)
    if 'text' not in df.columns or 'cluster' not in df.columns:
        raise KeyError("输入文件必须包含 'text' 和 'cluster' 两列")
    return df

# 按簇顺序分块
def chunk_comments(comments: List[str], size: int) -> List[List[str]]:
    return [comments[i:i+size] for i in range(0, len(comments), size)]

# 通用函数：调用 DeepSeek
def call_model(prompt: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {'role':'system', 'content':'You are a helpful assistant.'},
            {'role':'user',   'content':prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# 生成初始 persona
def generate_initial_persona(comments: List[str]) -> str:
    prompt = INIT_PROMPT.format(
        n=len(comments),
        comments='\n'.join(f'comment {i+1}: "{c}"' for i, c in enumerate(comments))
    )
    return call_model(prompt, max_tokens=256)

# 更新 persona
def update_persona(current: str, comments: List[str]) -> str:
    prompt = UPDATE_PROMPT.format(
        persona=current,
        n=len(comments),
        comments='\n'.join(f'comment {i+1}: "{c}"' for i, c in enumerate(comments))
    )
    return call_model(prompt, max_tokens=256)

# 最终总结 persona
def finalize_persona(refined: str) -> str:
    prompt = FINAL_SUMMARY_PROMPT.format(persona=refined)
    return call_model(prompt, max_tokens=1024)

# 主流程
def main():
    df = load_comments(INPUT_TSV)
    clusters = sorted(df['cluster'].unique(), key=lambda x: int(x))
    results = []

    # 对簇使用进度条
    for cid in tqdm(clusters, desc='Clusters'):
        texts = df.loc[df['cluster']==cid, 'text'].tolist()
        chunks = chunk_comments(texts, CHUNK_SIZE)
        # 初始 persona
        persona = generate_initial_persona(chunks[0])
        # 迭代更新
        for chunk in tqdm(chunks[1:], desc=f'Cluster {cid}', leave=False):
            persona = update_persona(persona, chunk)
        # 最终总结
        final_persona = finalize_persona(persona)
        results.append({'cluster': cid, 'persona': final_persona})

    # 保存结果
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_TSV, sep='\t', index=False)
    print(f"All personas saved to {OUTPUT_TSV}")

if __name__ == '__main__':
    main()
