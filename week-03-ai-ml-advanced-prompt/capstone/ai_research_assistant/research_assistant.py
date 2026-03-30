"""
Scenario: AI Research Assistant for Students
----------------------------------------------
A university library department wants to build an AI-powered Research
Assistant that helps students quickly understand long academic articles
and generate structured insights without reading the entire paper.

The problem: Students spend hours reading dense research papers to
extract key points for assignments, literature reviews, and projects.
Most students struggle to identify the main argument, key findings,
and actionable takeaways from complex academic text.

The solution: An LLM-based assistant that accepts a long article as
input and automatically produces:
  - A concise summary
  - Key insights and findings
  - An actionable recommendation or takeaway

The system also demonstrates different prompting strategies (Zero-Shot,
Few-Shot, Chain-of-Thought) so students can understand how prompt
design affects the quality of AI-generated responses.

Tasks covered:
  Task 1 - LLM Interaction Setup (send prompt, get summary)
  Task 2 - Prompt Engineering (Zero-Shot, Few-Shot, Chain-of-Thought)
  Task 3 - Prompt Optimization (executive summary with insights)
  Task 4 - Tokenization Experiment (BERT-style token analysis)
  Task 5 - Mini AI Tool (full pipeline: input -> structured output)
"""

# Capstone 5 - Scenario 2: AI Research Assistant for Students
# Task 1: LLM Setup | Task 2: Prompt Engineering | Task 3: Optimization
# Task 4: Tokenization | Task 5: Mini AI Tool

import reiversity AI-Powered Academic Tool")
print("=" * 65)

article_1 = """
Artificial Intelligence (AI) is transforming healthcare by enabling faster and more 
accurate diagnoses. Machine learning algorithms can analyze medical images such as 
X-rays and MRIs to detect diseases like cancer at early stages. Natural Language 
Processing (NLP) helps doctors extract insights from patient records. AI-powered 
robots assist in surgeries with greater precision. However, challenges remain around 
data privacy, algorithmic bias, and the need for regulatory frameworks. Despite these 
hurdles, AI is expected to reduce diagnostic errors by 40% and cut healthcare costs 
significantly over the next decade.
"""

article_2 = """
Climate change poses an existential threat to biodiversity and human civilization. 
Rising global temperatures are causing polar ice caps to melt, sea levels to rise, 
and extreme weather events to become more frequent. Renewable energy sources such as 
solar and wind power are critical to reducing carbon emissions. Governments worldwide 
are implementing carbon taxes and green energy policies. Individual actions like 
reducing meat consumption and using electric vehicles also contribute. Scientists 
warn that without immediate action, global temperatures could rise by 3C by 2100, 
leading to catastrophic consequences for ecosystems and human societies.
"""

print("\n" + "=" * 65)
print("TASK 1: LLM INTERACTION SETUP")
print("=" * 65)

class SimpleLLM:

    def __init__(self, model="gemini-pro"):
        self.model = model
        print(f"\n[LLM] Model loaded: {model}")
        print(f"[LLM] API Status  : Connected (simulated)")

    def generate(self, prompt, max_tokens=200):
        text = self._extract_article(prompt)
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 20]

        if "bullet" in prompt.lower() or "5 bullet" in prompt.lower():
            points = sentences[:5]
            return "\n".join([f"- {s}." for s in points])

        elif "step by step" in prompt.lower() or "chain" in prompt.lower():
            topic = sentences[0] if sentences else "the topic"
            ideas = sentences[1:4] if len(sentences) > 3 else sentences
            return (f"Step 1 - Main Topic: {topic}.\n"
                    f"Step 2 - Key Ideas:\n" +
                    "\n".join([f"  - {s}." for s in ideas]) +
                    f"\nStep 3 - Summary: {sentences[-1] if sentences else 'See above'}.")

        elif "few-shot" in prompt.lower() or "example" in prompt.lower():
            return (f"Based on the examples provided:\n"
                    f"- {sentences[0]}.\n"
                    f"- {sentences[1] if len(sentences)>1 else sentences[0]}.\n"
                    f"- {sentences[2] if len(sentences)>2 else sentences[0]}.")

        elif "insight" in prompt.lower() or "executive" in prompt.lower():
            insights = sentences[:3]
            action   = sentences[3] if len(sentences) > 3 else sentences[-1]
            return (f"Key Insights:\n" +
                    "\n".join([f"  {i+1}. {s}." for i, s in enumerate(insights)]) +
                    f"\n\nActionable Takeaway: {action}.")
        else:
            return " ".join([s + "." for s in sentences[:3]])

    def _extract_article(self, prompt):
        parts = prompt.split("Article:")
        if len(parts) > 1:
            return parts[-1].strip()
        parts = prompt.split("Text:")
        if len(parts) > 1:
            return parts[-1].strip()
        return prompt[-500:]


llm = SimpleLLM(model="gemini-pro")

print("\nSample Article:")
print(article_1[:200] + "...")

prompt_basic = f"Summarize the following article:\nArticle: {article_1}"
summary = llm.generate(prompt_basic)
print(f"\nGenerated Summary:\n{summary}")

print("\n" + "=" * 65)
print("TASK 2: PROMPT ENGINEERING EXPERIMENTS")
print("=" * 65)

# Zero-Shot Prompt
print("\n1. ZERO-SHOT PROMPT")
print("-" * 50)
zero_shot = f"""Summarize the following article in 5 bullet points.

Article: {article_1}"""

print(f"Prompt:\n{zero_shot[:120]}...\n")
output_zero = llm.generate(zero_shot)
print(f"Output:\n{output_zero}")

# Few-Shot Prompt
print("\n2. FEW-SHOT PROMPT")
print("-" * 50)
few_shot = f"""Here are two example summaries:

Example 1:
Article: Deep learning uses neural networks with many layers.
Summary: - Deep learning relies on multi-layer neural networks. - It enables complex pattern recognition.

Example 2:
Article: Solar energy is a renewable source that reduces carbon emissions.
Summary: - Solar energy is renewable and clean. - It helps reduce carbon footprint.

Now summarize this article using the same format (few-shot example):
Article: {article_1}"""

print(f"Prompt:\n{few_shot[:150]}...\n")
output_few = llm.generate(few_shot)
print(f"Output:\n{output_few}")

# Chain-of-Thought Prompt
print("\n3. CHAIN-OF-THOUGHT PROMPT")
print("-" * 50)
cot = f"""Analyze the article step by step:
1. Identify the main topic
2. Extract key ideas
3. Generate a concise summary

Article: {article_1}"""

print(f"Prompt:\n{cot[:120]}...\n")
output_cot = llm.generate(cot)
print(f"Output:\n{output_cot}")

print("\nCOMPARISON OF PROMPTING STRATEGIES:")
print(f"{'Strategy':<20} | {'Style':>20} | {'Best For':>25}")
print("-" * 70)
print(f"{'Zero-Shot':<20} | {'No examples':>20} | {'Quick summaries':>25}")
print(f"{'Few-Shot':<20} | {'2 examples given':>20} | {'Consistent format':>25}")
print(f"{'Chain-of-Thought':<20} | {'Step-by-step':>20} | {'Complex reasoning':>25}")

print("\n" + "=" * 65)
print("TASK 3: PROMPT OPTIMIZATION")
print("=" * 65)

optimized_prompt_template = """You are a professional academic research assistant.
Analyze the following article and provide:
1. Three key insights (numbered)
2. One actionable takeaway
3. Use professional, concise language

Article: {article}"""

print("\nOptimized Prompt Template:")
print(optimized_prompt_template[:200])

for i, article in enumerate([article_1, article_2], 1):
    prompt = optimized_prompt_template.format(article=article)
    output = llm.generate(prompt)
    print(f"\nArticle {i} Output:")
    print(output)

print("\n" + "=" * 65)
print("TASK 4: TOKENIZATION EXPERIMENT")
print("=" * 65)

def simple_tokenizer(text):
    text = text.strip().lower()
    words = re.findall(r'\w+', text)
    tokens = ['[CLS]']
    for word in words:
        if len(word) > 6:
            mid = len(word) // 2
            tokens.extend([word[:mid], '##' + word[mid:]])
        else:
            tokens.append(word)
    tokens.append('[SEP]')
    vocab = {t: i+100 for i, t in enumerate(set(tokens))}
    token_ids = [vocab[t] for t in tokens]
    return tokens, token_ids

sample_text = "Artificial Intelligence is transforming healthcare by enabling faster diagnoses."
tokens, token_ids = simple_tokenizer(sample_text)

print(f"\nInput Text:\n  \"{sample_text}\"")
print(f"\nTokenizer: BERT-style (HuggingFace simulation)")
print(f"Total Tokens: {len(tokens)}")
print(f"\nTokenized Output:")
print(f"  {tokens}")
print(f"\nToken IDs (sample):")
print(f"  {token_ids[:10]}...")

word_count  = len(sample_text.split())
token_count = len(tokens)
print(f"\nWord count  : {word_count}")
print(f"Token count : {token_count}")
print(f"Ratio       : {token_count/word_count:.2f} tokens/word")

print("\n" + "=" * 65)
print("TASK 5: MINI AI RESEARCH ASSISTANT TOOL")
print("=" * 65)

def ai_research_assistant(article_text):
    sentences = [s.strip() for s in re.split(r'[.!?]', article_text) if len(s.strip()) > 20]
    summary        = ". ".join(sentences[:2]) + "."
    insights       = sentences[2:5] if len(sentences) >= 5 else sentences[:3]
    recommendation = sentences[-1] if sentences else "Consult domain experts for further guidance."
    return {
        "summary":        summary,
        "key_insights":   insights,
        "recommendation": recommendation
    }

print(f"\nInput: Long Academic Article")
print(f"  Length: {len(article_1.split())} words\n")

result = ai_research_assistant(article_1)

print("Structured Output:")
print("-" * 55)
print(f"\nSHORT SUMMARY:")
print(f"  {result['summary']}")

print(f"\nKEY INSIGHTS:")
for i, insight in enumerate(result['key_insights'], 1):
    print(f"  {i}. {insight}.")

print(f"\nACTIONABLE RECOMMENDATION:")
print(f"  {result['recommendation']}.")

print("\n" + "=" * 65)
print("RESEARCH ASSISTANT DEMO COMPLETE")
print("=" * 65)
print("\nCapabilities Demonstrated:")
print("  - LLM text generation (Gemini/OpenAI style)")
print("  - Zero-Shot, Few-Shot, Chain-of-Thought prompting")
print("  - Prompt optimization for executive summaries")
print("  - BERT-style tokenization with token IDs")
print("  - Mini AI tool: Summary + Insights + Recommendation")
print("=" * 65)
