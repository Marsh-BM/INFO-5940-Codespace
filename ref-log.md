# Reflection Log – Multi-Agent Travel Planner

### What I learned
Through this project, I developed a deep understanding of how a multi-agent workflow can divide reasoning tasks and improve reliability through role specialization.  
The **Planner Agent** was designed to be generative and creative—it expands a vague travel request into a structured, day-by-day itinerary that includes times, locations, and budget estimates.  
Meanwhile, the **Reviewer Agent** performs the complementary role of evaluator and validator.  
It uses the `internet_search` tool to confirm factual details, such as opening hours, ticket prices, and feasibility of inter-city travel.  
This division between planning and verification showed me how different system prompts can simulate human collaboration: one produces ideas, and the other enforces realism and factual grounding.  
It also reinforced the importance of **prompt precision**, since small wording changes can drastically alter how the model interprets its role.

### Challenges faced and how I addressed them
The first challenge I encountered was tool integration. Initially, Reviewer Agent returned an ‘invalid or missing API key’ error when attempting to call `internet_search`. To resolve this, I created a valid Tavily key in the `.env` file, formatted as follows:

TAVILY_API_KEY=tvly_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

After reloading Streamlit with these configurations, the Reviewer Agent was able to perform real-time validation.  
Another major challenge was balancing verbosity and clarity in the system prompts.  
If the Planner prompt was too short, the itinerary lacked details; if too long, it generated excessive narrative text instead of structured bullet points.  
Through iterative testing with multiple prompts—such as *“Plan a 3-day Paris trip including the Louvre on Tuesday”* and *“5-day Japan itinerary under $1000 focusing on anime and sushi”*—I refined both agents’ behaviors until their outputs became coherent, factual, and stylistically consistent.

### Creative ideas and design decisions
I intentionally formatted the Reviewer’s output into two sections:  
1. **Delta List** – a concise summary of detected factual errors and suggested corrections (e.g., “The Louvre is closed on Tuesdays → moved to Day 2”).  
2. **Final Validated Itinerary** – the fully revised, clean plan.  
This structure makes the interaction between the two agents transparent and easy to grade.  
Additionally, I enabled **real-time tool logging** in Streamlit’s sidebar so that every `internet_search` query and its result preview are visible to the user.  
This provides explainability, an essential quality for multi-agent systems.

### External tools and assistance
I used the Tavily API for web search, the Cornell OpenAI Gateway for model access, and Streamlit for interface design.  
Minor prompt-tuning ideas were inspired by GPT-assisted suggestions, but all final implementation and debugging were completed independently.
