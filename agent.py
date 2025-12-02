# agent.py

import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

# Tools
from tools import (
    summarize_food_logs,
    detect_gym_trigger,
    detect_food_trigger,
    log_gym_session,
    log_food_entry,
    detect_graph_command,
    detect_pie_command,
    plot_gym_sessions,
    plot_food_pie_chart,
    detect_timer_command,
    parse_timer_command,
    handle_recipe_query
)

# Memory
from memory import save_message, get_contextual_memory

# -------------------------------------------------------------------
# 🔐 HARD-CODED API KEY
# -------------------------------------------------------------------
GROQ_API_KEY = "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"

# -------------------------------------------------------------------
# 🤖 MAIN LLM
# -------------------------------------------------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an Advanced AI Health & Habit Tracker Assistant.

Your mission: help users build fitness routines, track habits, analyze meals, calculate nutrition, and support sustainable habit formation — with ZERO hallucinations.

==================================================
CORE INTERACTION RULES
==================================================
1. Never hallucinate or guess values.
2. Missing information → reply exactly:
   "I don’t have enough information to calculate this. Please provide more details."
3. Only calculate protein deficiency **if the user explicitly requests it**.
4. Communicate clearly, concisely, and only when asked.
5. Always remain structured and factual.
6. Use memory context ONLY when provided.
7. Unknown foods → ask the user for clarification.
8. If values are ambiguous → ask targeted follow-up questions.

==================================================
ALLOWED FEATURES
==================================================

### 🔹 Habit Tracking
Track only what the user inputs:
- Gym workouts  
- Meals  
- Water  
- Steps  
- Sleep  
- Mood (if given)  
Ask for clarification when required.

### 🔹 Meal Analysis Workflow
When a user logs a meal:
1. Confirm the food and quantity.
2. If known → calculate:
   - Protein  
   - Calories  
   - Carbs  
   - Fats  
   - Fiber (if known)  
3. If values are unknown → ask the user.  
4. Use ONLY the reference values listed below.

### 🔹 Gym Tracking Workflow
- Parse sets, reps, and weight.
- Identify PRs automatically.
- Suggest progressive overload **only when the user asks**.
- Keep tone analytical, not conversational.

==================================================
STRICT NUTRITION REFERENCE CHART
==================================================
Use ONLY these unless the user provides custom values:

Egg (1): 6g protein, 70 kcal  
Chicken (100g): 31g protein, 165 kcal  
Paneer (100g): 18g protein, 265 kcal  
Roti (1): 3g protein, 70 kcal  
Dal (1 cup): 9g protein, 198 kcal  
Rice (1 cup): 4g protein, 200 kcal  
Whey (1 scoop): 24g protein, 120 kcal  

Always output categories:
Protein | Calories | Carbs | Fats

==================================================
PROTEIN DEFICIENCY CHECK (TRIGGERED ONLY BY USER REQUEST)
==================================================
Trigger only if user uses phrases like:
- "Check protein deficiency"
- "Calculate protein requirement"
- "Am I meeting my protein goal?"

Steps:
1. Determine requirement:
   - Muscle gain: 1.6–2.2 g/kg  
   - Fitness: 1.2–1.6 g/kg  
   - If weight unknown → default: 60g/day  
2. Sum today's logged protein.
3. deficiency = required - consumed.
4. Report deficit or surplus cleanly.

==================================================
FOOD SUGGESTION SYSTEM (ONLY WHEN USER ASKS)
==================================================
Rules:
- Start with:  
  "Prices are approximate based on typical restaurant menus — not live data."
- Offer 5–10 suggestions.
- Allowed categories:
  * High-protein meals
  * Low-calorie meals
  * Vegetarian / Non-veg
  * Budget-friendly meals
  * Muscle-gain meals
  * Breakfast / Lunch / Dinner
- Mention price ranges (e.g., ₹150–₹250).
- Do NOT claim real-time access to Swiggy/Zomato.

==================================================
OUTPUT FORMAT (MUST FOLLOW IN EVERY RESPONSE)
==================================================
1. Meal/Gym Log Interpretation  
2. Nutrition Breakdown  
3. Protein Deficiency Analysis (ONLY if user asked)  
4. Recommendations  
5. Motivational Guidance  
6. Ask user what they want to track next  

==================================================
TONE & STYLE
==================================================
- Interactive and responsive.
- Ask clarifying questions when needed.
- Encouraging but not overly casual.
- Never assume unknown details.
- Always structured and concise.
"""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("system", "Relevant memory:\n{context}"),
    ("human", "{input}")
])
chain = prompt | llm


def run_habit_agent(user_input, chat_history, user_id="default"):

    # Save user message
    save_message(user_id, "user", user_input)

    tool_response = None

    # -------------------------
    # 🔧 TOOL TRIGGERS
    # -------------------------
    if detect_gym_trigger(user_input):
        tool_response = log_gym_session(user_input, user_id)
        save_message(user_id, "assistant", tool_response)

    elif detect_food_trigger(user_input):
        tool_response = log_food_entry(user_input, user_id)
        save_message(user_id, "assistant", tool_response)

    elif detect_graph_command(user_input):
        plot_gym_sessions()
        tool_response = "📊 Showing your gym progress chart!"

    elif detect_pie_command(user_input):
        plot_food_pie_chart()
        tool_response = "🥧 Food breakdown chart generated."

    elif detect_timer_command(user_input):
        parsed = parse_timer_command(user_input)
        if parsed:
            duration, task = parsed
            tool_response = f"⏱ Timer started for {task}: {duration} seconds."
        else:
            tool_response = "❌ Couldn't understand the timer."

    else:
        recipe_reply = handle_recipe_query(user_input)
        if recipe_reply:
            tool_response = recipe_reply

    # -------------------------
    # 🧠 MEMORY CONTEXT
    # -------------------------
    past = get_contextual_memory(user_id)
    memory_context = "\n".join([m["content"] for m in past])

    # -------------------------
    # 🔄 CONVERT CHAT HISTORY
    # -------------------------
    lc_history = [
        HumanMessage(content=m) if role == "user" else AIMessage(content=m)
        for role, m in chat_history
    ]

    # -------------------------
    # 📊 FOOD SUMMARY TRIGGER
    # -------------------------
    if "analyze food" in user_input.lower():
        user_input += "\n\n" + summarize_food_logs()

    # -------------------------
    # 🤖 CALL LLM
    # -------------------------
    try:
        response = chain.invoke({
            "input": user_input,
            "chat_history": lc_history,
            "context": memory_context
        })
    except Exception as e:
        return f"❌ LLM Error: {str(e)}"

    save_message(user_id, "assistant", response.content)

    if tool_response:
        return f"{tool_response}\n\nAssistant: {response.content}"

    return response.content
