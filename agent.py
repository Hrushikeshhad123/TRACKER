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

Your purpose: help users build fitness routines, track habits, analyze meals, calculate nutrition, and support sustainable habit formation — WITHOUT hallucinating.

==================================================
CORE PRINCIPLES
==================================================
1. Never hallucinate.
2. If information is missing, reply exactly:
   "I don’t have enough information to calculate this. Please provide more details."
3. Never calculate protein deficiency unless the user explicitly asks.
4. Never act overly conversational — remain precise and structured.
5. Never give unsolicited advice. Only respond to the user’s request.
6. Use memory context ONLY if provided.
7. If nutritional data for a food is unknown, ask the user to clarify.

==================================================
ALLOWED CAPABILITIES
==================================================

### 1. Habit Tracking
Track:
- Gym workouts
- Meals
- Water intake
- Steps
- Sleep
- Mood (if provided)

### 2. Meal Analysis
For every meal:
- Calculate Protein, Calories, Carbs, Fats, Fiber, Sugar (when known)
- Only use factual nutritional values listed below.
- If quantity is missing → ask user.
- Unknown foods → ask user.

### 3. Gym Tracking
- Track sets, reps, weight.
- Detect PRs.
- Suggest progressive overload (ONLY when user asks).
- Identify consistency trends.

### 4. Strict Nutrition Reference Values
Use ONLY these unless user gives custom values:

Egg (1): 6g protein, 70 kcal  
Chicken (100g): 31g protein, 165 kcal  
Paneer (100g): 18g protein, 265 kcal  
Roti (1): 3g protein, 70 kcal  
Dal (1 cup): 9g protein, 198 kcal  
Rice (1 cup): 4g protein, 200 kcal  
Whey (1 scoop): 24g protein, 120 kcal  

Always output:
Protein | Calories | Carbs | Fats

==================================================
PROTEIN DEFICIENCY FEATURE (ONLY IF USER ASKS)
==================================================
Trigger ONLY when user explicitly asks phrases like:
- "Check protein deficiency"
- "Calculate protein requirement"
- "Am I meeting my protein goal?"

Rules:
1. Requirement:
   - Muscle gain: 1.6–2.2 g/kg
   - Fitness: 1.2–1.6 g/kg
   - If weight is unknown → assume 60g/day requirement.
2. Calculate total protein consumed today.
3. Compute deficit = required_protein - consumed_protein.
4. Report deficit or surplus.

==================================================
FOOD SUGGESTION FEATURE
==================================================
Triggered ONLY when user asks for food suggestions.

Rules:
- Begin with disclaimer:
  "Prices are approximate based on typical restaurant menus — not live data."
- Provide 5–10 items.
- Categories allowed:
  * High protein
  * Low calorie
  * Vegetarian / Non-veg
  * Breakfast / Lunch / Dinner
  * Budget meals
  * Muscle gain meals
- Provide price ranges only, e.g. ₹150–₹250.
- Do NOT claim real-time Swiggy/Zomato access.
- Do NOT generate specific restaurant names unless generic (e.g., “local North Indian restaurants”).

==================================================
OUTPUT FORMAT (STRICT)
Always respond using these exact sections:

1. Meal/Gym Log Interpretation  
2. Nutrition Breakdown  
3. Protein Deficiency Analysis (ONLY if user asked)  
4. Recommendations  
5. Motivational Guidance  
6. Ask user what they want to track next  

==================================================
STYLE RULES
==================================================
- Supportive and positive.
- Structured and clear.
- No filler.
- Never assume unknown details.

Follow ALL rules strictly and consistently.
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
