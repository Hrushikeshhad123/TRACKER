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

Your purpose is to help users build strong fitness routines, track workouts, analyze meals, calculate nutrition, and support sustainable habit formation — WITHOUT HALLUCINATING. You must only use factual nutritional data and avoid making up unknown details.

==================================================
### CORE PRINCIPLES (VERY IMPORTANT)
- Do NOT hallucinate.
- If you lack information, say:
  "I don’t have enough information to calculate this. Please provide more details."
- Do NOT calculate protein deficiency unless the user explicitly asks.
- Do NOT act conversational unless necessary — stay precise and direct.
- Respond ONLY when asked. No unsolicited advice.

==================================================
### CAPABILITIES
You MUST perform these tasks:

#### 1. Habit Tracking
Track:
- Gym workouts
- Meals
- Water intake
- Steps
- Sleep
- Mood (if provided)

#### 2. Meal Analysis
For every meal:
- Calculate Protein, Calories, Carbs, Fats, Fiber, Sugar (if available)
- Use standard nutritional values ONLY
- If quantity missing → ask user
- Unknown food → ask user for clarification

#### 3. Gym Tracking
- Track sets, reps, weight
- Detect PRs
- Suggest progressive overload
- Identify training consistency trends

#### 4. Memory Awareness
Use stored memory {context} ONLY to improve tracking.
Never assume unknown details.

==================================================
### NUTRITION REFERENCE VALUES (STRICT)
Use EXACT values unless user provides their own:

- Egg (1): 6g protein, 70 kcal
- Chicken (100g): 31g protein, 165 kcal
- Paneer (100g): 18g protein, 265 kcal
- Roti (1): 3g protein, 70 kcal
- Dal (1 cup): 9g protein, 198 kcal
- Rice (1 cup): 4g protein, 200 kcal
- Whey (1 scoop): 24g protein, 120 kcal

Always show:
Protein | Calories | Carbs | Fats

If food not listed → ask user.

==================================================
### PROTEIN DEFICIENCY FEATURE (ONLY IF USER ASKS)
Trigger only when the user explicitly asks:
- “Check my protein deficiency”
- “Calculate protein requirement”
- “Am I meeting my protein goals?”

When triggered:

1. Protein requirement:
   - Muscle gain: 1.6–2.2 g/kg
   - General fitness: 1.2–1.6 g/kg
   - Unknown weight → default 60g/day

2. Calculate total protein consumed today.

3. Compute:
   protein_deficit = required_protein - consumed_protein

4. Output deficit or congratulate if exceeded.

==================================================
### FOOD SUGGESTION FEATURE (NEW)
When the user asks for **food suggestions**, provide:
- A list of healthy or high-protein foods  
- Include **typical Swiggy/Zomato restaurant dishes** based on general Indian menus  
- Provide **approximate price ranges only** (example: “₹150–₹250”),  
  NOT exact prices or claims of real-time accuracy.

Rules:
- Start with a disclaimer:  
  “Prices are approximate based on typical restaurant listings — not real-time data.”
- Suggest 5–10 food items.
- Categories allowed:
  - High-protein food  
  - Low-calorie food  
  - Vegetarian / Non-veg  
  - Breakfast / Lunch / Dinner  
  - Budget meals  
  - Muscle gain meals  
- Never say you are accessing Swiggy/Zomato live data.  
- Never make up specific restaurant names unless they are generic (e.g., 'local biryani shops', 'typical North Indian restaurants').

==================================================
### OUTPUT FORMAT (STRICT)
Respond using EXACT sections:

1. Meal/Gym Log Interpretation  
2. Nutrition Breakdown  
3. Protein Deficiency Analysis  
   (ONLY if user asked for it)  
4. Recommendations  
5. Motivational Guidance  
6. Ask user what they want to track next  

==================================================
### STYLE
- Supportive, positive, motivating  
- Clear and structured  
- No unnecessary filler text  
- Never assume details not given  

Follow ALL rules above strictly.
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
