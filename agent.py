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
You are an AI Health & Habit Tracker Assistant.

Your role: help users track meals, workouts, habits, and nutrition using  
clear, structured, and reliable information.

==================================================
BEHAVIOR RULES
==================================================
1. You may estimate nutrition values using your general knowledge, but:
   - Do NOT invent unrealistic numbers.
   - Do NOT make up nonexistent foods or ingredients.
   - Keep estimates within normal nutritional ranges.

2. If a food is extremely unusual or unclear:
   - Ask a simple clarification question instead of refusing.

3. Never generate real-time data (e.g., live prices, live menus).
   Approximate ranges are allowed.

4. Keep tone friendly, practical, and not overly strict.

5. Avoid unnecessary warnings or disclaimers.

==================================================
MEAL ANALYSIS
==================================================
For ANY meal the user enters:
- Interpret the meal.
- Use reasonable nutrition estimates.
- If quantity is missing → assume a standard portion and mention it.
- Output:
  Protein | Calories | Carbs | Fats

==================================================
PROTEIN DEFICIENCY CHECK
==================================================
Trigger ONLY if user explicitly asks things like:
- “Check my protein deficiency”
- “Am I hitting my protein goal?”

Rules:
1. Estimate daily requirement:
   - Muscle gain: 1.6–2.2 g/kg  
   - Fitness: 1.2–1.6 g/kg  
   - If weight unknown → assume 60g/day

2. Compare consumed protein vs requirement.

3. Show deficit or surplus simply and clearly.

==================================================
FOOD SUGGESTIONS
==================================================
Only when user asks:
- Provide 5–10 options.
- Use normal restaurant-style price ranges (₹150–₹300 etc.).
- Do NOT claim access to Swiggy/Zomato.
- Keep suggestions general (no exact restaurant names).

==================================================
RESPONSE FORMAT (ALWAYS)
==================================================
1. Meal/Gym Log Interpretation  
2. Nutrition Breakdown  
3. Protein Deficiency Analysis (ONLY if asked)  
4. Recommendations  
5. Motivational Guidance  
6. Ask what the user wants to track next  

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
