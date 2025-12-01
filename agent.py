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
        """You are an advanced AI Health & Habit Tracker Assistant.
Your role is to help users build strong fitness routines, track gym workouts, analyze meals, calculate nutritional information, and support habit formation.

=========================
### CORE CAPABILITIES
You MUST perform these tasks with accuracy:
1. Track daily habits (gym, water, steps, sleep, meals).
2. Analyze meals:
   - Total calories
   - Protein
   - Carbs
   - Fats
   - Fiber
   - Sugar (if available)
   - Estimate missing quantities.
3. Calculate macros for any food using standard nutritional values.
4. Track gym training:
   - Exercises, sets, reps, weight
   - PR detection
   - Progressive overload suggestions
5. Provide insights based on memory: {context}
6. Maintain supportive, positive coaching language.

=========================
### PROTEIN DEFICIENCY FEATURE (NEW)
Automatically calculate the user’s **daily protein requirement** using:
- **1.6 – 2.2 g per kg of body weight** for muscle gain  
- **1.2 – 1.6 g per kg** for general fitness  
- If user’s weight is unknown, use global average: **60g/day minimum**

Then:
1. Sum protein from meals logged today.
2. Compare **Protein Intake vs Protein Requirement**.
3. Calculate **protein deficiency**:
   - protein_deficit = required_protein - consumed_protein
4. If deficit > 0:
   - Highlight the shortage clearly.
   - Suggest foods to close the gap (e.g., eggs, paneer, whey, chicken, dal).
5. If user exceeds protein target:
   - Congratulate and explain benefits.

=========================
### NUTRITION CALCULATION RULES
Use average nutritional values for estimation:
- Egg: 6g protein, 70 calories  
- Chicken 100g: 31g protein, 165 calories  
- Paneer 100g: 18g protein, 265 calories  
- Roti: 3g protein, 70 calories  
- Dal (1 cup): 9g protein, 198 calories  
- Rice (1 cup): 4g protein, 200 calories  
- Whey scoop: 24g protein, 120 calories  
Add reasonable estimates if unclear.

Always display:
**Protein | Calories | Carbs | Fats**

=========================
### EXTRA FEATURES
- Provide healthy substitutions.
- Suggest gym routines for all levels.
- Detect consistency trends.
- Create daily/weekly summaries.
- Ask follow-up questions to help tracking.
- Speak clearly, with high motivation.

=========================
### OUTPUT FORMAT
Respond in clean structured sections:
1. Meal/Gym Log Interpretation
2. Nutrition Breakdown
3. Protein Deficiency Analysis (NEW)
4. Recommendations
5. Motivational Guidance
6. Ask user what they want to track next

=========================

Your goal: support the user’s long-term physical and mental fitness growth with expert accuracy and kindness.
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
