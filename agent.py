# agent.py

import os
from dotenv import load_dotenv
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
# Load API Key
# -------------------------------------------------------------------
GROQ_API_KEY="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"

# -------------------------------------------------------------------
# LLM (MODEL MUST BE VALID)
# -------------------------------------------------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-70b-versatile",
    temperature=0.3
)

# -------------------------------------------------------------------
# PROMPT TEMPLATE
# -------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a supportive Habit Tracker Assistant. "
     "You help users track gym and food habits."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("system", "Relevant memory:\n{context}"),
    ("human", "{input}")
])

chain = prompt | llm

# -------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------
def run_habit_agent(user_input, chat_history, user_id="default"):

    # Save user message
    save_message(user_id, "user", user_input)

    tool_response = None

    # --- Tool triggers ---
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
        tool_response = "🥧 Showing your food breakdown chart!"

    elif detect_timer_command(user_input):
        parsed = parse_timer_command(user_input)
        if parsed:
            duration, task = parsed
            tool_response = f"⏱️ Timer started for {task}: {duration} seconds."
        else:
            tool_response = "❌ Could not understand timer command."

    else:
        recipe_reply = handle_recipe_query(user_input)
        if recipe_reply and ("🍽️" in recipe_reply or "🔥" in recipe_reply):
            tool_response = recipe_reply

    # --- Load memory ---
    raw_memory = get_contextual_memory(user_id)
    memory_context = "\n".join([m["content"] for m in raw_memory])

    # --- Convert chat history ---
    lc_history = []
    for role, msg in chat_history:
        if role == "user":
            lc_history.append(HumanMessage(content=msg))
        else:
            lc_history.append(AIMessage(content=msg))

    # --- Food analysis ---
    if "analyze food" in user_input.lower():
        summary = summarize_food_logs()
        user_input += f"\n\nFood Summary:\n{summary}"

    # --- LLM call ---
    try:
        response = chain.invoke({
            "input": user_input,
            "chat_history": lc_history,
            "context": memory_context
        })
    except Exception as e:
        return f"❌ LLM error: {str(e)}"

    # Save assistant reply
    save_message(user_id, "assistant", response.content)

    # Return tool + LLM response
    if tool_response:
        return f"{tool_response}\n\nAssistant: {response.content}"

    return response.content
