# agent.py

import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

# Import tools
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

load_dotenv()
GROQ_API_KEY = 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

# LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",
    temperature=0.3
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a Smart Habit Tracker Assistant.

Your role is to help users track and reflect on their gym workouts and food habits.

### Behavior Guidelines
- Friendly, supportive tone.
- Summaries, insights, warnings.
- Use memory context.
- Ask clarifying questions when needed.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("system", "Relevant past memory:\n{context}"),
    ("human", "{input}")
])

chain = prompt | llm


# -------------------------------------------------------------------
# MAIN AGENT FUNCTION (fixed version)
# -------------------------------------------------------------------

def run_habit_agent(user_input, chat_history, user_id="default"):

    # ---------------------------------------------------------
    # 1. Save the incoming user message
    # ---------------------------------------------------------
    save_message(user_id, "user", user_input)

    tool_response = None

    # ---------------------------------------------------------
    # 2. Tool triggers
    # ---------------------------------------------------------
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
        tool_response = "🥧 Here's your food intake breakdown."

    elif detect_timer_command(user_input):
        result = parse_timer_command(user_input)
        if result:
            duration, task = result
            tool_response = f"⏱️ Timer started for {task} — {duration} seconds."
        else:
            tool_response = "❌ Couldn't parse timer info."

    else:
        # recipe handler
        recipe_reply = handle_recipe_query(user_input)
        if recipe_reply and ("🍽️" in recipe_reply or "🔥" in recipe_reply):
            tool_response = recipe_reply

    # ---------------------------------------------------------
    # 3. Load contextual memory (convert to string)
    # ---------------------------------------------------------
    raw_memory = get_contextual_memory(user_id)
    memory_context = "\n".join(m["content"] for m in raw_memory if "content" in m)

    # ---------------------------------------------------------
    # 4. Convert Streamlit chat history → LangChain Messages
    # ---------------------------------------------------------
    lc_history = []
    for role, msg in chat_history:
        if role == "user":
            lc_history.append(HumanMessage(content=msg))
        else:
            lc_history.append(AIMessage(content=msg))

    # ---------------------------------------------------------
    # 5. Add triggered food summary
    # ---------------------------------------------------------
    if "analyze food" in user_input.lower() or "diet analysis" in user_input.lower():
        summary = summarize_food_logs()
        user_input += f"\n\nHere is your food data:\n{summary}"

    # ---------------------------------------------------------
    # 6. Call LLM (safe invocation)
    # ---------------------------------------------------------
    response = chain.invoke({
        "input": user_input,
        "chat_history": lc_history,
        "context": memory_context
    })

    # Save assistant output
    save_message(user_id, "assistant", response.content)

    # ---------------------------------------------------------
    # 7. Return combined tool + LLM message if needed
    # ---------------------------------------------------------
    if tool_response:
        return f"{tool_response}\n\nAssistant: {response.content}"

    return response.content
