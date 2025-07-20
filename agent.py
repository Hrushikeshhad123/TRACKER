# agent.py

import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from memory import save_message, get_contextual_memory
from tools import (
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

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",
    temperature=0.3
)

# Smart, contextual assistant behavior
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Smart Habit Tracker Assistant.

Your job is to help users track and reflect on their gym and food habits, and provide feedback, motivation, and stats based on their logs.

Behavior rules:
1. Always be warm and encouraging.
2. Track gym workouts and food logs from natural messages.
3. Visualize user trends (workouts, meals) when asked.
4. Suggest recipes or calories using your tools if needed.
5. Use past memory context to tailor your answers.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("system", "Here is relevant past memory:\n{context}"),
    ("human", "{input}")
])

chain = prompt | llm


def run_habit_agent(user_input, chat_history, user_id="default"):
    # Save incoming message
    save_message(user_id, "user", user_input)

    tool_response = None

    # Run tool-based logic if triggers match
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

    # If not a direct tool match, try recipe handler
    elif not tool_response:
        response = handle_recipe_query(user_input)
        if response and "🍽️" in response or "🔥" in response:
            tool_response = response

    # Prepare memory + conversation history
    memory_context = get_contextual_memory(user_id)

    memory_as_messages = []
    for line in [m["content"] for m in memory_context if "content" in m]:
        if line.startswith("user:"):
            memory_as_messages.append(HumanMessage(content=line[5:].strip()))
        elif line.startswith("assistant:"):
            memory_as_messages.append(AIMessage(content=line[9:].strip()))

    full_history = memory_as_messages + [HumanMessage(content=user_input)]

    # Run main LLM reasoning
    response = chain.invoke({
        "input": user_input,
        "chat_history": full_history,
        "context": memory_context
    })

    # Save LLM output
    save_message(user_id, "assistant", response.content)

    if tool_response:
        return f"{tool_response}\n\nAssistant: {response.content}"
    return response.content
