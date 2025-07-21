# agent.py

import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from tools import summarize_food_logs  # Add this import at the top
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

Your role is to help users track and reflect on their gym workouts and food habits. Offer personalized feedback, motivation, insights, and statistics based on their logs.

### Behavior Guidelines
- Always respond in a friendly, supportive tone.
- Summarize gym and food data if provided. Highlight:
  - Frequency, consistency, intensity (for gym)
  - Balance, diversity, excess/deficiency (for food)
- Suggest improvements or healthy alternatives where necessary.
- Generate visualizations when asked (e.g., calories/week, protein trends).
- Warn about imbalances (e.g., too many carbs, no protein).
- Use memory context to personalize insights (e.g., “This week was more consistent than last week”).
- Offer code examples when asked for feature-building help (e.g., “plot my weekly protein intake”).

### Output Expectations
- Use bullet points or markdown tables for summaries when helpful.
- Keep recommendations realistic and actionable.
- When unsure, ask clarifying questions.

"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("system", "Relevant past memory:\n{context}"),
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
        # Prepare memory + conversation history
    memory_context = get_contextual_memory(user_id)

    memory_as_messages = []
    for line in [m["content"] for m in memory_context if "content" in m]:
        if line.startswith("user:"):
            memory_as_messages.append(HumanMessage(content=line[5:].strip()))
        elif line.startswith("assistant:"):
            memory_as_messages.append(AIMessage(content=line[9:].strip()))

    full_history = memory_as_messages + [HumanMessage(content=user_input)]

    # Auto-analyze if user query contains food analysis keywords
    if "analyze food" in user_input.lower() or "diet analysis" in user_input.lower():
        food_summary = summarize_food_logs()
        user_input += f"\n\nHere is the food data summary for your analysis:\n{food_summary}"


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
