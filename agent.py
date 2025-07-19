# agent.py
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from memory import save_message, get_contextual_memory
from tools import detect_gym_trigger, detect_food_trigger, log_gym_session, log_food_entry

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Smart Habit Tracker Assistant.

Your role is to help users track, improve, and reflect on their habits through meaningful conversations. You remember the user's goals, preferences, past actions, and feedback to guide them better over time.

Always use context from previous interactions to provide relevant, helpful responses.

Here are your key behaviors:
1. Greet the user by name if known, and maintain a warm, motivational tone.
2. Ask clarifying questions if a goal or habit is vague.
3. Track patterns and remind the user of past behavior or uncompleted tasks.
4. Give brief summaries of progress when appropriate.
5. Encourage consistency without being pushy.

You MUST always consider past messages and memories stored in ChromaDB to generate responses.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("system", "Here is relevant past memory:\n{context}"),
    ("human", "{input}")
])

chain = prompt | llm

def run_habit_agent(user_input, chat_history, user_id="default"):
    latest_user_msg = chat_history[-1][1]
    save_message("user", latest_user_msg, user_id=user_id)

    tool_response = None
    if detect_gym_trigger(user_input):
        tool_response = log_gym_session(user_input, user_id)
        save_message("assistant", tool_response, user_id)
    elif detect_food_trigger(user_input):
        tool_response = log_food_entry(user_input, user_id)
        save_message("assistant", tool_response, user_id)

    # Use last few messages to build better memory context
    recent_history_text = " ".join(msg for _, msg in chat_history[-3:])
    memory_context = get_contextual_memory(recent_history_text, user_id=user_id, k=8)

    memory_as_messages = []
    for line in memory_context.split("\n"):
        if line.startswith("user:"):
            memory_as_messages.append(HumanMessage(content=line[5:].strip()))
        elif line.startswith("assistant:"):
            memory_as_messages.append(AIMessage(content=line[9:].strip()))

    formatted_history = memory_as_messages + [HumanMessage(content=user_input)]

    response = chain.invoke({
        "input": user_input,
        "chat_history": formatted_history,
        "context": memory_context
    })

    save_message("assistant", response.content, user_id=user_id)

    if tool_response:
        return f"{tool_response}\n\nAssistant: {response.content}"
    return response.content
