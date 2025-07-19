import streamlit as st
import time
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from agent import run_habit_agent
from tools import detect_timer_command, parse_timer_command
from memory import clear_user_memory, is_plot_request, plot_memory_graph

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "gym_data" not in st.session_state:
    st.session_state.gym_data = []

# Page config
st.set_page_config(page_title="🏋️‍♀️ Habit Tracker Assistant", layout="centered")

# Custom CSS for modern dark mode UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body, .stApp {
        background-color: #0f1117;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        color: #66bb6a !important;
        font-weight: 600;
    }

    .stTextArea textarea {
        background-color: #1e1e2e;
        color: #f0f0f0;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px;
    }

    .stButton>button {
        background: linear-gradient(to right, #43cea2, #185a9d);
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: bold;
        transition: background 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #66bb6a, #1de9b6);
    }

    .stDataFrame {
        background-color: #1e1e2e;
        color: white;
        border-radius: 10px;
        overflow: hidden;
    }

    .element-container:has(.stButton) {
        margin-bottom: 20px;
    }

    hr {
        border: none;
        height: 1px;
        background-color: #444;
        margin: 20px 0;
    }

    .chat-bubble {
        padding: 12px;
        border-radius: 12px;
        margin: 6px 0;
        line-height: 1.5;
    }

    .user-msg {
        background-color: #2d2d3a;
        border-left: 5px solid #64b5f6;
    }

    .assistant-msg {
        background-color: #1f2c2c;
        border-left: 5px solid #81c784;
    }

    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Fancy gradient header
st.markdown("""
<h1 style='text-align: center;
           background: linear-gradient(to right, #43cea2, #185a9d);
           -webkit-background-clip: text;
           -webkit-text-fill-color: transparent;
           font-size: 2.8em;
           font-weight: 700;
           margin-top: 0;
           margin-bottom: 0.5em;'>🏋️‍♀️ Habit Tracker Assistant</h1>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Clear memory
with st.expander("🧹 Clear Data & Memory", expanded=False):
    if st.button("Clear All", use_container_width=True):
        clear_user_memory(user_id="default")
        st.session_state.chat_history.clear()
        st.session_state.gym_data.clear()
        st.success("✅ Memory and logs cleared.")

# Extract gym session info
def extract_gym_data(text):
    text = text.lower()
    gym_keywords = ["gym", "workout", "bench press", "deadlift"]
    if not any(word in text for word in gym_keywords):
        return None

    match = re.search(r"(\d+)\s*(minutes|min|hrs|hours|hr)", text)
    if not match:
        return None
    duration = int(match.group(1))

    now = datetime.now()
    datetime_obj = now

    if "day before yesterday" in text:
        datetime_obj = now - pd.Timedelta(days=2)
    elif "yesterday" in text:
        datetime_obj = now - pd.Timedelta(days=1)
    elif "today" in text:
        datetime_obj = now
    else:
        date_match = re.search(r"\b(\d{1,2})(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", text)
        if not date_match:
            date_match = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(st|nd|rd|th)?", text)
        try:
            if date_match:
                if date_match.lastindex == 3:
                    day = int(date_match.group(1))
                    month = date_match.group(3)
                else:
                    month = date_match.group(1)
                    day = int(date_match.group(2))
                month_num = datetime.strptime(month[:3], "%b").month
                year = now.year
                datetime_obj = datetime(year, month_num, day, now.hour, now.minute, now.second)
        except Exception:
            pass

    return {"DateTime": datetime_obj, "Duration": duration}

# Input UI
st.markdown("### 💬 Talk to Your Habit Assistant")
user_input = st.text_area("Type something like: *'I did gym for 45 minutes yesterday'*", key="input")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    reply = None

    # Timer
    if detect_timer_command(user_input):
        parsed = parse_timer_command(user_input)
        if parsed:
            duration, task = parsed
            st.success(f"⏱️ Starting a {duration}-second timer for: {task}")
            placeholder = st.empty()
            for i in range(duration, 0, -1):
                mins, secs = divmod(i, 60)
                placeholder.markdown(f"### ⏳ {mins:02d}:{secs:02d} remaining for **{task}**")
                time.sleep(1)
            placeholder.markdown(f"### ✅ Timer complete for: **{task}**")
            st.session_state.chat_history.append(("assistant", f"Timer complete for: {task}"))

    # Gym entry
    else:
        gym_data = extract_gym_data(user_input)
        if gym_data:
            st.session_state.gym_data.append(gym_data)
            formatted_time = gym_data['DateTime'].strftime('%B %d, %Y %I:%M %p')
            msg = f"💪 Logged your gym session: {gym_data['Duration']} minutes on {formatted_time}"
            st.success(msg)
            st.session_state.chat_history.append(("assistant", msg))

        # Plot request (use session data instead of external memory)
        elif is_plot_request(user_input):
            if st.session_state.gym_data:
                df = pd.DataFrame(st.session_state.gym_data)
                df = df.sort_values("DateTime")
                st.markdown("### 📊 Gym Progress Chart")
                fig, ax = plt.subplots()
                fig.patch.set_facecolor('#121212')
                ax.set_facecolor('#1e1e1e')
                ax.plot(df["DateTime"], df["Duration"], marker='o', linestyle='-', color='#81c784')
                ax.set_xlabel("Date & Time", color='white')
                ax.set_ylabel("Duration (minutes)", color='white')
                ax.set_title("Gym Duration Trend", color='white')
                ax.tick_params(axis='x', colors='white', rotation=45)
                ax.tick_params(axis='y', colors='white')
                ax.grid(True, color='#444')
                st.pyplot(fig)
                st.session_state.chat_history.append(("assistant", "📈 Here's your gym session chart!"))
            else:
                st.warning("⚠️ No gym data available to plot.")
                st.session_state.chat_history.append(("assistant", "No gym data available to plot."))

        # Default agent response
        else:
            reply = run_habit_agent(user_input, st.session_state.chat_history)
            if reply == "__PLOT_GYM_GRAPH__":
                if st.session_state.gym_data:
                    df = pd.DataFrame(st.session_state.gym_data)
                    df = df.sort_values("DateTime")
                    st.markdown("### 📈 Gym Progress Chart")
                    fig, ax = plt.subplots()
                    fig.patch.set_facecolor('#121212')
                    ax.set_facecolor('#1e1e1e')
                    ax.plot(df["DateTime"], df["Duration"], marker='o', linestyle='-', color='#81c784')
                    ax.set_xlabel("Date & Time", color='white')
                    ax.set_ylabel("Duration (minutes)", color='white')
                    ax.set_title("Gym Duration Trend", color='white')
                    ax.tick_params(axis='x', colors='white', rotation=45)
                    ax.tick_params(axis='y', colors='white')
                    ax.grid(True, color='#444')
                    st.pyplot(fig)
                    st.session_state.chat_history.append(("assistant", "📈 Here's your gym session chart!"))
                else:
                    st.warning("⚠️ No gym data found to plot.")
                    st.session_state.chat_history.append(("assistant", "No gym data found to plot."))
            else:
                st.session_state.chat_history.append(("assistant", reply))
                st.markdown(f"<div class='chat-bubble assistant-msg'><strong>Assistant:</strong> {reply}</div>", unsafe_allow_html=True)

# Chat history display
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🧾 Chat History")
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for role, msg in st.session_state.chat_history:
    css_class = "user-msg" if role == "user" else "assistant-msg"
    st.markdown(f"""
        <div class="chat-bubble {css_class}">
            <strong>{role.capitalize()}:</strong> {msg}
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Show gym logs table
if st.session_state.gym_data:
    st.markdown("---")
    st.markdown("### 🗓️ Logged Gym Sessions")
    df = pd.DataFrame(st.session_state.gym_data)
    df["DateTime"] = df["DateTime"].apply(lambda dt: dt.strftime('%Y-%m-%d %I:%M %p'))
    st.dataframe(df, use_container_width=True)
