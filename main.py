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

# Dark mode custom CSS
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #e0e0e0;
    }
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }
    .stTextArea textarea {
        background-color: #1e1e1e;
        color: white;
    }
    .stButton button {
        background-color: #2e7d32;
        color: white;
    }
    .stDataFrame {
        background-color: #1e1e1e;
        color: white;
    }
    .element-container:has(.stButton) {
        margin-bottom: 20px;
    }
    hr {
        border-color: #444;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: #66bb6a;'>🏋️‍♀️ Habit Tracker Assistant</h1>", unsafe_allow_html=True)
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

    # Timer handling
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

    # Gym data extraction
    elif gym_data := extract_gym_data(user_input):
        st.session_state.gym_data.append(gym_data)
        formatted_time = gym_data['DateTime'].strftime('%B %d, %Y %I:%M %p')
        msg = f"💪 Logged your gym session: {gym_data['Duration']} minutes on {formatted_time}"
        st.success(msg)
        st.session_state.chat_history.append(("assistant", msg))

    # Plot request handling
    elif is_plot_request(user_input):
        match = re.search(r"plot (?:graph|chart) for (\w+)", user_input.lower())
        habit_type = match.group(1) if match else "gym"
        try:
            img_base64 = plot_memory_graph(user_id="default", habit_type=habit_type)
            if img_base64:
                st.markdown(f"### 📊 {habit_type.capitalize()} Progress")
                st.image(img_base64)
                st.session_state.chat_history.append(("assistant", f"📊 Here's your {habit_type} graph!"))
            else:
                st.warning(f"⚠️ No {habit_type} data available to plot.")
                st.session_state.chat_history.append(("assistant", f"No {habit_type} data available to plot."))
        except Exception as e:
            st.error(f"❌ Error generating plot: {e}")
            st.session_state.chat_history.append(("assistant", f"Plotting error: {e}"))

    # Default LLM response handling
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
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.grid(True, color='#444')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                st.session_state.chat_history.append(("assistant", "📈 Here's your gym session chart!"))
            else:
                st.warning("⚠️ No gym data found to plot.")
                st.session_state.chat_history.append(("assistant", "No gym data found to plot."))
        else:
            st.session_state.chat_history.append(("assistant", reply))
            st.markdown(f"<div style='padding:10px; background-color:#263238; color:#e0e0e0; border-left:5px solid #42a5f5;'>"
                        f"<strong>Assistant:</strong> {reply}</div>", unsafe_allow_html=True)

# Chat history display
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🧾 Chat History")
for role, msg in st.session_state.chat_history:
    color = "#2c2c2c" if role == "user" else "#1f1f1f"
    st.markdown(f"""
        <div style="background-color: {color}; padding: 10px; border-radius: 8px; margin-bottom: 5px; color: #e0e0e0;">
        <strong>{role.capitalize()}:</strong> {msg}
        </div>
    """, unsafe_allow_html=True)

# Show gym logs table
if st.session_state.gym_data:
    st.markdown("---")
    st.markdown("### 🗓️ Logged Gym Sessions")
    df = pd.DataFrame(st.session_state.gym_data)
    df["DateTime"] = df["DateTime"].apply(lambda dt: dt.strftime('%Y-%m-%d %I:%M %p'))
    st.dataframe(df, use_container_width=True)
