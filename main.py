import streamlit as st
import time
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import base64

from agent import run_habit_agent
from tools import detect_timer_command, parse_timer_command
from memory import clear_user_memory, is_plot_request

# ---------- Session Initialization ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "gym_data" not in st.session_state:
    st.session_state.gym_data = []
if "input_area" not in st.session_state:
    if "input_area" not in st.session_state:
        st.session_state.input_area = ""



# ---------- Authentication ----------
def login():
    st.markdown("## 🔐 Login Required")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username.lower().strip() == "hrushikesh mama" and password == "mamamami":
                st.session_state["authenticated"] = True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    login()
    st.stop()

# ---------- Page Configuration ----------
st.set_page_config(page_title="🏋️‍♀️ Habit Tracker Assistant", layout="centered")

# ---------- Custom Styles ----------
st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #0f1117;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    .container {
        max-width: 800px;
        margin: auto;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 10px;
        background-color: #1e1e2e;
        margin-top: 10px;
    }
    .chat-bubble {
        padding: 14px;
        border-radius: 10px;
        margin: 10px 0;
        line-height: 1.6;
        font-size: 15px;
    }
    .user-msg {
        background-color: #2d2d3a;
        border-left: 5px solid #64b5f6;
        text-align: right;
    }
    .assistant-msg {
        background-color: #1f2c2c;
        border-left: 5px solid #81c784;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Header with Image ----------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/png;base64,{encoded}"

image_data_url = get_base64_image("unnamed.png")

st.markdown(f"""
<div class='container' style='text-align: center;'>
    <img src='{image_data_url}' style='width: 70px; height: 70px; border-radius: 50%; margin-bottom: 10px;' />
    <h1 style='background: linear-gradient(to right, #43cea2, #185a9d); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Habit Tracker Assistant</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------- Clear Data ----------
with st.expander("🧹 Clear Data & Memory", expanded=False):
    if st.button("Clear All", use_container_width=True):
        clear_user_memory(user_id="default")
        st.session_state.chat_history.clear()
        st.session_state.gym_data.clear()
        st.success("✅ Memory and logs cleared.")

# ---------- Gym Data Extractor ----------
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

# ---------- Chat Input (Enter to Send) ----------
st.markdown("### 💬 Talk to Your Habit Assistant")

# Handle input

def handle_input():
    user_input = st.session_state.input_area.strip()
    if not user_input:
        return
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.input_area = ""

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

    elif (gym_data := extract_gym_data(user_input)):
        st.session_state.gym_data.append(gym_data)
        formatted_time = gym_data['DateTime'].strftime('%B %d, %Y %I:%M %p')
        msg = f"💪 Logged your gym session: {gym_data['Duration']} minutes on {formatted_time}"
        st.success(msg)
        st.session_state.chat_history.append(("assistant", msg))

    elif is_plot_request(user_input):
        if st.session_state.gym_data:
            df = pd.DataFrame(st.session_state.gym_data).sort_values("DateTime")
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

    else:
        reply = run_habit_agent(user_input, st.session_state.chat_history)
        if reply == "__PLOT_GYM_GRAPH__":
            if st.session_state.gym_data:
                df = pd.DataFrame(st.session_state.gym_data).sort_values("DateTime")
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

st.text_input(
    label="Message",
    key="input_area",
    on_change=handle_input,
    placeholder="Type a message and press Enter...",
    label_visibility="collapsed"
)

# ---------- Chat History ----------
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

# ---------- Show Logged Gym Sessions ----------
if st.session_state.gym_data:
    st.markdown("---")
    st.markdown("### 🗓️ Logged Gym Sessions")
    df = pd.DataFrame(st.session_state.gym_data)
    df["DateTime"] = df["DateTime"].apply(lambda dt: dt.strftime('%Y-%m-%d %I:%M %p'))
    st.dataframe(df.style.set_properties(**{
        'background-color': '#1e1e2e',
        'color': 'white',
        'border-color': 'white'
    }), use_container_width=True)
def plot_food_graph(food_df):

    # Example: plot total calories per day
    daily_calories = food_df.groupby("date")["calories"].sum()

    plt.figure(figsize=(10, 4))
    daily_calories.plot(kind="bar", color="orange")
    plt.title("Daily Calorie Intake")
    plt.xlabel("Date")
    plt.ylabel("Calories")
    st.pyplot(plt)
