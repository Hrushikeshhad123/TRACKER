import streamlit as st
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from agent import run_habit_agent
from tools import detect_timer_command, parse_timer_command
from memory import clear_user_memory


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Habit Tracker Assistant", layout="centered")


# ---------------------------------------------------------
# CSS STYLING (Pastel Blue + Mint Theme)
# ---------------------------------------------------------
st.markdown("""
<style>

body, html, .stApp {
    background-color: #F9FAFC;
    font-family: 'Inter', sans-serif;
    color: #2D2D2D;
}

.container {
    max-width: 850px;
    margin: auto;
}

/* Header */
.header-logo {
    width: 80px;
    height: 80px;
    border-radius: 50%;
}

.main-title {
    font-size: 36px;
    font-weight: 700;
    background: linear-gradient(90deg, #A9C9FF, #7AE1C3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Chat Container */
.chat-container {
    max-height: 480px;
    overflow-y: auto;
    padding: 18px;
    border-radius: 16px;
    background-color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

/* Chat Bubbles */
.chat-bubble {
    padding: 14px 16px;
    border-radius: 14px;
    margin-bottom: 14px;
    font-size: 15px;
    line-height: 1.5;
}

.user-msg {
    background-color: #DCEBFF;
    margin-left: 80px;
    text-align: right;
}

.assistant-msg {
    background-color: #E5FFF7;
    margin-right: 80px;
    text-align: left;
}

/* Input Box */
input[type="text"] {
    border-radius: 10px !important;
}

/* Data Cards */
.data-card {
    background-color: #FFFFFF;
    padding: 14px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "gym_data" not in st.session_state:
    st.session_state.gym_data = []

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


# ---------------------------------------------------------
# LOGIN
# ---------------------------------------------------------
def login():
    st.markdown("## 🔐 Login Required")

    with st.form("login_form", clear_on_submit=False):
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if user.lower().strip() == "hrushikesh" and pwd == "tracker123":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Wrong username or password")


if not st.session_state.authenticated:
    login()
    st.stop()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="container" style="text-align:center; margin-top:20px;">
    <h1 class="main-title">Habit Tracker Assistant</h1>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# CLEAR MEMORY BUTTON
# ---------------------------------------------------------
if st.button("🧹 Clear All Data & Memory"):
    st.session_state.chat_history.clear()
    st.session_state.gym_data.clear()
    clear_user_memory("default")
    st.success("Memory cleared!")


# ---------------------------------------------------------
# GYM DATA EXTRACTOR
# ---------------------------------------------------------
def extract_gym_data(text):
    text = text.lower()
    if not any(word in text for word in ["gym", "workout", "bench", "deadlift"]):
        return None

    match = re.search(r"(\d+)\s*(minutes|min|hours|hrs|hr)", text)
    if not match:
        return None

    minutes = int(match.group(1))
    now = datetime.now()
    dt = now

    if "yesterday" in text:
        dt = now - pd.Timedelta(days=1)
    elif "day before yesterday" in text:
        dt = now - pd.Timedelta(days=2)

    return {"DateTime": dt, "Duration": minutes}


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------
user_input = st.text_input("Type your message...")


if user_input:
    st.session_state.chat_history.append(("user", user_input))

    # Timer
    if detect_timer_command(user_input):
        duration, task = parse_timer_command(user_input)
        placeholder = st.empty()

        for i in range(duration, 0, -1):
            placeholder.markdown(f"### ⏳ {i}s left for **{task}**")
            time.sleep(1)

        placeholder.markdown(f"### ✅ Timer complete for **{task}**")
        st.session_state.chat_history.append(("assistant", f"Timer completed for: {task}"))

    # Gym logging
    elif (gd := extract_gym_data(user_input)):
        st.session_state.gym_data.append(gd)
        st.session_state.chat_history.append(
            ("assistant", f"💪 Logged your **{gd['Duration']} min** gym session.")
        )

    # Chatbot
    else:
        bot_reply = run_habit_agent(user_input, st.session_state.chat_history)
        st.session_state.chat_history.append(("assistant", bot_reply))

    st.rerun()


# ---------------------------------------------------------
# CHAT DISPLAY
# ---------------------------------------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for role, msg in st.session_state.chat_history:
    css = "user-msg" if role == "user" else "assistant-msg"
    st.markdown(
        f'<div class="chat-bubble {css}"><strong>{role.capitalize()}:</strong> {msg}</div>',
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# GYM DATA DISPLAY
# ---------------------------------------------------------
if st.session_state.gym_data:
    st.markdown("### 🗓️ Logged Gym Sessions")
    df = pd.DataFrame(st.session_state.gym_data)
    df["DateTime"] = df["DateTime"].dt.strftime("%Y-%m-%d %I:%M %p")
    st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------
# PLOT BUTTON
# ---------------------------------------------------------
if st.button("📈 Show Gym Progress"):
    if st.session_state.gym_data:
        df = pd.DataFrame(st.session_state.gym_data).sort_values("DateTime")

        st.markdown("### 📊 Gym Progress")

        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor("#F9FAFC")
        ax.set_facecolor("white")

        ax.plot(df["DateTime"], df["Duration"], marker="o", color="#7AE1C3")
        ax.set_title("Workout Duration Over Time", color="#2D2D2D")
        ax.set_ylabel("Minutes")
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        st.pyplot(fig)

    else:
        st.warning("No gym data available.")
