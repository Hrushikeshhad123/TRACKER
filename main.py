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


# ======================================================
#  UI CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="Habit Tracker Assistant",
    page_icon="💪",
    layout="centered"
)


# ======================================================
#  THEME COLORS (Adaptive)
# ======================================================
THEMES = {
    "dashboard": {
        "bg": "#F5EFFF",
        "card": "#FFFFFF",
        "accent": "#7F56D9",
        "text": "#2D2A32"
    },
    "details": {
        "bg": "#FFFFFF",
        "card": "#FFFFFF",
        "accent": "#43A047",
        "text": "#282828"
    },
    "recipes": {
        "bg": "#E8FBE8",
        "card": "#FFFFFF",
        "accent": "#4CAF50",
        "text": "#1B1B1B"
    }
}


# ------------------ APPLY THEME -----------------------
def apply_theme(section="dashboard"):
    t = THEMES[section]
    st.markdown(
        f"""
        <style>
            body {{ background-color: {t['bg']}; }}
            .stApp {{ background-color: {t['bg']}; }}
            h1, h2, h3, h4, h5, h6 {{
                color: {t['text']} !important;
                font-family: 'Inter', sans-serif;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


apply_theme("dashboard")

# ======================================================
#  SESSION INIT
# ======================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "gym_data" not in st.session_state:
    st.session_state.gym_data = []

if "input_area" not in st.session_state:
    st.session_state.input_area = ""


# ======================================================
#  LOGIN PAGE (Styled)
# ======================================================
def login():
    apply_theme("details")

    st.markdown(
        """
        <h2 style="text-align:center; margin-top:20px;">
            🔐 Secure Login
        </h2>
        """,
        unsafe_allow_html=True
    )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if username.lower().strip() == "hrushikesh" and password == "tracker123":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect username or password ❌")


if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    login()
    st.stop()


# ======================================================
#  HEADER WITH AVATAR (Modern)
# ======================================================
def load_base64_img(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


avatar = load_base64_img("unnamed.png")

st.markdown(
    f"""
    <div style="
        text-align:center;
        padding: 10px 0;
    ">
        <img src="data:image/png;base64,{avatar}"
             style="width:85px;height:85px;border-radius:50%;box-shadow:0 4px 12px rgba(0,0,0,0.2);" />
        <h1 style="margin-top:12px;font-size:32px;
                   background: linear-gradient(to right, #7F56D9, #5433FF);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;">
           Habit Tracker Assistant
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# ======================================================
#  CLEAR MEMORY EXPANDER
# ======================================================
with st.expander("🧹 Clear Data & Memory"):
    if st.button("Clear All"):
        st.session_state.chat_history.clear()
        st.session_state.gym_data.clear()
        clear_user_memory("default")
        st.success("All memory cleared!")


# ======================================================
#  SMART PARSING FUNCTIONS
# ======================================================
def extract_gym_data(text):
    text = text.lower()
    keywords = ["gym", "workout", "bench", "squat", "deadlift"]

    if not any(k in text for k in keywords):
        return None

    m = re.search(r"(\d+)\s*(min|minutes|hrs|hours)", text)
    if not m:
        return None

    duration = int(m.group(1))
    now = datetime.now()

    return {"DateTime": now, "Duration": duration}


# ======================================================
#  CHAT INPUT HANDLER
# ======================================================
def handle_input():
    user_input = st.session_state.input_area.strip()
    if user_input == "":
        return

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.input_area = ""

    # TIMER LOGIC
    if detect_timer_command(user_input):
        dur, task = parse_timer_command(user_input)
        st.success(f"⏱ Timer started: {task}")

        ph = st.empty()
        for i in range(dur, 0, -1):
            ph.markdown(f"### ⏳ {i} seconds left for **{task}**")
            time.sleep(1)

        st.success(f"✅ Timer Finished: {task}")
        st.session_state.chat_history.append(("assistant", f"Timer finished for {task}"))
        return

    # GYM LOG
    data = extract_gym_data(user_input)
    if data:
        st.session_state.gym_data.append(data)
        st.success(f"💪 Workout logged: {data['Duration']} minutes")
        st.session_state.chat_history.append(("assistant", "Gym data recorded!"))
        return

    # CHART REQUEST
    if is_plot_request(user_input):
        plot_gym_chart()
        return

    # NORMAL MESSAGE
    reply = run_habit_agent(user_input, st.session_state.chat_history)
    st.session_state.chat_history.append(("assistant", reply))


# ======================================================
#  GYM CHART
# ======================================================
def plot_gym_chart():
    if not st.session_state.gym_data:
        st.warning("No workout data yet.")
        return

    df = pd.DataFrame(st.session_state.gym_data).sort_values("DateTime")

    st.subheader("📈 Gym Progress Chart")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df["DateTime"], df["Duration"], marker="o")
    ax.set_ylabel("Minutes")
    ax.grid(True)

    st.pyplot(fig)
    st.session_state.chat_history.append(("assistant", "Gym progress chart updated!"))


# ======================================================
#  CHAT UI (Modern Bubbles)
# ======================================================
st.markdown("### 💬 Chat")

chat_container = st.container()

with chat_container:
    for role, msg in st.session_state.chat_history:
        color = "#E8E0FF" if role == "assistant" else "#D1E8FF"
        align = "left" if role == "assistant" else "right"

        st.markdown(
            f"""
            <div style="
                background:{color};
                padding:10px 16px;
                border-radius:12px;
                margin:8px 0;
                width:75%;
                float:{align};
                box-shadow:0 2px 8px rgba(0,0,0,0.15);
            ">
                <strong>{role.capitalize()}:</strong> {msg}
            </div>
            <div style="clear:both;"></div>
            """,
            unsafe_allow_html=True
        )


# ======================================================
#  INPUT FIELD
# ======================================================
st.text_input(
    "Message",
    key="input_area",
    on_change=handle_input,
    placeholder="Type a message..."
)


# ======================================================
#  LOGGED SESSIONS
# ======================================================
if st.session_state.gym_data:
    st.markdown("---")
    st.subheader("📅 Logged Gym Sessions")

    df = pd.DataFrame(st.session_state.gym_data)
    df["DateTime"] = df["DateTime"].dt.strftime("%Y-%m-%d %H:%M")

    st.dataframe(df, use_container_width=True)
