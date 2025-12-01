import streamlit as st
import time
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import base64

# -------------------- PAGE SETTINGS --------------------
st.set_page_config(
    page_title="Habit Tracker Assistant",
    page_icon="💬",
    layout="centered"
)

# -------------------- GLOBAL STYLES --------------------
st.markdown("""
<style>

body, .stApp {
    background-color: #FAFAFA !important;
    font-family: 'Inter', sans-serif;
}

.section-header {
    font-weight: 700;
    font-size: 26px;
    margin-bottom: 6px;
}

.card {
    background: #FFFFFF;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 3px 15px rgba(0,0,0,0.07);
    margin-bottom: 20px;
}

.chat-bubble-user {
    background: #4B4B4B;
    color: white;
    padding: 12px 16px;
    border-radius: 14px;
    margin: 8px 0;
    width: 75%;
    float: right;
}

.chat-bubble-assistant {
    background: #EAF1FF;
    color: #333;
    padding: 12px 16px;
    border-radius: 14px;
    margin: 8px 0;
    width: 75%;
    float: left;
}

.nav-bar {
    display: flex;
    justify-content: space-around;
    padding: 12px 0;
    background: #2F2F2F;
    border-radius: 12px;
    margin-bottom: 18px;
}

.nav-item {
    color: white !important;
    padding: 10px 16px;
    font-weight: 600;
    border-radius: 10px;
}

.nav-item-selected {
    background: #6A4CE3;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)


# -------------------- STATE --------------------
if "tab" not in st.session_state:
    st.session_state.tab = "Chat"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------- NAVIGATION --------------------
st.markdown("""
<div class="nav-bar">
    <a class="nav-item {chat}" href="?tab=Chat">Chat</a>
    <a class="nav-item {gym}" href="?tab=Gym">Gym Logs</a>
    <a class="nav-item {food}" href="?tab=Food">Food</a>
</div>
""".format(
    chat="nav-item-selected" if st.session_state.tab == "Chat" else "",
    gym="nav-item-selected" if st.session_state.tab == "Gym" else "",
    food="nav-item-selected" if st.session_state.tab == "Food" else "",
), unsafe_allow_html=True)


# -------------------- CHAT TAB --------------------
if st.session_state.tab == "Chat":
    st.markdown("<div class='section-header'>💬 Chat Assistant</div>", unsafe_allow_html=True)

    chat_box = st.container()

    with chat_box:
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"<div class='chat-bubble-user'>{message}</div><br>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble-assistant'>{message}</div><br>", unsafe_allow_html=True)

    user_input = st.text_input("Message")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", "This is a placeholder reply."))
        st.rerun()


# -------------------- GYM TAB --------------------
elif st.session_state.tab == "Gym":
    st.markdown("<div class='section-header'>🏋️ Gym Log</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <strong>Log your workouts here.</strong><br>
        Light background. Clean aesthetic.  
    </div>
    """, unsafe_allow_html=True)


# -------------------- FOOD TAB --------------------
elif st.session_state.tab == "Food":
    st.markdown("<div class='section-header'>🍽️ Food Tracking</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <strong>Track your meals and calories.</strong><br>
        Light color cards. Soft shadows.
    </div>
    """, unsafe_allow_html=True)
