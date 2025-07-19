from datetime import datetime
import re
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import dateparser

# Storage
gym_sessions = []
food_log = []

# --- Triggers ---
def detect_gym_trigger(text):
    keywords = ["gym", "workout", "exercise", "fitness", "training", "lifting", "cardio", "squats", "weights"]
    return any(word in text.lower() for word in keywords)

def detect_food_trigger(text):
    keywords = ["ate", "eating", "had for lunch", "had for dinner", "snack", "meal", "breakfast", "lunch", "dinner", "brunch"]
    return any(word in text.lower() for word in keywords)

def detect_graph_command(text):
    keywords = ["graph", "chart", "plot", "visualize", "display", "show"]
    gym_words = ["gym", "workout", "training", "fitness"]
    return any(k in text.lower() for k in keywords) and any(g in text.lower() for g in gym_words)

def detect_pie_command(text):
    keywords = ["pie", "chart", "graph", "plot", "visualize"]
    food_words = ["food", "meal", "calorie", "nutrition"]
    return any(k in text.lower() for k in keywords) and any(f in text.lower() for f in food_words)

def detect_timer_command(text):
    timer_phrases = ["start", "set", "begin", "run", "initiate", "do", "remind", "track", "countdown"]
    has_time = bool(re.search(r'(\d+)\s*(second|sec|s|minute|min|m)', text.lower()))
    has_intent = any(word in text.lower() for word in timer_phrases)
    return has_time and has_intent

# --- Date parsing helper ---
def extract_date_from_text(text):
    parsed = dateparser.parse(text, settings={'RELATIVE_BASE': datetime.now()})
    if parsed:
        return parsed.date()
    return datetime.now().date()

# --- Duration extraction helper ---
def extract_duration(text):
    match = re.search(r'(\d+)\s*(min|minutes|mins|hrs|hours|hr)', text.lower())
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        return value * 60 if 'hour' in unit or 'hr' in unit else value
    return 0

# --- Timer parser ---
def parse_timer_command(text):
    match = re.search(r'(\d+)\s*(second|sec|s|minute|min|m)', text.lower())
    task_match = re.search(r'for (.+)', text.lower())

    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2)

    duration = value * 60 if 'min' in unit else value
    task = task_match.group(1).strip() if task_match else "your task"
    return duration, task.capitalize()

# --- Logging ---
def log_gym_session(text, user_id="default"):
    date = extract_date_from_text(text)
    duration = extract_duration(text)
    session = {
        "user": user_id,
        "timestamp": datetime.combine(date, datetime.now().time()).isoformat(),
        "note": text,
        "duration": duration
    }
    gym_sessions.append(session)
    return f"💪 Logged your gym session: \"{text}\" ({duration} min) on {date}"

def log_food_entry(text, user_id="default"):
    entry = {
        "user": user_id,
        "timestamp": datetime.now().isoformat(),
        "note": text
    }
    food_log.append(entry)
    return f"🍽️ Noted what you ate: \"{text}\" at {entry['timestamp']}"

# --- Plotting ---
def plot_gym_sessions():
    if not gym_sessions:
        print("No gym sessions to plot.")
        return

    df = pd.DataFrame(gym_sessions)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    total_duration = df.groupby('date')['duration'].sum()

    plt.figure(figsize=(8, 4))
    total_duration.plot(kind='bar', color='skyblue')
    plt.title("🏋️‍♀️ Gym Sessions Over Time")
    plt.xlabel("Date")
    plt.ylabel("Duration (minutes)")
    plt.tight_layout()
    plt.show()

def plot_food_pie_chart():
    if not food_log:
        print("No food entries to plot.")
        return

    categories = []
    for entry in food_log:
        text = entry['note'].lower()
        if any(word in text for word in ["breakfast", "morning"]):
            categories.append("Breakfast")
        elif any(word in text for word in ["lunch", "afternoon"]):
            categories.append("Lunch")
        elif any(word in text for word in ["dinner", "night", "evening"]):
            categories.append("Dinner")
        else:
            categories.append("Snack/Other")

    counts = Counter(categories)

    plt.figure(figsize=(5, 5))
    plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=140)
    plt.title("🍽️ Food Intake Breakdown")
    plt.axis('equal')
    plt.show()

# --- Interactive CLI (optional) ---
if __name__ == "__main__":
    while True:
        user_input = input("You: ")

        if detect_graph_command(user_input):
            plot_gym_sessions()

        elif detect_pie_command(user_input):
            plot_food_pie_chart()

        elif detect_timer_command(user_input):
            result = parse_timer_command(user_input)
            if result:
                duration, task = result
                print(f"⏱️ Timer started for {task} — {duration} seconds.")

        elif detect_food_trigger(user_input):
            print(log_food_entry(user_input))

        elif detect_gym_trigger(user_input):
            print(log_gym_session(user_input))

        elif user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting Habit Tracker. Stay consistent!")
            break

        else:
            print("❓ I didn't understand that. Try again.")
