import os
import json
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import base64
from io import BytesIO

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
import base64
import re

MEMORY_LOG_FILE = "memory_log.json"

def load_user_messages(user_id="default"):
    if not os.path.exists(MEMORY_LOG_FILE):
        return []

    with open(MEMORY_LOG_FILE, "r") as f:
        data = json.load(f)
    return data.get(user_id, [])

def plot_memory_graph(user_id="default", plot_type="gym"):
    messages = load_user_messages(user_id)
    data = {}

    for entry in messages:
        message = entry.get("user", "")
        date_str = entry.get("timestamp", "")
        date_obj = datetime.fromisoformat(date_str).date()

        # Gym
        if plot_type == "gym":
            match = re.search(r'(\d+)\s*(minutes|min|hrs|hours)', message.lower())
            if match:
                value = int(match.group(1))
                if "hr" in match.group(2):
                    value *= 60  # convert hours to minutes
                data[date_obj] = data.get(date_obj, 0) + value

        # Food
        elif plot_type == "food":
            match = re.search(r'ate.*?(\d+)\s*(kcal|calories)', message.lower())
            if match:
                value = int(match.group(1))
                data[date_obj] = data.get(date_obj, 0) + value

    if not data:
        return None

    # Sort by date
    dates = sorted(data)
    values = [data[date] for date in dates]

    plt.figure(figsize=(8, 4))
    plt.plot(dates, values, marker='o')
    plt.title(f'{plot_type.capitalize()} Stats Over Time')
    plt.xlabel("Date")
    plt.ylabel("Minutes" if plot_type == "gym" else "Calories")
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64

def is_plot_request(text):
    return "plot" in text.lower() and ("gym" in text.lower() or "food" in text.lower())

class HabitMemory:
    def __init__(self, memory_file=os.path.join(DATA_DIR, "habit_memory.json")):
        self.memory_file = memory_file
        self.memory = self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                return json.load(f)
        return {}

    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=4)

    def add_entry(self, user_id, prompt):
        entry = self.extract_data_from_prompt(prompt)
        if entry is None:
            return False

        if user_id not in self.memory:
            self.memory[user_id] = []

        self.memory[user_id].append(entry)
        self.save_memory()
        return True

    def extract_data_from_prompt(self, prompt):
        prompt = prompt.lower()
        now = datetime.now()

        # Parse date
        if "yesterday" in prompt:
            date = now - timedelta(days=1)
        else:
            date = now  # Default to today

        date_str = date.strftime("%Y-%m-%d")

        # Extract values
        cal_match = re.search(r'(\d+)\s*(k?calories?)', prompt)
        min_match = re.search(r'(\d+)\s*(minutes?|mins?)', prompt)
        hr_match = re.search(r'(\d+(\.\d+)?)\s*(hours?|hrs?)', prompt)

        if cal_match:
            return {"date": date_str, "type": "calories", "value": int(cal_match.group(1))}
        elif hr_match:
            return {"date": date_str, "type": "hours", "value": float(hr_match.group(1))}
        elif min_match:
            minutes = int(min_match.group(1))
            return {"date": date_str, "type": "hours", "value": round(minutes / 60, 2)}

        return None

    def get_entries(self, user_id):
        return self.memory.get(user_id, [])

    def plot_graph(self, user_id):
        entries = self.get_entries(user_id)
        if not entries:
            print("⚠️ No entries found.")
            return None

        data = {}
        for entry in entries:
            date = entry["date"]
            value = entry["value"]
            key = entry["type"]

            if key not in data:
                data[key] = {}

            data[key][date] = data[key].get(date, 0) + value

        if not data:
            print("⚠️ No valid data to plot.")
            return None

        plt.figure(figsize=(10, 5))
        for key, date_map in data.items():
            dates = sorted(date_map.keys())
            values = [date_map[d] for d in dates]
            plt.plot(dates, values, marker='o', label=key.capitalize())

        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Workout Habit Progress")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()
        return img_base64

# Save a raw user/assistant message for chat history
def save_message(user_id, role, content):
    filepath = os.path.join(DATA_DIR, f"{user_id}_messages.json")

    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }

    messages = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            messages = json.load(f)

    messages.append(message)

    with open(filepath, "w") as f:
        json.dump(messages, f, indent=2)

# Retrieve recent message history
def get_contextual_memory(user_id, limit=5):
    filepath = os.path.join(DATA_DIR, f"{user_id}_messages.json")
    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, "r") as f:
            messages = json.load(f)
        return messages[-limit:]
    except Exception as e:
        print(f"Failed to load contextual memory: {e}")
        return []

# Clear chat memory
def clear_user_memory(user_id):
    filepath = os.path.join(DATA_DIR, f"{user_id}_messages.json")
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Failed to clear memory: {e}")

# Determine if the prompt requests a graph
def is_plot_request(prompt):
    keywords = ["graph", "plot", "chart", "visualize", "show progress"]
    prompt = prompt.lower()
    return any(kw in prompt for kw in keywords)

# External wrapper to call from main
def plot_memory_graph(user_id):
    memory = HabitMemory()
    return memory.plot_graph(user_id)
import matplotlib.pyplot as plt
import pandas as pd
import base64
import io
import re
from datetime import datetime

def parse_calories_from_text(text):
    # Look for patterns like "500 calories", "ate 600 kcal"
    match = re.search(r"(\d+)\s*(calories|kcal)", text.lower())
    return int(match.group(1)) if match else None

def parse_workout_minutes(text):
    # Match patterns like "30 minutes", "1 hour", "45 mins", "2 hr"
    text = text.lower()
    hr_match = re.search(r"(\d+(?:\.\d+)?)\s*(hour|hr|hrs)", text)
    min_match = re.search(r"(\d+)\s*(minutes|min)", text)
    
    minutes = 0
    if hr_match:
        minutes += float(hr_match.group(1)) * 60
    if min_match:
        minutes += int(min_match.group(1))
    return int(minutes) if minutes > 0 else None

def parse_date_from_text(text):
    # Simplistic: fallback to current date for now
    return datetime.today().date()

import os
import json
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# === Message Handling ===

def save_message(user_id, role, content):
    filepath = os.path.join(DATA_DIR, f"{user_id}_messages.json")
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }

    messages = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            messages = json.load(f)

    messages.append(message)

    with open(filepath, "w") as f:
        json.dump(messages, f, indent=2)


def get_contextual_memory(user_id, limit=5):
    filepath = os.path.join(DATA_DIR, f"{user_id}_messages.json")
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r") as f:
            messages = json.load(f)
        return messages[-limit:]
    except Exception as e:
        print(f"Failed to load contextual memory: {e}")
        return []


def clear_user_memory(user_id):
    filepath = os.path.join(DATA_DIR, f"{user_id}_messages.json")
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Failed to clear memory: {e}")

# === Graph Plotting ===

def plot_memory_graph(habit_type, user_id="default"):
    log_data = load_user_messages(user_id)
    filtered_data = [
        entry for entry in log_data
        if habit_type.lower() in entry["message"].lower()
    ]

    if not filtered_data:
        return None  # or return a default image

    timestamps = [
        datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
        for entry in filtered_data
    ]
    counts = list(range(1, len(timestamps) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, counts, marker="o")
    plt.xlabel("Date")
    plt.ylabel("Habit Count")
    plt.title(f"Habit Graph for {habit_type}")
    plt.xticks(rotation=45)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64
