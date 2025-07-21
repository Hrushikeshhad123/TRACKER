import os
import json
import re
import base64
from datetime import datetime, timedelta
from io import BytesIO
import matplotlib.pyplot as plt

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Utility Functions ----------

def safe_parse_date(date_str):
    try:
        return datetime.fromisoformat(date_str).date()
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").date()
        except Exception:
            return datetime.today().date()

# ---------- Message Handling ----------

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
            try:
                messages = json.load(f)
            except json.JSONDecodeError:
                messages = []

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

# ---------- Memory Class for Habits ----------

class HabitMemory:
    def __init__(self, memory_file=os.path.join(DATA_DIR, "habit_memory.json")):
        self.memory_file = memory_file
        self.memory = self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=4)

    def prune_old_entries(self, user_id, days=30):
        entries = self.get_entries(user_id)
        cutoff = datetime.now().date() - timedelta(days=days)
        pruned = [entry for entry in entries if datetime.strptime(entry["date"], "%Y-%m-%d").date() >= cutoff]
        self.memory[user_id] = pruned
        self.save_memory()

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
        date = now
        if "yesterday" in prompt:
            date = now - timedelta(days=1)
        date_str = date.strftime("%Y-%m-%d")

        # Extract values
        cal_match = re.search(r'(\d+)\s*(k?calories?)', prompt)
        min_match = re.search(r'(\d+)\s*(minutes?|mins?)', prompt)
        hr_match = re.search(r'(\d+(?:\.\d+)?)\s*(hours?|hrs?)', prompt)

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
        self.prune_old_entries(user_id)
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
        plt.title("Habit Progress")
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

# ---------- Graph Wrapper ----------

def plot_memory_graph(user_id="default"):
    memory = HabitMemory()
    return memory.plot_graph(user_id)

# ---------- Plot Trigger Detector ----------

def is_plot_request(prompt):
    keywords = ["graph", "plot", "chart", "visualize", "show progress"]
    prompt = prompt.lower()
    return any(kw in prompt for kw in keywords)
