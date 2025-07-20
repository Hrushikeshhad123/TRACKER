from datetime import datetime
import re
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import dateparser
import requests

# 🧠 LLM utility (Groq-based)
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "llama3-8b-8192"

def query_llm(user_input, system_instruction):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.3
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ LLM API Error:", e)
        return ""

# 🧾 Storage
gym_sessions = []
food_log = []

# 📥 Load Indian Food Dataset
def load_food_data(path="IndianFoodDatasetXLS.xlsx"):
    try:
        df = pd.read_excel(path)
        df = df[['TranslatedRecipeName', 'TranslatedIngredients', 'TotalTimeInMins', 'Servings', 'Cuisine', 'Course', 'Diet']]
        df.dropna(subset=['TranslatedRecipeName', 'TranslatedIngredients'], inplace=True)
        df['TranslatedIngredients'] = df['TranslatedIngredients'].apply(lambda x: x.lower())
        return df
    except Exception as e:
        print(f"❌ Failed to load food data: {e}")
        return pd.DataFrame()

food_df = load_food_data()

# 🔥 Calorie Estimation
def estimate_calories(ingredients_text):
    calorie_dict = {
        "rice": 130, "potato": 110, "paneer": 265, "chicken": 239, "egg": 78,
        "milk": 42, "ghee": 115, "oil": 120, "dal": 120, "bread": 80,
        "cheese": 113, "curd": 60, "butter": 102, "flour": 100, "sugar": 60
    }
    calories = sum(cal for item, cal in calorie_dict.items() if item in ingredients_text)
    return calories

# 🍽️ Recipe Suggestion
def suggest_recipe(course="Lunch", diet="Vegetarian"):
    if food_df.empty:
        return "⚠️ Recipe data not available."

    filtered = food_df[
        (food_df['Course'].str.contains(course, case=False, na=False)) &
        (food_df['Diet'].str.contains(diet, case=False, na=False))
    ]

    if filtered.empty:
        return f"⚠️ No {diet} {course.lower()} recipes found."

    recipe = filtered.sample(1).iloc[0]
    calories = estimate_calories(recipe['TranslatedIngredients'])

    return f"""
🍽️ {recipe['TranslatedRecipeName']}
🕒 Time: {recipe['TotalTimeInMins']} mins | 🍛 Course: {recipe['Course']} | 🥗 Diet: {recipe['Diet']}
🔥 Estimated Calories: ~{calories} kcal
📋 Ingredients: {recipe['TranslatedIngredients']}
"""

# 🤖 Intent Detection using LLM
def detect_gym_trigger(text):
    reply = query_llm(text, "Does this message describe a gym or workout session? Reply with true or false.")
    return "true" in reply.lower()

def detect_food_trigger(text):
    reply = query_llm(text, "Is this message logging a food entry or something the user ate? Reply with true or false.")
    return "true" in reply.lower()

def detect_graph_command(text):
    reply = query_llm(text, "Is the user asking to show a graph related to gym/workout? Reply with true or false.")
    return "true" in reply.lower()

def detect_pie_command(text):
    reply = query_llm(text, "Is the user asking for a pie chart or graph related to food/nutrition? Reply with true or false.")
    return "true" in reply.lower()

def detect_timer_command(text):
    reply = query_llm(text, "Does the message ask to start or set a timer with a specific time? Reply with true or false.")
    return "true" in reply.lower()

# ⏱️ Timer Parsing
def parse_timer_command(text):
    instruction = """
Extract the timer duration in seconds and task name from this message.
Return JSON like: {"duration": 300, "task": "Reading"}
If no task found, use "your task".
"""
    reply = query_llm(text, instruction)
    try:
        parsed = eval(reply) if isinstance(reply, str) else reply
        return parsed["duration"], parsed["task"]
    except:
        return None

# 🍲 Recipe and Calorie Query
def handle_recipe_query(text):
    instruction = """
Understand if the user is:
- Asking for recipe suggestions (e.g., dinner, lunch, breakfast)
- Asking for calorie info for a known Indian dish

Return a JSON:
{
  "intent": "suggest_recipe" or "calorie_query",
  "course": "Lunch" or "Dinner" or "Breakfast",
  "diet": "Vegetarian" or "Non-Vegetarian",
  "recipe_name": "rajma chawal"
}
"""
    reply = query_llm(text, instruction)
    try:
        parsed = eval(reply) if isinstance(reply, str) else reply
        if parsed["intent"] == "suggest_recipe":
            return suggest_recipe(parsed.get("course", "Lunch"), parsed.get("diet", "Vegetarian"))
        elif parsed["intent"] == "calorie_query":
            recipe_name = parsed.get("recipe_name", "").strip().lower()
            result = food_df[food_df['TranslatedRecipeName'].str.lower() == recipe_name]
            if not result.empty:
                recipe = result.iloc[0]
                calories = estimate_calories(recipe['TranslatedIngredients'])
                return f"🔥 Calories in {recipe['TranslatedRecipeName']}: ~{calories} kcal"
            else:
                return f"❌ Could not find recipe '{recipe_name}'."
    except:
        return "❓ Try asking: 'Suggest dinner for vegetarian' or 'Calories in Paneer Butter Masala'"

# 📅 Helpers
def extract_date_from_text(text):
    parsed = dateparser.parse(text, settings={'RELATIVE_BASE': datetime.now()})
    return parsed.date() if parsed else datetime.now().date()

def extract_duration(text):
    match = re.search(r'(\d+)\s*(min|minutes|mins|hrs|hours|hr)', text.lower())
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        return value * 60 if 'hour' in unit or 'hr' in unit else value
    return 0

# ✅ Logging
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

# 📊 Plotting
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

# 🧪 CLI for Testing (Optional)
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
            print(handle_recipe_query(user_input))
