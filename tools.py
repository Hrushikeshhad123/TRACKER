# tools.py

import re
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter

from langchain_groq import ChatGroq

# -------------------------------------------------------
# 🔐 HARD-CODED API KEY
# -------------------------------------------------------
GROQ_API_KEY = "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"

# -------------------------------------------------------
# ⚡ FAST MODEL FOR INTENT DETECTION
# -------------------------------------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
    temperature=0.1
)

# ----------------------- LLM Helper -----------------------
def ask_llm(question: str) -> str:
    try:
        response = llm.invoke([{"role": "user", "content": question}])
        return response.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


# ----------------------- FOOD DATA -----------------------
def load_food_data(path="IndianFoodDatasetXLS.xlsx"):
    try:
        df = pd.read_excel(path)
        df = df[['TranslatedRecipeName','TranslatedIngredients','TotalTimeInMins','Course','Diet']]
        df.dropna(inplace=True)
        df["TranslatedIngredients"] = df["TranslatedIngredients"].str.lower()
        return df
    except:
        return pd.DataFrame()

food_df = load_food_data()


# ----------------------- INTENT DETECTION -----------------------
def detect_gym_trigger(text):
    return "true" in ask_llm(f"Is this about gym? Reply true/false:\n{text}").lower()

def detect_food_trigger(text):
    return "true" in ask_llm(f"Is this about food logging? Reply true/false:\n{text}").lower()

def detect_graph_command(text):
    return "true" in ask_llm(f"Is the user asking for a gym graph? Reply true/false:\n{text}").lower()

def detect_pie_command(text):
    return "true" in ask_llm(f"Does the user want a food pie chart? Reply true/false:\n{text}").lower()

def detect_timer_command(text):
    return "true" in ask_llm(f"Does this message ask for a timer? Reply true/false:\n{text}").lower()

def parse_timer_command(text):
    reply = ask_llm(
        f"""
Return JSON: {{"duration":SECONDS,"task":"TASK"}}
Extract timer info from:
{text}
"""
    )
    try:
        parsed = eval(reply)
        return parsed["duration"], parsed["task"]
    except:
        return None


# ----------------------- FOOD/RECIPE LOGIC -----------------------
def estimate_calories(ingredients):
    base = {"rice":130,"potato":110,"paneer":265,"chicken":239,"egg":78,
            "milk":42,"ghee":115,"oil":120,"dal":120,"bread":80}
    return sum(cal for item, cal in base.items() if item in ingredients)

def suggest_recipe(course, diet):
    df = food_df[
        (food_df["Course"].str.contains(course, case=False)) &
        (food_df["Diet"].str.contains(diet, case=False))
    ]

    if df.empty:
        return f"No {diet} {course} recipes found."

    item = df.sample(1).iloc[0]
    calories = estimate_calories(item["TranslatedIngredients"])

    return f"""
🍽 {item['TranslatedRecipeName']}
🔥 ~{calories} kcal
⏱ {item['TotalTimeInMins']} mins
📋 {item['TranslatedIngredients']}
"""

def calorie_lookup(name):
    row = food_df[food_df["TranslatedRecipeName"].str.lower() == name.lower()]
    if row.empty:
        return f"❌ No recipe found: {name}"

    item = row.iloc[0]
    cal = estimate_calories(item["TranslatedIngredients"])
    return f"🔥 {name} approx {cal} kcal"


def handle_recipe_query(text):
    reply = ask_llm(
        f"""
Return JSON:
{{
 "intent":"suggest_recipe" OR "calorie_query",
 "course":"Lunch",
 "diet":"Vegetarian",
 "recipe_name":"paneer butter masala"
}}
User text: {text}
"""
    )
    try:
        parsed = eval(reply)

        if parsed["intent"] == "suggest_recipe":
            return suggest_recipe(parsed.get("course","Lunch"), parsed.get("diet","Vegetarian"))
        else:
            return calorie_lookup(parsed.get("recipe_name",""))
    except:
        return None


# ----------------------- LOGGING -----------------------
gym_sessions = []
food_log = []

def extract_date(text):
    return datetime.now().date()

def extract_duration(text):
    match = re.search(r'(\d+)\s*(min|hour|hr)', text.lower())
    if not match:
        return 0
    val = int(match.group(1))
    return val * 60 if "hour" in match.group(2) else val

def log_gym_session(text, user_id="default"):
    dur = extract_duration(text)
    date = extract_date(text)
    gym_sessions.append({"user":user_id,"duration":dur,"date":date})
    return f"💪 Logged gym: {dur} minutes on {date}"

def log_food_entry(text, user_id="default"):
    food_log.append({"user":user_id,"note":text})
    return f"🍽 Logged food: {text}"


# ----------------------- GRAPHS -----------------------
def plot_gym_sessions():
    if not gym_sessions:
        return
    df = pd.DataFrame(gym_sessions)
    df.groupby("date")["duration"].sum().plot(kind="bar")
    plt.title("Gym Duration")
    plt.show()

def plot_food_pie_chart():
    if not food_log:
        return
    categories = []
    for f in food_log:
        t = f["note"].lower()
        if "breakfast" in t: categories.append("Breakfast")
        elif "lunch" in t: categories.append("Lunch")
        elif "dinner" in t: categories.append("Dinner")
        else: categories.append("Other")
    count = Counter(categories)
    plt.pie(count.values(), labels=count.keys(), autopct="%.1f%%")
    plt.show()


# ----------------------- SUMMARY -----------------------
def summarize_food_logs():
    if not food_log:
        return "No food logs yet."
    cats = {"Breakfast":0,"Lunch":0,"Dinner":0,"Other":0}
    for f in food_log:
        t=f["note"].lower()
        if "breakfast" in t: cats["Breakfast"]+=1
        elif "lunch" in t: cats["Lunch"]+=1
        elif "dinner" in t: cats["Dinner"]+=1
        else: cats["Other"]+=1

    return f"""
🥗 Food Summary:
Breakfast: {cats['Breakfast']}
Lunch: {cats['Lunch']}
Dinner: {cats['Dinner']}
Other: {cats['Other']}
"""
