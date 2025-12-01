# tools.py
import re
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
from langchain_groq import ChatGroq

# Load dataset
def load_food_data(path="IndianFoodDatasetXLS.xlsx"):
    try:
        df = pd.read_excel(path)
        df = df[['TranslatedRecipeName', 'TranslatedIngredients','TotalTimeInMins','Servings','Cuisine','Course','Diet']]
        df.dropna(subset=['TranslatedRecipeName', 'TranslatedIngredients'], inplace=True)
        df['TranslatedIngredients'] = df['TranslatedIngredients'].str.lower()
        return df
    except:
        return pd.DataFrame()

food_df = load_food_data()

# -------------------------------
# UNIFIED LLM FOR ALL TOOLS
# -------------------------------
from dotenv import load_dotenv
import os
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY","").strip()

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-8b-specdec",
    temperature=0.1
)

def ask_llm(question):
    response = llm.invoke([{"role":"user","content":question}])
    return response.content.strip()

# -------------------------------
# INTENT DETECTION
# -------------------------------
def detect_gym_trigger(text):
    return "true" in ask_llm(f"Does this describe a workout? Reply true/false.\n{text}").lower()

def detect_food_trigger(text):
    return "true" in ask_llm(f"Is this food logging? Reply true/false.\n{text}").lower()

def detect_graph_command(text):
    return "true" in ask_llm(f"Is user requesting a gym graph? Reply true/false.\n{text}").lower()

def detect_pie_command(text):
    return "true" in ask_llm(f"Is user asking for a food pie chart? Reply true/false.\n{text}").lower()

def detect_timer_command(text):
    return "true" in ask_llm(f"Does this set a timer? Reply true/false.\n{text}").lower()

def parse_timer_command(text):
    response = ask_llm(
        f"Extract duration and task as JSON: {{'duration':SECONDS,'task':'NAME'}}\n{text}"
    )
    try:
        parsed = eval(response)
        return parsed["duration"], parsed["task"]
    except:
        return None

# -------------------------------
# FOOD, CALORIES & RECIPES
# -------------------------------
def estimate_calories(ingredients):
    calorie_dict = {
        "rice": 130, "potato": 110, "paneer": 265, "chicken": 239, "egg": 78,
        "milk": 42, "ghee": 115, "oil": 120, "dal": 120, "bread": 80
    }
    return sum(cal for item, cal in calorie_dict.items() if item in ingredients)

def handle_recipe_query(text):
    instruction = f"""
Identify user intent:
Return JSON:
{{ 'intent': 'suggest_recipe' OR 'calorie_query',
  'course':'Lunch','diet':'Vegetarian','recipe_name':'...' }}
User text: {text}
"""
    reply = ask_llm(instruction)

    try:
        parsed = eval(reply)
        if parsed["intent"] == "suggest_recipe":
            course = parsed.get("course","Lunch")
            diet = parsed.get("diet","Vegetarian")
            return suggest_recipe(course, diet)
        else:
            return calorie_lookup(parsed.get("recipe_name",""))
    except:
        return None

def suggest_recipe(course="Lunch", diet="Vegetarian"):
    if food_df.empty:
        return "⚠️ No recipe data available."

    df = food_df[
        (food_df['Course'].str.contains(course, case=False)) &
        (food_df['Diet'].str.contains(diet, case=False))
    ]

    if df.empty:
        return f"No {diet} {course} recipes found."

    item = df.sample(1).iloc[0]
    calories = estimate_calories(item['TranslatedIngredients'])

    return f"""
🍽️ {item['TranslatedRecipeName']}
🔥 Calories: ~{calories}
🕒 Time: {item['TotalTimeInMins']} mins
Ingredients: {item['TranslatedIngredients']}
"""

def calorie_lookup(name):
    name = name.lower().strip()
    match = food_df[food_df['TranslatedRecipeName'].str.lower()==name]
    if match.empty:
        return f"❌ No recipe found for {name}"

    item = match.iloc[0]
    calories = estimate_calories(item['TranslatedIngredients'])
    return f"🔥 {name} has approx {calories} kcal."

# -------------------------------
# GYM & FOOD LOGGING
# -------------------------------
gym_sessions = []
food_log = []

def extract_date(text):
    return datetime.now().date()

def extract_duration(text):
    match = re.search(r'(\d+)\s*(min|hrs|hour)', text.lower())
    if not match:
        return 0
    value = int(match.group(1))
    return value * (60 if "hour" in match.group(2) else 1)

def log_gym_session(text, user_id="default"):
    duration = extract_duration(text)
    date = extract_date(text)
    gym_sessions.append({"user":user_id,"duration":duration,"date":date})
    return f"💪 Logged gym session: {duration} mins on {date}"

def log_food_entry(text, user_id="default"):
    food_log.append({"user":user_id,"note":text})
    return f"🍽️ Logged food: {text}"

# -------------------------------
# PLOTS
# -------------------------------
def plot_gym_sessions():
    if not gym_sessions:
        return
    df = pd.DataFrame(gym_sessions)
    plt.bar(df["date"], df["duration"])
    plt.title("Gym Duration")
    plt.show()

def plot_food_pie_chart():
    if not food_log:
        return
    categories = []
    for f in food_log:
        t = f["note"].lower()
        if "breakfast" in t:
            categories.append("Breakfast")
        elif "lunch" in t:
            categories.append("Lunch")
        elif "dinner" in t:
            categories.append("Dinner")
        else:
            categories.append("Other")
    count = Counter(categories)
    plt.pie(count.values(), labels=count.keys(), autopct="%1.1f%%")
    plt.show()

# -------------------------------
# FOOD SUMMARY
# -------------------------------
def summarize_food_logs():
    if not food_log:
        return "No food logs."

    cats = {"Breakfast":0,"Lunch":0,"Dinner":0,"Other":0}
    for f in food_log:
        t=f["note"].lower()
        if "breakfast" in t: cats["Breakfast"]+=1
        elif "lunch" in t: cats["Lunch"]+=1
        elif "dinner" in t: cats["Dinner"]+=1
        else: cats["Other"]+=1

    return f"""Food Summary:
Breakfast: {cats['Breakfast']}
Lunch: {cats['Lunch']}
Dinner: {cats['Dinner']}
Other: {cats['Other']}
"""
