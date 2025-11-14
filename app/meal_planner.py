import streamlit as st
import pandas as pd
import ollama
import re
import plotly.graph_objects as go

# ---------- Sidebar Width & Styling Fix ----------
st.markdown("""
    <style>
        /* Expand sidebar width */
        [data-testid="stSidebar"] {
            width: 400px !important;
            min-width: 500px !important;
        }

        /* Allow selectbox text to wrap nicely */
        div[data-baseweb="select"] > div {
            white-space: normal !important;
        }

        /* Adjust font for readability */
        div[data-baseweb="select"] {
            font-size: 15px !important;
        }

        /* Optional: Add padding and clean header look */
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Load dataset ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/merged.csv")
    return df

df = load_data()

# ---------- Streamlit UI ----------
st.title("🥗 AI Meal Planner Assistant")

st.sidebar.header("🍽️ Meal Plan Settings")

age_group = st.sidebar.selectbox(
    "Select your age group:",
    ["18–25", "26–35", "36–50", "51+"]
)

calorie_goal = st.sidebar.slider(
    "Select your daily calorie goal (kcal):",
    min_value=1200,
    max_value=2500,
    value=1800,
    step=100
)

meal_type = st.sidebar.selectbox(
    "Diet Type",
    [
        "Carb-Dense Balanced Meals — Ideal for Weight Gain",
        "Low-Calorie Light Foods — Ideal for Weight Loss",
        "Protein-Rich Balanced Meals — Ideal for Maintenance",
        "High-Protein Energy Meals — Ideal for Maintenance or Muscle Gain"
    ]
)

diet_pref = st.sidebar.selectbox(
    "Diet Preference",
    [
        "No Restriction",
        "Vegetarian",
        "Vegan",
        "Pescatarian",
        "Halal",
        "No Beef"
    ]
)

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🥗 Meal Plan & Macros", "🍳 Recipes", "🛒 Grocery List"])

with tab1:
    if st.button("Generate Meal Plan"):
        st.write(f"### Generating meal plan for: {meal_type} (Target: {calorie_goal} kcal)")

        # ---------- Filter dataset ----------
        filtered = df[df["meal_type"] == meal_type]

        # Convert to text for the LLM
        meals_text = ""
        for _, row in filtered.head(5).iterrows():
            meals_text += f"- {row['description']} ({row['calories']} kcal, P: {row['protein_g']}g, C: {row['carbs_g']}g, F: {row['fat_g']}g)\n"

        # ---------- Construct prompt ----------
        prompt = f"""
You are a certified nutritionist creating a personalized meal plan for a {age_group} individual
who has a daily calorie goal of {calorie_goal} kcal and prefers a {meal_type} diet.

This person follows a {diet_pref} eating preference — ensure all meals comply with it.

Guidelines for diet preferences:
- **Halal:** Avoid pork and alcohol; use Halal-certified meats only.
- **No Beef:** Avoid beef and red meat entirely.
- **Vegetarian:** Exclude all meat and seafood.
- **Vegan:** Exclude all animal products (meat, dairy, eggs, etc.).
- **Pescatarian:** Include seafood occasionally (1–2 meals per day max), but other meals can be vegetarian or plant-based.

Design a detailed 1-day plan with exactly 3 meals (Breakfast, Lunch, Dinner).

Each meal should include:
1. Meal Name
2. Ingredients with realistic quantities (e.g., 100g tofu, 1 cup quinoa)
3. A short description
4. Calories, Protein (g), Carbs (g), and Fat (g)
In a clear table format.
Use these example meals as inspiration:
{meals_text}

Format the output clearly and end with:
- Total Calories
- Total Protein
"""

        # ---------- Run via Ollama ----------
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        meal_plan = response["message"]["content"]

        st.markdown("### 🧠 AI-Generated Meal Plan")
        st.write(meal_plan)

        # ---------- Extract totals using regex ----------
        cal_match = re.search(r"Total Calories.*?(\d+)", meal_plan, re.IGNORECASE)
        protein_match = re.search(r"Total Protein.*?(\d+)", meal_plan, re.IGNORECASE)

        total_calories = int(cal_match.group(1)) if cal_match else calorie_goal
        total_protein = int(protein_match.group(1)) if protein_match else 80

        # Estimate carbs/fat if not provided
        protein_g = total_protein
        carbs_g = round((total_calories - protein_g * 4) * 0.55 / 4, 1)
        fat_g = round((total_calories - protein_g * 4 - carbs_g * 4) / 9, 1)

        # ---------- Plotly Donut ----------
        st.markdown("### 🍩 Macro Breakdown")
        labels = ["Protein (g)", "Carbs (g)", "Fat (g)"]
        values = [protein_g, carbs_g, fat_g]

        cal_from_protein = protein_g * 4
        cal_from_carbs = carbs_g * 4
        cal_from_fat = fat_g * 9

        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                hoverinfo="label+percent+text",
                textinfo="percent",
                hovertext=[
                    f"Protein: {protein_g:.1f}g ({cal_from_protein:.0f} kcal)",
                    f"Carbs: {carbs_g:.1f}g ({cal_from_carbs:.0f} kcal)",
                    f"Fat: {fat_g:.1f}g ({cal_from_fat:.0f} kcal)"
                ],
                marker=dict(line=dict(color="white", width=2))
            )
        )

        fig.update_layout(
            showlegend=True,
            margin=dict(t=20, b=20, l=20, r=20),
            annotations=[
                dict(
                    text=f"<b>{int(total_calories)} kcal</b><br><span style='font-size:12px'>Protein {protein_g:.0f}g</span>",
                    x=0.5, y=0.5, font_size=16, showarrow=False
                )
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------- Macro Table ----------
        st.markdown("### 📊 Macro Details")
        st.write(pd.DataFrame({
            "Macro": ["Protein (g)", "Carbs (g)", "Fat (g)"],
            "Grams": [protein_g, carbs_g, fat_g],
            "Calories": [cal_from_protein, cal_from_carbs, cal_from_fat]
        }).assign(Total=lambda d: d["Calories"].sum()))

        # ---------- Generate Recipes ----------
        with tab2:
            st.markdown("### 🍳 Recipes")
            recipe_prompt = f"Write simple, clear cooking instructions for each meal in this plan:\n\n{meal_plan}"
            recipe_resp = ollama.chat(model="llama3", messages=[{"role": "user", "content": recipe_prompt}])
            st.write(recipe_resp["message"]["content"])

        # ---------- Generate Grocery List ----------
        with tab3:
            st.markdown("### 🛒 Grocery List")
            grocery_prompt = f"From this meal plan, create a grocery list grouped by category (Proteins, Grains, Vegetables, Fruits, Dairy, Others):\n\n{meal_plan}"
            grocery_resp = ollama.chat(model="llama3", messages=[{"role": "user", "content": grocery_prompt}])
            st.write(grocery_resp["message"]["content"])
