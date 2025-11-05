import streamlit as st
import pandas as pd
import ollama
import re
import plotly.graph_objects as go

# ---------- Load dataset ----------
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/madhupriyapulletikurthi/Desktop/ai_meal_planner/merged.csv")
    return df

df = load_data()

# ---------- Streamlit UI ----------
st.title("🥗 AI Meal Planner + Macro Donut")

st.sidebar.header("User Preferences")
calorie_goal = st.sidebar.number_input("Daily Calorie Requirement", min_value=100, max_value=4000, value=1800)
meal_type = st.sidebar.selectbox(
    "Select Meal Type Preference",
    ["Balanced Medium-Carb Meals", "Light Snacks", "Protein-Rich Balanced Meals", "High-Calorie Protein Meals","Protein-Rich Light Foods"]
)

if st.sidebar.button("Generate Meal Plan"):
    st.write(f"### Generating meal plan for: {meal_type} (Target: {calorie_goal} kcal)")
    
    # ---------- Filter dataset ----------
    filtered = df[df["meal_type"] == meal_type]
    
    # Convert to text for the LLM
    meals_text = ""
    for _, row in filtered.head(5).iterrows():  # only use 5 examples for brevity
        meals_text += f"- {row['description']} ({row['calories']} kcal, P: {row['protein_g']}g, C: {row['carbs_g']}g, F: {row['fat_g']}g)\n"

    # ---------- Construct prompt ----------
    prompt = f"""
You are a nutritionist AI. You have access to a dataset of meals categorized as High Protein, Low Calorie, High Calorie, and Balanced.

Here are some sample {meal_type}:
{meals_text}

The user's daily calorie goal is {calorie_goal} kcal.

Using this as context, generate a one-day meal plan (breakfast, lunch, and dinner)
that reflects the user's preference for {meal_type}.

If the provided sample meals are not suitable for a specific time of day
(e.g., too heavy for breakfast or too light for dinner),
replace them with more context-appropriate options that still maintain a balanced nutrient profile.

Each meal should include:
- Meal Name
- Short Description
- Calories, Protein, Carbs, Fat
The meals should be displayed in a tabular format.
Finally, calculate and display:
- Total Calories and Protein for the day.
    """

    # ---------- Run locally via Ollama ----------
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    meal_plan = response["message"]["content"]

    # ---------- Display the AI output ----------
    st.markdown("### 🧠 AI-Generated Meal Plan")
    st.write(meal_plan)

    # ---------- Extract totals using regex ----------
    cal_match = re.search(r"Total Calories.*?(\d+)", meal_plan, re.IGNORECASE)
    protein_match = re.search(r"Total Protein.*?(\d+)", meal_plan, re.IGNORECASE)

    total_calories = int(cal_match.group(1)) if cal_match else calorie_goal
    total_protein = int(protein_match.group(1)) if protein_match else 80

    # Estimate carbs/fat if not available
    protein_g = total_protein
    carbs_g = round((total_calories - protein_g * 4) * 0.55 / 4, 1)  # assume ~55% carbs
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

    # ---------- Macro details table ----------
    st.markdown("### 📊 Macro Details")
    st.write(pd.DataFrame({
        "Macro": ["Protein (g)", "Carbs (g)", "Fat (g)"],
        "Grams": [protein_g, carbs_g, fat_g],
        "Calories": [cal_from_protein, cal_from_carbs, cal_from_fat]
    }).assign(Total=lambda d: d["Calories"].sum()))
