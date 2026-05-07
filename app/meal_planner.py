import streamlit as st
import pandas as pd
import re
import plotly.graph_objects as go
import requests
import json

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NourishAI · Meal Planner",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Theme config override: always light ── */
.stApp {
    background-color: #FAF7F2 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1C2B1E !important;
    border-right: none !important;
    min-width: 320px !important;
    width: 320px !important;
}
[data-testid="stSidebar"] * { color: #D9E8D5 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #A8C5A0 !important; font-size: 0.75rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; font-family: 'DM Mono', monospace !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #243328 !important;
    border: 1px solid #3a5040 !important;
    border-radius: 10px !important;
    color: #D9E8D5 !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] * { color: #D9E8D5 !important; }
[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] { display: none; }
[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: #5A8A5E !important;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #5A8A5E, #3D6641) !important;
    color: white !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.75rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    cursor: pointer !important;
    margin-top: 0.5rem !important;
    transition: opacity 0.2s !important;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

/* ── Main content text ── */
.main p, .main li, .main span:not([class]),
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { color: #2C2416 !important; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { color: #1C2B1E !important; }

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0.25rem;
    border-bottom: 2px solid #E8E0D0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #8A7A60 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.25rem !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #1C2B1E !important;
    border-bottom: 2px solid #5A8A5E !important;
    font-weight: 500 !important;
}

/* ── Cards ── */
.meal-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
    border: 1px solid #EDE5D8;
    box-shadow: 0 2px 16px rgba(28,43,30,0.05);
}
.meal-card h3, .meal-card p, .meal-card li { color: #2C2416 !important; }

.stat-chip {
    display: inline-block;
    background: #F0EBE0;
    color: #5A4A2A !important;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    margin: 0.2rem;
}
.stat-chip-green {
    background: #DDF0DD;
    color: #2D5A30 !important;
}

/* ── Hero ── */
.hero {
    padding: 2rem 0 1.5rem;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.5rem;
    font-weight: 700;
    color: #1C2B1E;
    line-height: 1.05;
    margin-bottom: 0.25rem;
}
.hero-title em {
    font-style: italic;
    color: #5A8A5E;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #8A7A60;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #E8E0D0; margin: 1.25rem 0; }

/* ── Generate button (main) ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #5A8A5E, #3D6641);
    color: white;
    border: none;
    border-radius: 999px;
    padding: 0.75rem 2.5rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85; transform: translateY(-1px); }

/* ── Info banner ── */
.info-banner {
    background: #1C2B1E;
    color: #D9E8D5;
    border-radius: 14px;
    padding: 1.1rem 1.5rem;
    margin-bottom: 1.25rem;
    font-size: 0.9rem;
    line-height: 1.6;
}
.info-banner * { color: #D9E8D5 !important; }

/* ── Warning banner ── */
.warn-banner {
    background: #FFF8EC;
    border: 1px solid #E8C97A;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #6B4F00 !important;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #5A8A5E !important; }
</style>
""", unsafe_allow_html=True)

# ── Streamlit theme config (force light) ──────────────────────────────────────
# Also set via .streamlit/config.toml

# ── Load dataset ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/merged.csv")
    return df

df = load_data()

MEAL_TYPES = [
    "Carb-Dense Balanced Meals — Ideal for Weight Gain",
    "Low-Calorie Light Foods — Ideal for Weight Loss",
    "Protein-Rich Balanced Meals — Ideal for Maintenance",
    "High-Protein Energy Meals — Ideal for Maintenance or Muscle Gain",
]
DIET_PREFS = ["No Restriction", "Vegetarian", "Vegan", "Pescatarian", "Halal", "No Beef"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.5rem 0.5rem 0.5rem;'>
        <div style='font-size:2rem; margin-bottom:0.4rem;'>🌿</div>
        <div style='font-family:"Cormorant Garamond",serif; font-size:1.5rem; font-weight:700; color:#D9E8D5; line-height:1.1;'>NourishAI</div>
        <div style='font-family:"DM Mono",monospace; font-size:0.62rem; color:#6A9A6E; letter-spacing:0.12em; text-transform:uppercase; margin-top:0.25rem;'>Intelligent Meal Planning</div>
    </div>
    <hr style='border-color:#2e4a32; margin:1rem 0;'>
    """, unsafe_allow_html=True)

    age_group = st.selectbox("Age Group", ["18–25", "26–35", "36–50", "51+"])
    calorie_goal = st.slider("Daily Calorie Goal (kcal)", 1200, 2500, 1800, 100)
    meal_type = st.selectbox("Goal / Diet Type", MEAL_TYPES, format_func=lambda x: x.split("—")[0].strip())
    diet_pref = st.selectbox("Diet Preference", DIET_PREFS)

    st.markdown("<hr style='border-color:#2e4a32; margin:1rem 0;'>", unsafe_allow_html=True)

    generate = st.button("✦ Generate My Plan")

    st.markdown("""
    <hr style='border-color:#2e4a32; margin:1.5rem 0 0.75rem;'>
    <div style='font-family:"DM Mono",monospace; font-size:0.6rem; color:#4a7050; text-align:center; line-height:1.8;'>
        Powered by Ollama · llama3<br>
        NourishAI © 2025
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Your personal<br><em>AI nutritionist</em></div>
    <div class="hero-sub">// Personalized · Data-Driven · Delicious</div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🥗  Meal Plan & Macros", "🍳  Recipes", "🛒  Grocery List"])

# ── Check Ollama availability ─────────────────────────────────────────────────
def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def ask_ollama(prompt: str) -> str:
    """Call Ollama llama3 locally."""
    try:
        import ollama
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        return f"⚠️ Ollama error: {e}"

# ── Generate plan ─────────────────────────────────────────────────────────────
with tab1:
    if not generate:
        st.markdown("""
        <div class="info-banner">
            🌿 <b>Configure your preferences</b> in the sidebar, then click <b>✦ Generate My Plan</b> to get a personalised AI-crafted meal plan with macro breakdown.
        </div>
        """, unsafe_allow_html=True)

        # Show dataset stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="meal-card"><div style="font-family:\'Cormorant Garamond\',serif; font-size:2.2rem; font-weight:700; color:#1C2B1E;">19,369</div><div style="font-family:\'DM Mono\',monospace; font-size:0.7rem; color:#8A7A60; text-transform:uppercase; letter-spacing:0.1em;">Food items in database</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="meal-card"><div style="font-family:\'Cormorant Garamond\',serif; font-size:2.2rem; font-weight:700; color:#1C2B1E;">4</div><div style="font-family:\'DM Mono\',monospace; font-size:0.7rem; color:#8A7A60; text-transform:uppercase; letter-spacing:0.1em;">Meal plan categories</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="meal-card"><div style="font-family:\'Cormorant Garamond\',serif; font-size:2.2rem; font-weight:700; color:#1C2B1E;">6</div><div style="font-family:\'DM Mono\',monospace; font-size:0.7rem; color:#8A7A60; text-transform:uppercase; letter-spacing:0.1em;">Diet preferences</div></div>', unsafe_allow_html=True)

    else:
        # Check Ollama
        ollama_ok = check_ollama()
        if not ollama_ok:
            st.markdown("""
            <div class="warn-banner">
                ⚠️ <b>Ollama is not running locally.</b> Start it with <code>ollama serve</code> and make sure <code>llama3</code> is pulled (<code>ollama pull llama3</code>). The app requires Ollama to generate meal plans.
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # Filter dataset for examples
        filtered = df[df["meal_type"] == meal_type]
        meals_text = ""
        for _, row in filtered.sample(min(8, len(filtered)), random_state=42).iterrows():
            meals_text += f"- {row['description']} ({row['calories']:.0f} kcal, P:{row['protein_g']}g, C:{row['carbs_g']}g, F:{row['fat_g']}g)\n"

        prompt = f"""You are a certified nutritionist creating a personalized meal plan.

Client profile:
- Age group: {age_group}
- Daily calorie goal: {calorie_goal} kcal
- Diet type: {meal_type}
- Diet preference: {diet_pref}

Diet preference rules:
- Halal: No pork or alcohol; Halal-certified meats only.
- No Beef: No beef or red meat.
- Vegetarian: No meat or seafood.
- Vegan: No animal products at all.
- Pescatarian: Seafood max 1-2 meals/day; other meals plant-based.

Design a 1-day meal plan with exactly 3 meals: Breakfast, Lunch, Dinner.

For each meal provide:
1. Meal name
2. Ingredients with quantities
3. Short description (1-2 sentences)
4. Macros: Calories, Protein (g), Carbs (g), Fat (g)

Example foods from our database for inspiration:
{meals_text}

End your response with:
Total Calories: [number]
Total Protein: [number]g
Total Carbs: [number]g
Total Fat: [number]g
"""

        with st.spinner("Crafting your personalised meal plan…"):
            meal_plan = ask_ollama(prompt)

        # ── Display plan ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:1rem; margin-bottom:1rem;'>
            <div>
                <div style='font-family:"Cormorant Garamond",serif; font-size:1.8rem; font-weight:700; color:#1C2B1E;'>Your Meal Plan</div>
                <div style='font-family:"DM Mono",monospace; font-size:0.7rem; color:#8A7A60; text-transform:uppercase; letter-spacing:0.1em;'>{age_group} · {calorie_goal} kcal · {diet_pref}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="meal-card">', unsafe_allow_html=True)
        st.markdown(meal_plan)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Extract macros ────────────────────────────────────────────────────
        cal_match     = re.search(r"Total Calories[:\s]+(\d+)", meal_plan, re.IGNORECASE)
        protein_match = re.search(r"Total Protein[:\s]+(\d+)", meal_plan, re.IGNORECASE)
        carbs_match   = re.search(r"Total Carbs[:\s]+(\d+)", meal_plan, re.IGNORECASE)
        fat_match     = re.search(r"Total Fat[:\s]+(\d+)", meal_plan, re.IGNORECASE)

        total_calories = int(cal_match.group(1))     if cal_match     else calorie_goal
        protein_g      = int(protein_match.group(1)) if protein_match else 80
        carbs_g        = int(carbs_match.group(1))   if carbs_match   else round((total_calories - protein_g * 4) * 0.55 / 4)
        fat_g          = int(fat_match.group(1))     if fat_match     else round((total_calories - protein_g * 4 - carbs_g * 4) / 9)

        cal_from_protein = protein_g * 4
        cal_from_carbs   = carbs_g * 4
        cal_from_fat     = fat_g * 9

        # ── Macro donut ───────────────────────────────────────────────────────
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'Cormorant Garamond\',serif; font-size:1.5rem; font-weight:700; color:#1C2B1E; margin-bottom:0.75rem;">Macro Breakdown</div>', unsafe_allow_html=True)

        col_chart, col_table = st.columns([1, 1])
        with col_chart:
            fig = go.Figure(go.Pie(
                labels=["Protein", "Carbs", "Fat"],
                values=[protein_g, carbs_g, fat_g],
                hole=0.62,
                hoverinfo="label+percent+text",
                textinfo="percent",
                hovertext=[
                    f"Protein: {protein_g}g · {cal_from_protein} kcal",
                    f"Carbs: {carbs_g}g · {cal_from_carbs} kcal",
                    f"Fat: {fat_g}g · {cal_from_fat} kcal",
                ],
                marker=dict(
                    colors=["#5A8A5E", "#D4A853", "#C4614A"],
                    line=dict(color="#FAF7F2", width=3)
                )
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(font=dict(family="DM Mono", size=11, color="#2C2416")),
                margin=dict(t=10, b=10, l=10, r=10),
                annotations=[dict(
                    text=f"<b>{total_calories}</b><br><span style='font-size:11px;color:#8A7A60'>kcal</span>",
                    x=0.5, y=0.5, font=dict(size=20, color="#1C2B1E"), showarrow=False
                )]
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown("<br><br>", unsafe_allow_html=True)
            for macro, grams, kcal, color in [
                ("Protein", protein_g, cal_from_protein, "#5A8A5E"),
                ("Carbohydrates", carbs_g, cal_from_carbs, "#D4A853"),
                ("Fat", fat_g, cal_from_fat, "#C4614A"),
            ]:
                pct = round(kcal / total_calories * 100) if total_calories else 0
                st.markdown(f"""
                <div style='margin-bottom:1rem;'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:0.3rem;'>
                        <span style='font-family:"DM Sans",sans-serif; font-weight:500; color:#2C2416;'>{macro}</span>
                        <span style='font-family:"DM Mono",monospace; font-size:0.8rem; color:#8A7A60;'>{grams}g · {kcal} kcal</span>
                    </div>
                    <div style='background:#EDE5D8; border-radius:999px; height:8px; overflow:hidden;'>
                        <div style='height:100%; width:{pct}%; background:{color}; border-radius:999px;'></div>
                    </div>
                    <div style='font-family:"DM Mono",monospace; font-size:0.68rem; color:#A09080; margin-top:0.2rem;'>{pct}% of total calories</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background:#1C2B1E; border-radius:12px; padding:0.85rem 1.25rem; text-align:center; margin-top:0.5rem;'>
                <span style='font-family:"Cormorant Garamond",serif; font-size:1.4rem; font-weight:700; color:#D9E8D5;'>{total_calories} kcal</span>
                <span style='font-family:"DM Mono",monospace; font-size:0.7rem; color:#6A9A6E; display:block; margin-top:0.1rem;'>TOTAL DAILY CALORIES</span>
            </div>
            """, unsafe_allow_html=True)

        # Store plan in session for other tabs
        st.session_state["meal_plan"] = meal_plan
        st.session_state["plan_ready"] = True

with tab2:
    if not st.session_state.get("plan_ready"):
        st.markdown('<div class="info-banner">🍳 Generate a meal plan first from the <b>Meal Plan & Macros</b> tab.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-family:\'Cormorant Garamond\',serif; font-size:1.8rem; font-weight:700; color:#1C2B1E; margin-bottom:0.75rem;">Step-by-Step Recipes</div>', unsafe_allow_html=True)
        if st.button("🍳 Generate Recipes", key="recipe_btn"):
            recipe_prompt = f"Write simple, clear step-by-step cooking instructions for each meal in this plan. Format each recipe with: Meal name as heading, Prep time, Cook time, numbered steps.\n\n{st.session_state['meal_plan']}"
            with st.spinner("Writing your recipes…"):
                recipes = ask_ollama(recipe_prompt)
            st.session_state["recipes"] = recipes

        if st.session_state.get("recipes"):
            st.markdown('<div class="meal-card">', unsafe_allow_html=True)
            st.markdown(st.session_state["recipes"])
            st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    if not st.session_state.get("plan_ready"):
        st.markdown('<div class="info-banner">🛒 Generate a meal plan first from the <b>Meal Plan & Macros</b> tab.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-family:\'Cormorant Garamond\',serif; font-size:1.8rem; font-weight:700; color:#1C2B1E; margin-bottom:0.75rem;">Your Grocery List</div>', unsafe_allow_html=True)
        if st.button("🛒 Generate Grocery List", key="grocery_btn"):
            grocery_prompt = f"From this meal plan, create a neatly organised grocery list grouped by category: Proteins, Grains & Starches, Vegetables, Fruits, Dairy & Eggs, Pantry & Spices, Others. Include estimated quantities.\n\n{st.session_state['meal_plan']}"
            with st.spinner("Building your grocery list…"):
                groceries = ask_ollama(grocery_prompt)
            st.session_state["groceries"] = groceries

        if st.session_state.get("groceries"):
            st.markdown('<div class="meal-card">', unsafe_allow_html=True)
            st.markdown(st.session_state["groceries"])
            st.markdown('</div>', unsafe_allow_html=True)
