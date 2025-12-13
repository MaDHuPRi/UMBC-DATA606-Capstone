# UMBC-DATA606-Capstone
# AI-Powered Personalized Meal Planner

The **AI-Powered Meal Planner** is a personalized recommendation system that generates daily meal plans based on a user’s **calorie requirements, dietary preferences,** and **nutritional goals**.  
It combines **unsupervised learning** to cluster foods into meaningful nutritional groups and uses a **local LLM (Llama 3 via Ollama)** to c

## File Structure

```
project/
├── app/
│   ├── meal_planner.py     # Application (Streamlit)
|   ├── requirement.txt            
│   └── README.md
├── data/
│   ├── meal_features_cleaned.csv  # Data for ML model      
│   ├── merged.csv                 # Final data for the app
│   ├── original.csv              # Original Processed data
│   └── usda_foods_20k.csv        # Original, unprocessed data
├── docs/
│   ├── Images/
│   ├── Final_PPT.pptx
│   ├── README.md
│   ├── Resume.md
│   ├── headshot.jpg
│   ├── proposal.md
│   └── report.md
├── notebooks/
│   ├── 01_Data.ipynb      # Data collection and cleaning
│   └── 02_EDA.ipynb     # Exploratory Data Analysis
│
└── README.md
