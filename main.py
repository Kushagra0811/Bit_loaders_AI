from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List, Dict
import json
import re
import pandas as pd
import google.generativeai as genai

app = FastAPI()

# Request Model for Subscriptions
class Subscription(BaseModel):
    provider: str
    category: str
    amount: float
    billing_cycle: str
    renewal_date: str
    auto_renewal: bool
    trial: bool
    shared_with: str  # Users sharing the subscription (semicolon-separated)

# Calculate Monthly Cost
def calculate_monthly_cost(subs: List[Dict]):
    total_cost = 0
    for sub in subs:
        amount = float(sub["amount"])
        cycle = sub["billing_cycle"].lower()

        if cycle == "monthly":
            total_cost += amount
        elif cycle == "yearly":
            total_cost += amount / 12

    return round(total_cost, 2)

# Calculate Category-wise Spending
def category_wise_spending(subs: List[Dict]):
    df = pd.DataFrame(subs)

    # Convert 'shared_with' to number of users
    df["shared_with"] = df["shared_with"].apply(lambda x: len(x.split(";")) if isinstance(x, str) else 1)

    # Ensure amount is float
    df["amount"] = df["amount"].astype(float)

    # Calculate per-user cost
    df["per_user_cost"] = df["amount"] / df["shared_with"]

    # Adjust cost for yearly subscriptions
    df["monthly_cost"] = df["per_user_cost"].where(df["billing_cycle"] == "MONTHLY", df["per_user_cost"] / 12)

    # Group by category and sum up costs
    category_spending = df.groupby("category")["monthly_cost"].sum().round(2).to_dict()

    return category_spending

# AI Recommendation Function
def generate_ai_recommendations(subs: List[Dict], api_key: str):
    """Generates smart spending insights for subscription management in JSON format."""

    genai.configure(api_key=api_key)

    subs_text = "\n".join([
        f"{sub['provider']} - {sub['amount']} INR/{sub['billing_cycle']} (Shared with {sub.get('shared_with', 'N/A')} users)"
        for sub in subs
    ])

    prompt = f"""
    **Your estimated monthly subscription cost is {calculate_monthly_cost(subs)} INR.**
    You are an AI financial assistant helping users optimize their subscription spending.
    Given the following subscription details:

    {subs_text}

    Provide cost-saving insights such as:
    - Switching to better plans (e.g., family plans instead of multiple individual ones)
    - Identifying unused subscriptions for cancellation
    - Detecting duplicate services (e.g., multiple music streaming platforms)
    - Suggesting cost-effective bundling options

    Your response **must be in valid JSON format** and wrapped in triple backticks (```json):

    ```json
    {{
        "monthly_cost": <Total calculated monthly cost>,
        "breakdown": [
            {{
                "provider": "<Service Name>",
                "amount": <Amount>,
                "shared_with": <Number of users>,
                "insights": "<Specific recommendation>"
            }}
        ],
        "overall_recommendations": [
            "<Recommendation 1>",
            "<Recommendation 2>"
        ]
    }}
    ```
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    match = re.search(r"```json\n(.*?)\n```", response.text, re.DOTALL)
    if match:
        json_text = match.group(1)
    else:
        json_text = response.text

    try:
        recommendations_json = json.loads(json_text)
    except json.JSONDecodeError:
        recommendations_json = {
            "error": "Invalid JSON response from AI"
        }

    return recommendations_json

# FastAPI Routes
@app.get("/")
def home():
    return {"message": "FastAPI is running on Vercel!"}

@app.post("/generate-recommendations/")
def generate_recommendations(subs: List[Subscription]):
    subs_dict = [sub.dict() for sub in subs]
    
    # Generate AI recommendations
    recommendations = generate_ai_recommendations(subs_dict, "AIzaSyB0dqQ0YjJzq7cdY_NAHagisjZ1AMQRdjU")

    # Get category-wise spending
    category_spending = category_wise_spending(subs_dict)

    return {
        "ai_recommendations": recommendations,
        "category_wise_spending": category_spending
    }
