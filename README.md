# 💎 Prestige Appraisal — AI-Powered Jewelry Valuation System

An AI-powered pre-owned jewelry appraisal system built as an open-ended AI & ML lab task. Uses KNN regression to estimate resale value and Cohere's LLM to generate professional auction-house style provenance reports.

## 🚀 Features
- KNN Regression Model — estimates fair market value from jewelry specifications
- Cohere API Integration — generates professional buyer-facing provenance reports
- Gradio Web Interface — clean, interactive UI for jewelry appraisal
- Graceful Error Handling — fallback report if API is unavailable
- Secure API Key Management — using python-dotenv

## 📊 Model Performance
| Metric | Value |
|--------|-------|
| Algorithm | KNN (k=7, distance-weighted) |
| R² Score | 0.897 |
| MAE | ~$605 |

## ⚙️ Setup
1. Clone the repo
2. pip install gradio scikit-learn numpy pandas joblib cohere python-dotenv
3. cp env.example .env and add your Cohere API key
4. python app.py
5. Open http://127.0.0.1:7860

## 📁 Project Structure
- app.py — Gradio UI
- jewelry_model.py — KNN model
- cohere_report.py — Cohere API integration
- env.example — API key template

## 👩‍💻 Author
Aiman — 6th Semester Computer Engineering Technology
International Islamic University Islamabad
