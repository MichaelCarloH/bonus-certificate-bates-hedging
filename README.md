# Credit Risk Modeling for Startups (May 2025)

## Objective  
Improve credit risk predictions for funded startups using enriched external data and machine learning, with a focus on interpretability and practical deployment.

## Project Summary  
Startups often suffer high default rates, and traditional credit scoring lacks timely, rich financial data. Our solution integrates Orbis firm-level data and applies **XGBoost** to predict defaults, achieving an **AUC of 0.84**. We focus on identifying high-risk firms and explain predictions using **SHAP values**. The output feeds into an interactive dashboard that enables informed decisions by credit risk managers.

---

## Data Pipeline  

### Enrichment & Matching  
- **Source:** Internal startup data + Orbis database (via Matching Research Tool)  
- **Region:** EU firms only (focus + lower compute)  
- **Match Rate:** ~60%  
- **Final Dataset:** ~5,500 companies, 160+ features  

### Preprocessing  
- **Missing Values:** Median imputation  
- **NACE Codes:** Normalized  
- **Target Variable:** Binary (Active vs Inactive)  
- **Feature Engineering:** Financial ratios, interactions, encodings  

---

## Modeling  

### Algorithm  
- **Model:** XGBoost  
- **Loss Function:** Cost-sensitive (penalizes missed defaults)  
- **Interpretability:** SHAP (local + global feature explanations)

### Metrics  
- **ROC AUC:** 0.845  
- **Precision:** 0.818  
- **MAE:** 0.298  
- **TSS (Optimal threshold = 0.3601):** 0.5445  
- **Calibration:** Strong alignment between predicted and observed default rates  

---

## Dashboard: RAAD (Risk Assessment from Augmented Data)  

- **Displays:**  
  - Credit score = \((1 - P_0) \times 100\)  
  - Key Orbis financials  
  - AI Assistant summary via LLM (ChatGPT 4.1 with browsing)  

---

## Future Development  

### Next Steps  
- CI/CD pipeline for retraining and updates  
- Integrate internal data (ERP, payment logs)  
- Deploy RAAD Dashboard on **AWS/Azure**  
- Extend to global firms and adapt to local accounting standards  

---

## Tech Stack  
- **Modeling:** Python, XGBoost, SHAP  
- **Data:** Orbis, internal credit data  
- **Frontend:** Streamlit / Dash (for dashboard)  
- **LLM Assistant:** OpenAI ChatGPT 4.1 (w/ web search)  
- **Deployment:** Docker, Azure (planned)

---

🚀 **Proof of Concept complete – contributions, ideas, and extensions welcome!**
