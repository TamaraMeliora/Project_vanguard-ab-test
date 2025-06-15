# Project Vanguard A/B Testing  

## 📌 Introduction  
Vanguard launched an A/B test to assess the effectiveness of a new user interface (UI) and in-context prompts designed to improve the online client experience.  

**Business Question:**  
Did the new UI lead to higher user completion rates?  

To answer this, we analyzed user behavior through client interaction data and conducted statistical testing to validate our findings.  

---

## 📊 Data Overview  

We used three databases:

- **Client Profiles**: `client_id`, age, tenure, gender  
- **Digital Footprints**: user actions — `visit_id`, `visitor_id`, `process_step`  
- **Experiment Roster**: assignment to Control or Test groups  

Data was joined on `client_id` and aggregated by `visit_id`. Duplicates were removed to ensure data quality.  

---

## 🔍 Exploratory Data Analysis (EDA)  

| Variable | Mean  | Median | Mode  |
|----------|-------|--------|-------|
| Age      | 46.4  | 47.0   | 58.5  |
| Tenure   | 12.0  | 11.0   | 6.0   |

---

## 📈 Key Performance Metrics  

| Metric                         | Value               |
|--------------------------------|---------------------|
| Total Visits                  | 158,095             |
| Visitors Reached Confirmation | 89,826              |
| Visitors Did Not Confirm      | 68,269              |
| **Completion Rate (Overall)** | **56.8%**           |
| Completion - Control Group    | 49.8%               |
| Completion - Test Group       | 58.5%               |
| Visits with Errors            | 40,469              |
| Visit-level Error Rate        | 25.6%               |
| Error Rate - Control          | 20.8%               |
| Error Rate - Test             | 27.2%               |

---

## 📐 Hypothesis Testing

### 1. **Completion Rate — Test vs Control**
- **H₀**: No difference in completion rates  
- **H₁**: Difference in completion rates exists  
- **Test**: Two-proportion Z-test  
- **Result**:  
  - Z = 19.516  
  - p-value ≈ 3.96 × 10⁻⁸⁵ ✅  
  - **Conclusion**: Statistically significant improvement in the Test group

---

### 2. **Cost-Effectiveness Threshold (≥ 5% uplift)**
- **H₀**: Uplift < 5%  
- **H₁**: Uplift ≥ 5%  
- **Test**: One-sample Z-test  
- **Result**:  
  - Z = 8.2613  
  - p-value ≈ 7.20 × 10⁻¹⁷ ✅  
  - **Conclusion**: The observed uplift exceeds the 5% threshold — business value confirmed.

---

### 3. **Average Age — Control vs Test**
- **H₀**: No difference in average age  
- **H₁**: Difference exists  
- **Test**: Two-sided T-test  
- **Result**:  
  - T = 7.83  
  - p-value ≈ 4.77 × 10⁻¹⁵ ✅  
  - **Cohen’s d**: 0.028 (negligible effect)  
  - **Conclusion**: Statistically significant, but practically irrelevant difference in age

---

## 📊 Tableau Dashboard

https://public.tableau.com/app/profile/mara.meli/viz/ProjectVanguard/Dashboard1

---

## 🧠 Team Collaboration

- Collaborative decision-making and division of analytical tasks  
- Cross-checked statistical methods and results  
- Shared data cleaning and visualization strategies  
- Aligned on insights and business implications

---

## 🚧 Challenges & Learnings

- Working with large datasets and merged tables required thorough preprocessing  
- Tableau visualizations required significant preparation and adjustment  
- Large sample size can produce statistically significant results with small practical effects — highlighted the need for practical significance analysis

---

## ✅ Final Conclusion

- The **Test group’s completion rate** was significantly and **practically higher** than the Control’s.  
- The **observed uplift exceeded the 5% business threshold**, making the new UI cost-effective.  
- The new interface demonstrably **improves user engagement**.  

**📢 Recommendation:**  
Roll out the new UI across the full platform — the data strongly supports its effectiveness.

