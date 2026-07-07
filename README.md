# 💼 Employee Salary Prediction System

## 📌 Overview

This project predicts whether an employee earns **more than $50K** or **less than or equal to $50K** per year using Machine Learning.

The model is trained on the **UCI Adult Income Dataset** and deployed as an interactive **Streamlit Web Application**.

---

## 🚀 Features

- Data Cleaning
- Data Preprocessing
- One-Hot Encoding
- Random Forest Classifier
- Model Evaluation
- Prediction Confidence
- Interactive Streamlit UI

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib

---

## 📂 Dataset

**Adult Income Dataset**

https://archive.ics.uci.edu/dataset/2/adult

---

## 📁 Project Structure

```
Salary-Prediction/

├── app/
│   └── app.py
│
├── data/
│   └── adult/
│       └── adult.data
│
├── model/
│   ├── salary_pipeline.pkl
│   └── metrics.json
│
├── train.py
├── requirements.txt
└── README.md
```

---

## ▶️ Train the Model

```bash
python train.py
```

---

## ▶️ Run the Application

```bash
streamlit run app/app.py
```

---

## 📊 Model

- Algorithm: Random Forest Classifier
- Accuracy: **~86%**

---

## 📷 Application

The user enters employee details such as:

- Age
- Education
- Workclass
- Occupation
- Hours Per Week
- Capital Gain
- Capital Loss

The application predicts whether the employee's income is:

- **> $50K**
- **≤ $50K**

and also displays the prediction confidence.

---

## 📚 Learning Outcomes

Through this project, I learned:

- Data Cleaning
- Feature Engineering
- Machine Learning Pipeline
- One-Hot Encoding
- Random Forest Classification
- Model Evaluation
- Streamlit Deployment

---

## 👨‍💻 Author

**VALLAMKONDA PEDDA MALLIKARJUNA RAO**

GitHub: https://github.com/mallikarjuna004