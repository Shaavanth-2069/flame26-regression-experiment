
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Student Marks Prediction")
st.write(
    "A Machine Learning experiment using Linear Regression "
    "to predict exam marks from study hours."
)

# Example dataset created for this demonstration.
# These are illustrative values, not real student records.
data = {
    "Study Hours": [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6,
                    6.5, 7, 7.5, 8, 9, 1.5, 3.2, 4.8, 6.8, 8.5],
    "Marks": [18, 25, 30, 35, 38, 45, 50, 54, 58, 65,
              68, 74, 78, 85, 92, 22, 37, 52, 71, 89]
}

df = pd.DataFrame(data)

# Separate input feature and target.
X = df[["Study Hours"]]
y = df["Marks"]

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train the regression model.
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set.
test_predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

st.divider()

st.subheader("🔮 Predict Your Marks")

study_hours = st.slider(
    "How many hours do you study per day?",
    min_value=0.0,
    max_value=12.0,
    value=5.0,
    step=0.5
)

predicted_marks = model.predict(
    pd.DataFrame({"Study Hours": [study_hours]})
)[0]

predicted_marks = max(0, min(100, predicted_marks))

st.metric(
    label="Illustrative predicted marks",
    value=f"{predicted_marks:.1f} / 100"
)

st.caption(
    "This is a demonstration model trained on a small, "
    "illustrative dataset. It is not a reliable prediction "
    "of any individual student's actual exam performance."
)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("📊 Dataset")
    st.dataframe(df, use_container_width=True)
    st.write(f"**Number of sample records:** {len(df)}")

with right:
    st.subheader("📈 Regression Graph")

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(
        df["Study Hours"],
        df["Marks"],
        label="Example observations"
    )

    graph_hours = pd.DataFrame({
        "Study Hours": list(range(0, 13))
    })

    graph_marks = model.predict(graph_hours)

    ax.plot(
        graph_hours["Study Hours"],
        graph_marks,
        label="Regression line"
    )

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Marks")
    ax.set_title("Study Hours vs Exam Marks")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

st.divider()

st.subheader("🧪 Model Evaluation")

metric1, metric2 = st.columns(2)

metric1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
metric2.metric("Test R² score", f"{r2:.3f}")

st.write("**What do these metrics mean?**")
st.write(
    "- MAE measures the average absolute prediction error "
    "in marks on the held-out test data."
)
st.write(
    "- R² describes how well the model fits the test data. "
    "With a small dataset, this score can be unstable or negative."
)

st.divider()

st.subheader("🧠 How the Experiment Works")

st.markdown("""
1. **Data collection:** Load illustrative study-hour and marks data.
2. **Preprocessing:** Separate the input feature and target.
3. **Train-test split:** Reserve some records for evaluation.
4. **Model training:** Fit a Linear Regression model.
5. **Prediction:** Estimate marks for the selected study hours.
6. **Evaluation:** Calculate MAE and R² on the held-out records.
7. **Visualization:** Plot the sample observations and regression line.
""")

st.info(
    "Educational demonstration only. Study time is not the only "
    "factor affecting academic performance."
)
