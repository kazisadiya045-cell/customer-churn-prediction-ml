
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PAGE SETTINGS

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Analyze customer data, compare ML models "
    "and predict possible customer churn."
)

# LOAD DATASET

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    data = data.dropna()
    data = data.drop_duplicates()

    return data


try:
    df = load_data()
except Exception as error:
    st.error(
        "Dataset load nahi hua. Internet connection "
        "aur dataset URL check karo."
    )
    st.stop()


# PREPARE DATA

raw_features = df.drop(
    columns=["customerID", "Churn"]
)

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X = pd.get_dummies(
    raw_features,
    drop_first=True
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# TRAIN THREE MODELS

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    results = []
    confusion_matrices = {}
    reports = {}
    trained_models = {}

    for name, model in models.items():

        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(
                y_test, predictions
            ),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "Macro-F1": f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
        })

        confusion_matrices[name] = confusion_matrix(
            y_test, predictions
        )

        reports[name] = classification_report(
            y_test,
            predictions,
            zero_division=0,
            output_dict=True
        )

        trained_models[name] = model

    results_df = pd.DataFrame(results)

    return (
        scaler,
        trained_models,
        results_df,
        confusion_matrices,
        reports
    )


(
    scaler,
    models,
    results_df,
    confusion_matrices,
    reports
) = train_models(
    X_train, X_test, y_train, y_test
)

# SIDEBAR NAVIGATION

page = st.sidebar.radio(
    "Choose a page",
    [
        "Dashboard",
        "Instant Prediction",
        "Model Evaluation",
        "Ask a Question",
        "Assessment Test Cases"
    ]
)


# PAGE 1: DASHBOARD

if page == "Dashboard":

    st.header("Customer Analytics Dashboard")

    total_customers = len(df)

    churned_customers = (
        df["Churn"] == "Yes"
    ).sum()

    retained_customers = (
        df["Churn"] == "No"
    ).sum()

    churn_percentage = (
        churned_customers / total_customers
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    col3.metric(
        "Retained Customers",
        f"{retained_customers:,}"
    )

    col4.metric(
        "Churn Rate",
        f"{churn_percentage:.1f}%"
    )

    st.subheader("Customer Churn Distribution")

    churn_counts = df["Churn"].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        churn_counts.index,
        churn_counts.values
    )

    ax.set_xlabel("Churn Status")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customers Who Left vs Stayed")

    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Contract")

        contract_data = pd.crosstab(
            df["Contract"],
            df["Churn"]
        )

        st.bar_chart(contract_data)

    with col2:
        st.subheader("Average Monthly Charges")

        charges_data = df.groupby(
            "Churn"
        )["MonthlyCharges"].mean()

        st.bar_chart(charges_data)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.caption(
        "Dataset source: IBM Telco Customer Churn. "
        "The dashboard shows the cleaned dataset."
    )

# PAGE 2: INSTANT PREDICTION

elif page == "Instant Prediction":

    st.header("Predict Customer Churn")

    st.write(
        "Customer details enter karo aur prediction "
        "ke liye button click karo."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

            partner = st.selectbox(
                "Has Partner?",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Has Dependents?",
                ["Yes", "No"]
            )

            tenure = st.slider(
                "Tenure (months)",
                0, 72, 12
            )

            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with col2:
            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=2000.0,
                value=70.0
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=100000.0,
                value=840.0
            )

        selected_model = st.selectbox(
            "Choose Prediction Model",
            list(models.keys())
        )

        submitted = st.form_submit_button(
            "🔮 Predict Churn"
        )

    if submitted:

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        # Encode the new customer using the same
        # categories as the original dataset.
        combined_data = pd.concat(
            [raw_features, customer],
            ignore_index=True
        )

        combined_encoded = pd.get_dummies(
            combined_data,
            drop_first=True
        )

        customer_encoded = combined_encoded.tail(1)

        customer_encoded = customer_encoded.reindex(
            columns=X.columns,
            fill_value=0
        )

        customer_scaled = scaler.transform(
            customer_encoded
        )

        model = models[selected_model]

        prediction = model.predict(
            customer_scaled
        )[0]

        probability = model.predict_proba(
            customer_scaled
        )[0][1]

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("Likely to Churn")
            else:
                st.success("Likely to Stay")

        with col2:
            st.metric(
                "Estimated Churn Probability",
                f"{probability * 100:.1f}%"
            )

        st.progress(float(probability))

        st.caption(
            "Probability is the model's estimate, not a guarantee. "
            "Predictions can be wrong."
        )

# PAGE 3: MODEL EVALUATION

elif page == "Model Evaluation":

    st.header("Model Performance Comparison")

    st.write(
        "All three models were evaluated using "
        "the same held-out test dataset."
    )

    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "Macro-F1": "{:.3f}"
        }),
        use_container_width=True
    )

    st.subheader("Compare Model Scores")

    chart_data = results_df.set_index("Model")

    st.bar_chart(chart_data)

    selected_model = st.selectbox(
        "View Confusion Matrix",
        list(models.keys())
    )

    matrix = confusion_matrices[selected_model]

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Stay", "Actual Churn"],
        columns=["Predicted Stay", "Predicted Churn"]
    )

    st.write("Confusion Matrix")
    st.dataframe(matrix_df, use_container_width=True)

    st.write("Classification Report")

    st.dataframe(
        pd.DataFrame(reports[selected_model]).round(3),
        use_container_width=True
    )

    st.download_button(
        "Download Model Comparison CSV",
        data=results_df.to_csv(index=False),
        file_name="model_comparison.csv",
        mime="text/csv"
    )

    st.caption(
        "Scores are calculated at runtime. The model with "
        "the highest score is not automatically best for "
        "every business objective."
    )

# PAGE 4: SIMPLE DOCUMENT QUESTION ANSWERING

elif page == "Ask a Question":

    st.header("Project Question Answering")

    st.write(
        "Ask a question about churn, machine learning "
        "or the evaluation metrics."
    )

    documents = [
        {
            "title": "Customer Churn",
            "source": DATA_URL,
            "text": (
                "Customer churn means a customer stops "
                "using a company's service. Businesses "
                "study churn to improve customer retention."
            )
        },
        {
            "title": "Machine Learning Models",
            "source": "Scikit-learn documentation",
            "text": (
                "Logistic Regression is a classification "
                "algorithm. Decision Tree uses decision rules. "
                "Random Forest combines multiple decision trees."
            )
        },
        {
            "title": "Evaluation Metrics",
            "source": "Scikit-learn metrics documentation",
            "text": (
                "Accuracy measures the fraction of correct "
                "predictions. Precision measures how many "
                "positive predictions are correct. Recall "
                "measures how many actual positive cases "
                "are found. F1 combines precision and recall."
            )
        }
    ]

    question = st.text_input(
        "Enter your question"
    )

    if st.button("Find Answer"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            texts = [
                document["text"]
                for document in documents
            ]

            vectorizer = TfidfVectorizer()

            document_vectors = vectorizer.fit_transform(
                texts
            )

            question_vector = vectorizer.transform(
                [question]
            )

            scores = cosine_similarity(
                question_vector,
                document_vectors
            )[0]

            best_index = scores.argmax()

            if scores[best_index] < 0.10:
                st.warning(
                    "Relevant information was not found "
                    "in the available documents."
                )
            else:
                st.subheader("Retrieved Information")

                st.write(
                    documents[best_index]["text"]
                )

                st.caption(
                    "Source: "
                    + documents[best_index]["title"]
                )

                st.write(
                    "Reference: "
                    + documents[best_index]["source"]
                )

                st.caption(
                    "This basic demo retrieves the closest "
                    "text. It is not a full generative AI system."
                )

# PAGE 5: ASSESSMENT TEST CASES

elif page == "Assessment Test Cases":

    st.header("Question Answering Test Plan")

    test_cases = pd.DataFrame([
        {
            "Test": 1,
            "Scenario": "Direct answer in one document",
            "Expected Result": "Return relevant text and source"
        },
        {
            "Test": 2,
            "Scenario": "Question needs multiple passages",
            "Expected Result": "Retrieve all required passages"
        },
        {
            "Test": 3,
            "Scenario": "Question uses synonyms",
            "Expected Result": "Find relevant information if available"
        },
        {
            "Test": 4,
            "Scenario": "Topic absent from documents",
            "Expected Result": "Say information was not found"
        },
        {
            "Test": 5,
            "Scenario": "Empty question",
            "Expected Result": "Ask the user to enter a question"
        },
        {
            "Test": 6,
            "Scenario": "Missing or very short document",
            "Expected Result": "Skip or flag unusable text"
        },
        {
            "Test": 7,
            "Scenario": "Irrelevant retrieved passages",
            "Expected Result": "Avoid presenting irrelevant text as reliable"
        },
        {
            "Test": 8,
            "Scenario": "Conflicting evidence",
            "Expected Result": "Show sources and flag the conflict"
        },
        {
            "Test": 9,
            "Scenario": "Duplicate documents",
            "Expected Result": "Detect or remove duplicates"
        },
        {
            "Test": 10,
            "Scenario": "Misleading instructions in a document",
            "Expected Result": "Treat document text as evidence, not instructions"
        }
    ])

    st.dataframe(
        test_cases,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "This table is a test plan, not proof that all ten "
        "cases have passed. Several advanced cases require "
        "additional implementation and actual testing."
    )