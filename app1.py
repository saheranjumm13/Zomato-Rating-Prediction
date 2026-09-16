import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Zomato ML Analytics",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f7f7;
}

.block-container {
    padding-top: 1.5rem;
}

h1 {
    font-weight: 700;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
    text-align: center;
}

.metric-title {
    font-size: 15px;
    color: #666;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
}

.prediction-box {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    background-color: #ffffff;
    box-shadow: 0px 3px 15px rgba(0,0,0,0.10);
}

.prediction-value {
    font-size: 42px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv("zomato_cleaned.csv")

    return data


@st.cache_resource
def load_model():

    model = joblib.load("zomato_rating_model.pkl")

    return model


@st.cache_resource
def load_features():

    features = joblib.load("zomato_features.pkl")

    return features


try:

    df = load_data()
    model = load_model()
    model_features = load_features()

except FileNotFoundError as e:

    st.error(
        "Required file not found. Make sure these files are in the same folder as app.py:"
    )

    st.code("""
zomato_cleaned.csv
zomato_rating_model.pkl
zomato_features.pkl
""")

    st.stop()


# ============================================================
# BASIC CLEANING
# ============================================================

df = df.copy()

if "rate" in df.columns:

    df["rate"] = pd.to_numeric(
        df["rate"],
        errors="coerce"
    )

if "votes" in df.columns:

    df["votes"] = pd.to_numeric(
        df["votes"],
        errors="coerce"
    ).fillna(0)

if "approx_cost(for two people)" in df.columns:

    df["approx_cost(for two people)"] = (
        df["approx_cost(for two people)"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    df["approx_cost(for two people)"] = pd.to_numeric(
        df["approx_cost(for two people)"],
        errors="coerce"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🍽️ Zomato Analytics")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Restaurant Analysis",
        "🤖 ML Model",
        "🔮 Rating Prediction"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Zomato Restaurant Analytics

    Machine Learning based
    Restaurant Rating Prediction
    """
)


# ============================================================
# HELPER FUNCTION FOR MODEL INPUT
# ============================================================

def prepare_model_input(
    location,
    rest_type,
    cuisines,
    online_order,
    book_table,
    votes,
    cost,
    listed_type,
    listed_city
):

    row = pd.DataFrame([{

        "online_order": online_order,
        "book_table": book_table,
        "votes": votes,
        "location": location,
        "rest_type": rest_type,
        "cuisines": cuisines,
        "approx_cost(for two people)": cost,
        "listed_in(type)": listed_type,
        "listed_in(city)": listed_city

    }])

    # Numeric conversion if model expects numeric Yes/No
    if "online_order" in model_features:

        row["online_order"] = row["online_order"].map({
            "Yes": 1,
            "No": 0
        })

    if "book_table" in model_features:

        row["book_table"] = row["book_table"].map({
            "Yes": 1,
            "No": 0
        })

    # Feature engineering if used during training

    row["log_votes"] = np.log1p(votes)

    row["cuisine_count"] = len(
        str(cuisines).split(",")
    )

    row["rest_type_count"] = len(
        str(rest_type).split(",")
    )

    # Determine categorical columns
    categorical_columns = []

    for col in [
        "location",
        "rest_type",
        "cuisines",
        "listed_in(type)",
        "listed_in(city)"
    ]:

        if col in row.columns:

            if col in model_features or any(
                str(col) + "_" in str(feature)
                for feature in model_features
            ):

                categorical_columns.append(col)

    # One-hot encoding
    if len(categorical_columns) > 0:

        row = pd.get_dummies(
            row,
            columns=categorical_columns,
            drop_first=True
        )

    # Convert bool to int
    bool_columns = row.select_dtypes(
        include=["bool"]
    ).columns

    for col in bool_columns:

        row[col] = row[col].astype(int)

    # Make sure exact training columns exist
    row = row.reindex(
        columns=model_features,
        fill_value=0
    )

    return row


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "🏠 Dashboard":

    st.title("🍽️ Zomato Restaurant Analytics Dashboard")

    st.markdown(
        "### Explore restaurant trends, ratings, popularity and costs"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.sidebar.subheader("🔎 Dashboard Filters")

    locations = sorted(
        df["location"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_location = st.sidebar.selectbox(
        "Location",
        ["All"] + locations
    )

    order_options = ["All"]

    if "online_order" in df.columns:

        order_options += sorted(
            df["online_order"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    selected_order = st.sidebar.selectbox(
        "Online Order",
        order_options
    )

    table_options = ["All"]

    if "book_table" in df.columns:

        table_options += sorted(
            df["book_table"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    selected_table = st.sidebar.selectbox(
        "Table Booking",
        table_options
    )

    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------

    filtered_df = df.copy()

    if selected_location != "All":

        filtered_df = filtered_df[
            filtered_df["location"].astype(str)
            == selected_location
        ]

    if selected_order != "All":

        filtered_df = filtered_df[
            filtered_df["online_order"].astype(str)
            == selected_order
        ]

    if selected_table != "All":

        filtered_df = filtered_df[
            filtered_df["book_table"].astype(str)
            == selected_table
        ]

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    total_restaurants = len(filtered_df)

    avg_rating = filtered_df["rate"].mean()

    total_votes = filtered_df["votes"].sum()

    total_locations = filtered_df["location"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🍽️ Restaurants",
            f"{total_restaurants:,}"
        )

    with col2:

        st.metric(
            "⭐ Average Rating",
            f"{avg_rating:.2f}"
            if not np.isnan(avg_rating)
            else "N/A"
        )

    with col3:

        st.metric(
            "👍 Total Votes",
            f"{int(total_votes):,}"
        )

    with col4:

        st.metric(
            "📍 Locations",
            f"{total_locations:,}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # CHART 1 - RATING DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("⭐ Rating Distribution")

    fig1 = px.histogram(
        filtered_df,
        x="rate",
        nbins=20,
        title="Restaurant Rating Distribution",
        labels={
            "rate": "Rating",
            "count": "Restaurants"
        }
    )

    fig1.update_layout(
        height=450
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CHART 2 - TOP LOCATIONS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        location_data = (
            filtered_df["location"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        location_data.columns = [
            "Location",
            "Restaurants"
        ]

        fig2 = px.bar(
            location_data,
            x="Restaurants",
            y="Location",
            orientation="h",
            title="Top 10 Restaurant Locations"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # --------------------------------------------------------
    # CHART 3 - ONLINE ORDER
    # --------------------------------------------------------

    with col2:

        order_data = (
            filtered_df["online_order"]
            .value_counts()
            .reset_index()
        )

        order_data.columns = [
            "Online Order",
            "Restaurants"
        ]

        fig3 = px.pie(
            order_data,
            names="Online Order",
            values="Restaurants",
            title="Online Ordering Availability"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    # --------------------------------------------------------
    # CHART 4 - TABLE BOOKING
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        table_data = (
            filtered_df["book_table"]
            .value_counts()
            .reset_index()
        )

        table_data.columns = [
            "Book Table",
            "Restaurants"
        ]

        fig4 = px.pie(
            table_data,
            names="Book Table",
            values="Restaurants",
            title="Table Booking Availability"
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

    # --------------------------------------------------------
    # CHART 5 - VOTES VS RATING
    # --------------------------------------------------------

    with col2:

        scatter_df = filtered_df[
            ["votes", "rate"]
        ].dropna()

        # Reduce points for browser performance
        if len(scatter_df) > 10000:

            scatter_df = scatter_df.sample(
                10000,
                random_state=42
            )

        fig5 = px.scatter(
            scatter_df,
            x="votes",
            y="rate",
            opacity=0.5,
            title="Votes vs Restaurant Rating",
            labels={
                "votes": "Votes",
                "rate": "Rating"
            }
        )

        st.plotly_chart(
            fig5,
            use_container_width=True
        )


# ============================================================
# RESTAURANT ANALYSIS
# ============================================================

elif page == "📊 Restaurant Analysis":

    st.title("📊 Restaurant Analysis")

    st.markdown(
        "### Detailed analysis of restaurant types, cuisines, costs and ratings"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    locations = sorted(
        df["location"]
        .dropna()
        .astype(str)
        .unique()
    )

    location = st.selectbox(
        "📍 Select Location",
        ["All"] + locations
    )

    analysis_df = df.copy()

    if location != "All":

        analysis_df = analysis_df[
            analysis_df["location"].astype(str)
            == location
        ]

    # --------------------------------------------------------
    # CHART 6 - RESTAURANT TYPE
    # --------------------------------------------------------

    rest_type_data = (
        analysis_df["rest_type"]
        .value_counts()
        .head(15)
        .reset_index()
    )

    rest_type_data.columns = [
        "Restaurant Type",
        "Restaurants"
    ]

    fig6 = px.bar(
        rest_type_data,
        x="Restaurants",
        y="Restaurant Type",
        orientation="h",
        title="Top Restaurant Types"
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CHART 7 - AVERAGE RATING BY LOCATION
    # --------------------------------------------------------

    rating_location = (
        analysis_df
        .groupby("location")["rate"]
        .agg(["mean", "count"])
        .reset_index()
    )

    rating_location = rating_location[
        rating_location["count"] >= 20
    ]

    rating_location = (
        rating_location
        .sort_values("mean", ascending=False)
        .head(15)
    )

    fig7 = px.bar(
        rating_location,
        x="location",
        y="mean",
        title="Top Locations by Average Rating",
        labels={
            "location": "Location",
            "mean": "Average Rating"
        }
    )

    fig7.update_yaxes(
        range=[0, 5]
    )

    st.plotly_chart(
        fig7,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CHART 8 - COST VS RATING
    # --------------------------------------------------------

    cost_df = analysis_df[
        [
            "approx_cost(for two people)",
            "rate"
        ]
    ].dropna()

    fig8 = px.scatter(
        cost_df.sample(
            min(len(cost_df), 10000),
            random_state=42
        ),
        x="approx_cost(for two people)",
        y="rate",
        opacity=0.5,
        title="Cost for Two vs Rating",
        labels={
            "approx_cost(for two people)": "Cost for Two",
            "rate": "Rating"
        }
    )

    st.plotly_chart(
        fig8,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CHART 9 - TOP CUISINES
    # --------------------------------------------------------

    cuisine_series = (
        analysis_df["cuisines"]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
    )

    cuisine_data = (
        cuisine_series
        .value_counts()
        .head(15)
        .reset_index()
    )

    cuisine_data.columns = [
        "Cuisine",
        "Restaurants"
    ]

    fig9 = px.bar(
        cuisine_data,
        x="Restaurants",
        y="Cuisine",
        orientation="h",
        title="Top 15 Cuisines"
    )

    st.plotly_chart(
        fig9,
        use_container_width=True
    )

    # --------------------------------------------------------
    # TOP RATED RESTAURANTS
    # --------------------------------------------------------

    st.subheader("🏆 Top Rated Restaurants")

    if "name" in analysis_df.columns:

        top_restaurants = (
            analysis_df[
                ["name", "location", "rate", "votes"]
            ]
            .dropna(subset=["rate"])
            .sort_values(
                ["rate", "votes"],
                ascending=[False, False]
            )
            .head(20)
        )

        st.dataframe(
            top_restaurants,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ML MODEL PAGE
# ============================================================

elif page == "🤖 ML Model":

    st.title("🤖 Machine Learning Model Performance")

    st.markdown(
        "### Comparing machine learning models for restaurant rating prediction"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL RESULTS
    # --------------------------------------------------------

    model_results = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest"
        ],

        "MAE": [
            0.199690,
            0.183270,
            0.180538
        ],

        "RMSE": [
            0.286414,
            0.276404,
            0.260442
        ],

        "R2 Score": [
            0.575974,
            0.605096,
            0.649389
        ]
    })

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    best_model_name = (
        model_results
        .sort_values(
            "R2 Score",
            ascending=False
        )
        .iloc[0]["Model"]
    )

    best_r2 = (
        model_results
        .sort_values(
            "R2 Score",
            ascending=False
        )
        .iloc[0]["R2 Score"]
    )

    best_mae = (
        model_results
        .sort_values(
            "MAE",
            ascending=True
        )
        .iloc[0]["MAE"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🏆 Best Model",
            best_model_name
        )

    with col2:

        st.metric(
            "R² Score",
            f"{best_r2:.3f}"
        )

    with col3:

        st.metric(
            "Lowest MAE",
            f"{best_mae:.3f}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL TABLE
    # --------------------------------------------------------

    st.subheader("📋 Model Comparison")

    st.dataframe(
        model_results.style.format({
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R2 Score": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CHART 10 - R2 COMPARISON
    # --------------------------------------------------------

    fig10 = px.bar(
        model_results,
        x="Model",
        y="R2 Score",
        title="Model Comparison - R² Score",
        text="R2 Score"
    )

    fig10.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig10.update_yaxes(
        range=[0, 1]
    )

    st.plotly_chart(
        fig10,
        use_container_width=True
    )

    # --------------------------------------------------------
    # MAE / RMSE COMPARISON
    # --------------------------------------------------------

    metrics_long = model_results.melt(
        id_vars="Model",
        value_vars=["MAE", "RMSE"],
        var_name="Metric",
        value_name="Value"
    )

    fig11 = px.bar(
        metrics_long,
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        title="MAE vs RMSE"
    )

    st.plotly_chart(
        fig11,
        use_container_width=True
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("🌟 Feature Importance")

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({

            "Feature": model_features,

            "Importance": model.feature_importances_

        })

        importance = (
            importance
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(20)
        )

        fig12 = px.bar(
            importance.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 20 Important Features"
        )

        st.plotly_chart(
            fig12,
            use_container_width=True
        )

        st.dataframe(
            importance,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Feature importance is not available for this model."
        )

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    st.subheader("🎯 Actual vs Predicted Ratings")

    try:

        base_features = [
            "online_order",
            "book_table",
            "votes",
            "location",
            "rest_type",
            "cuisines",
            "approx_cost(for two people)",
            "listed_in(type)",
            "listed_in(city)"
        ]

        prediction_df = df[
            [
                col
                for col in base_features
                if col in df.columns
            ]
        ].copy()

        # Add missing required columns
        for col in base_features:

            if col not in prediction_df.columns:

                prediction_df[col] = 0

        # Numeric mappings if required
        if "online_order" in model_features:

            prediction_df["online_order"] = (
                prediction_df["online_order"]
                .map({
                    "Yes": 1,
                    "No": 0
                })
            )

        if "book_table" in model_features:

            prediction_df["book_table"] = (
                prediction_df["book_table"]
                .map({
                    "Yes": 1,
                    "No": 0
                })
            )

        prediction_df["votes"] = pd.to_numeric(
            prediction_df["votes"],
            errors="coerce"
        ).fillna(0)

        prediction_df["log_votes"] = np.log1p(
            prediction_df["votes"]
        )

        prediction_df["cuisine_count"] = (
            prediction_df["cuisines"]
            .astype(str)
            .apply(lambda x: len(x.split(",")))
        )

        prediction_df["rest_type_count"] = (
            prediction_df["rest_type"]
            .astype(str)
            .apply(lambda x: len(x.split(",")))
        )

        categorical_cols = [
            "location",
            "rest_type",
            "cuisines",
            "listed_in(type)",
            "listed_in(city)"
        ]

        categorical_cols = [
            col
            for col in categorical_cols
            if col in prediction_df.columns
        ]

        prediction_df = pd.get_dummies(
            prediction_df,
            columns=categorical_cols,
            drop_first=True
        )

        bool_columns = prediction_df.select_dtypes(
            include=["bool"]
        ).columns

        for col in bool_columns:

            prediction_df[col] = (
                prediction_df[col].astype(int)
            )

        prediction_df = prediction_df.reindex(
            columns=model_features,
            fill_value=0
        )

        prediction_df = prediction_df.fillna(0)

        actual = df.loc[
            prediction_df.index,
            "rate"
        ]

        predicted = model.predict(
            prediction_df
        )

        comparison_df = pd.DataFrame({
            "Actual": actual.values,
            "Predicted": predicted
        })

        comparison_df = comparison_df.dropna()

        comparison_df = comparison_df.sample(
            min(len(comparison_df), 3000),
            random_state=42
        )

        fig13 = px.scatter(
            comparison_df,
            x="Actual",
            y="Predicted",
            opacity=0.5,
            title="Actual vs Predicted Rating"
        )

        fig13.add_shape(
            type="line",
            x0=1,
            y0=1,
            x1=5,
            y1=5,
            line=dict(
                dash="dash"
            )
        )

        fig13.update_xaxes(
            range=[1, 5]
        )

        fig13.update_yaxes(
            range=[1, 5]
        )

        st.plotly_chart(
            fig13,
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            "Actual vs Predicted chart could not be generated."
        )

        st.code(str(e))


# ============================================================
# RATING PREDICTION PAGE
# ============================================================

elif page == "🔮 Rating Prediction":

    st.title("🔮 Restaurant Rating Prediction")

    st.markdown(
        "### Enter restaurant details to predict its rating"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        location_options = sorted(
            df["location"]
            .dropna()
            .astype(str)
            .unique()
        )

        location = st.selectbox(
            "📍 Restaurant Location",
            location_options
        )

        rest_type_options = sorted(
            df["rest_type"]
            .dropna()
            .astype(str)
            .unique()
        )

        rest_type = st.selectbox(
            "🍴 Restaurant Type",
            rest_type_options
        )

        cuisine_options = sorted(
            df["cuisines"]
            .dropna()
            .astype(str)
            .unique()
        )

        cuisine = st.selectbox(
            "🥘 Cuisine",
            cuisine_options
        )

        listed_type_options = sorted(
            df["listed_in(type)"]
            .dropna()
            .astype(str)
            .unique()
        )

        listed_type = st.selectbox(
            "📋 Listed In Type",
            listed_type_options
        )

    with col2:

        listed_city_options = sorted(
            df["listed_in(city)"]
            .dropna()
            .astype(str)
            .unique()
        )

        listed_city = st.selectbox(
            "🏙️ Listed In City",
            listed_city_options
        )

        online_order = st.selectbox(
            "🛵 Online Order",
            ["Yes", "No"]
        )

        book_table = st.selectbox(
            "🍽️ Table Booking",
            ["Yes", "No"]
        )

        votes = st.number_input(
            "👍 Number of Votes",
            min_value=0,
            max_value=100000,
            value=100,
            step=10
        )

        cost = st.number_input(
            "💰 Approximate Cost for Two",
            min_value=0,
            max_value=100000,
            value=500,
            step=50
        )

    st.markdown("---")

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🔮 Predict Restaurant Rating",
        use_container_width=True
    )

    if predict_button:

        try:
    
            # Prepare input
            input_data = prepare_model_input(
                location=location,
                rest_type=rest_type,
                cuisines=cuisine,
                online_order=online_order,
                book_table=book_table,
                votes=votes,
                cost=cost,
                listed_type=listed_type,
                listed_city=listed_city
            )

            # ==========================================
            # MAKE PREDICTION
            # ==========================================
    
            prediction = model.predict(input_data)[0]
    
            # Keep rating between 1 and 5
            prediction = np.clip(prediction, 1, 5)
    
            st.markdown("---")
    
            st.subheader("🎯 Predicted Restaurant Rating")

            # ==========================================
            # SHOW PREDICTED RATING
            # ==========================================
    
            st.markdown(
                f"""
                <div class="prediction-box">
                    <div class="metric-title">
                        Machine Learning Prediction
                    </div>
    
                    <div class="prediction-value">
                        ⭐ {prediction:.2f} / 5
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ==========================================
            # RATING INTERPRETATION
            # ==========================================
    
            if prediction >= 4.5:
                message = "Excellent restaurant rating! 🏆"
    
            elif prediction >= 4.0:
                message = "Very good restaurant! ⭐"
    
            elif prediction >= 3.5:
                message = "Good restaurant with positive potential 👍"
    
            elif prediction >= 3.0:
                message = "Average restaurant rating."
    
            else:
                message = "Below-average predicted rating."
    
            st.info(message)

            # ==========================================
            # RATING PROGRESS BAR
            # ==========================================
    
            st.write("Rating Score")
    
            rating_percentage = prediction / 5
    
            st.progress(float(rating_percentage))
    
            # ==========================================
            # RESTAURANT DETAILS
            # ==========================================
    
            st.subheader("📋 Restaurant Details")
    
            result_data = pd.DataFrame({

                "Feature": [
                    "Location",
                    "Restaurant Type",
                    "Cuisine",
                    "Online Order",
                    "Table Booking",
                    "Votes",
                    "Cost for Two",
                    "Listed Type",
                    "Listed City"
                ],
    
                "Value": [
                    location,
                    rest_type,
                    cuisine,
                    online_order,
                    book_table,
                    votes,
                    f"₹{cost:,}",
                    listed_type,
                    listed_city
                ]

            })
    
            st.dataframe(
                result_data,
                use_container_width=True,
                hide_index=True
            )


        except Exception as e:
    
            st.error("Prediction failed.")
    
            st.write("Error details:")
    
            st.code(str(e))
    
            st.markdown(
                        """
                        **Most likely reason:** the saved model and
                        `zomato_features.pkl` were created using different
                        preprocessing steps.
        
                        Make sure both files were saved from the same
                        final training code.
                        """
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:gray;">

    🍽️ <b>Zomato Restaurant Analytics & ML Rating Prediction</b>

    <br>

    Built with Python • Pandas • Scikit-learn • Plotly • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)