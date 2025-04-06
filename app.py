import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import squarify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 Customer Purchase Analysis Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("📌 Raw Data")
    st.dataframe(data.head(10))

    st.subheader("📊 Data Description")
    st.write(data.describe())

    # Clean data
    data = data.fillna(method='ffill')

    # Feature Engineering
    if 'Income' in data.columns and 'Tax' in data.columns:
        data['PAT'] = data['Income'] - data['Tax']

    # Sidebar filters
    st.sidebar.header("🎛 Filters")
    selected_product = st.sidebar.multiselect("Filter by Product", options=data['Product'].unique(), default=data['Product'].unique())
    selected_coupon = st.sidebar.multiselect("Filter by Coupon Type", options=data['CouponType'].unique(), default=data['CouponType'].unique())
    data = data[data['Product'].isin(selected_product) & data['CouponType'].isin(selected_coupon)]

    # Graph 1: Boxplot
    st.subheader("📦 Sales Distribution by Product")
    fig1, ax1 = plt.subplots()
    sns.boxplot(data=data, x='Product', y='Sales', palette='Set2', ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    st.pyplot(fig1)

    # Graph 2: Pie Chart
    st.subheader("🥧 Coupon Type Usage")
    fig2, ax2 = plt.subplots()
    data['CouponType'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax2, colors=sns.color_palette("pastel"))
    ax2.set_ylabel('')
    st.pyplot(fig2)

    # Graph 3: Stacked Bar
    st.subheader("🧱 Coupon Usage per Product")
    fig3, ax3 = plt.subplots()
    pd.crosstab(data['Product'], data['CouponUsed']).plot(kind='bar', stacked=True, ax=ax3, colormap='Paired')
    ax3.set_ylabel("Count")
    ax3.set_xlabel("Product")
    st.pyplot(fig3)

    # Graph 4: Pair Plot
    st.subheader("📊 Feature Pairwise Relationship")
    fig4 = sns.pairplot(data[['Sales', 'Income', 'Age', 'PurchaseMonth']], diag_kind='kde')
    st.pyplot(fig4)

    # Graph 5: Line Chart for Monthly Sales
    st.subheader("📈 Monthly Sales Trend")
    data['Year'] = data['Year'].astype(str)
    data['Date'] = pd.to_datetime(data['Year'] + '-' + data['PurchaseMonth'].astype(str))
    monthly_sales = data.groupby('Date')['Sales'].sum().reset_index()
    fig5, ax5 = plt.subplots()
    sns.lineplot(x='Date', y='Sales', data=monthly_sales, marker='o', ax=ax5)
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
    st.pyplot(fig5)

    # Graph 6: Treemap
    st.subheader("🗂️ Treemap - Product-wise Sales")
    product_sales = data.groupby('Product')['Sales'].sum()
    fig6 = plt.figure(figsize=(12, 6))
    squarify.plot(sizes=product_sales.values, label=product_sales.index, alpha=0.8, color=sns.color_palette("Set3"))
    plt.axis('off')
    st.pyplot(fig6)

    # Model Training
    st.subheader("🤖 Linear Regression Model")
    features = ['Income', 'Age', 'PurchaseMonth']
    if all(col in data.columns for col in features):
        X = data[features]
        y = data['Sales']

        imp = SimpleImputer(strategy='mean')
        X = imp.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Model Evaluation
        st.markdown(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")
        st.markdown(f"**Mean Absolute Error:** {mean_absolute_error(y_test, y_pred):.2f}")
        st.markdown(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.2f}")

        # Residual Plot
        st.subheader("📉 Residual Analysis")
        residuals = y_test - y_pred
        fig7, ax7 = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax7)
        ax7.set_title("Residual Distribution")
        st.pyplot(fig7)

        # Error % Plot
        st.subheader("🔍 Error Percentage")
        error_percent = abs((y_test - y_pred) / y_test) * 100
        fig8, ax8 = plt.subplots()
        sns.lineplot(y=error_percent, x=range(len(error_percent)), ax=ax8)
        ax8.set_ylabel("Error %")
        st.pyplot(fig8)

    else:
        st.error("Required features not found in data!")

else:
    st.info("Please upload a CSV file to get started.")
