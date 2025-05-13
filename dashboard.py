import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Retail Detection Dashboard", layout="wide")

st.title("🛍️ Retail Detection Summary")

# Load data
df = pd.read_csv("detection_summary.csv")

# Sidebar filters
st.sidebar.header("Filters")

# Store and date filters
if "Store" in df.columns:
    store_filter = st.sidebar.multiselect("Filter by Store", df["Store"].unique())
    if store_filter:
        df = df[df["Store"].isin(store_filter)]

if "Date" in df.columns:
    date_filter = st.sidebar.multiselect("Filter by Date", df["Date"].unique())
    if date_filter:
        df = df[df["Date"].isin(date_filter)]

# Video and label filters
video_filter = st.sidebar.multiselect("Filter by Video", df["Video Name"].unique())
label_filter = st.sidebar.multiselect("Filter by Label", df["Label"].unique())

if video_filter:
    df = df[df["Video Name"].isin(video_filter)]
if label_filter:
    df = df[df["Label"].isin(label_filter)]

# Chart: Total detections by label
# Clean column names and ensure string labels
df.columns = [col.strip().title() for col in df.columns]
df = df.dropna(subset=["Label", "Confidence"])
df["Label"] = df["Label"].astype(str)

# Count detections by label
label_counts = df["Label"].value_counts().reset_index()
label_counts.columns = ["Label", "Detections"]

# Plot with Altair
st.altair_chart(
    alt.Chart(label_counts)
    .mark_bar()
    .encode(
        x=alt.X("Label:N", sort="-y"),
        y=alt.Y("Detections:Q"),
        tooltip=["Label", "Detections"]
    )
    .properties(width=800),
    use_container_width=True
)

# Table
st.subheader("📄 Detection Entries")
st.dataframe(df)

# Download button
st.download_button("⬇️ Download CSV", filtered_df.to_csv(index=False), "detection_summary_filtered.csv")
