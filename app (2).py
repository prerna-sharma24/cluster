
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Customer Clustering & Prediction", layout="wide")
st.title("🤖 Customer Clustering & Prediction Dashboard (Updated Inputs)")

# ----- Utility / Load -----
@st.cache_resource
    # Try a few common filenames used in your notebook

    # Basic cleaning / filling as in your notebook
    df["Ever_Married"] = df.get("Ever_Married", pd.Series()).fillna('Yes')
    df["Graduated"] = df.get("Graduated", pd.Series()).fillna('Yes')
    df["Profession"] = df.get("Profession", pd.Series()).fillna('Artist')
    if "Work_Experience" in df.columns:
        df["Work_Experience"] = df["Work_Experience"].fillna(3).astype(int)
    if "Family_Size" in df.columns:
        df["Family_Size"] = df["Family_Size"].fillna(2).astype(int)

    # Drop columns used in notebook if present
    for c in ["Var_1", "ID"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # For later reference, keep a copy of original (unencoded) df
    df_original = df.copy()

    # Label encode object columns to produce numeric X like in notebook
    le_map = {}
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        le_map[col] = le

    feature_cols = ["Gender", "Ever_Married", "Age", "Graduated",
                    "Profession", "Work_Experience", "Spending_Score", "Family_Size"]
    # If some feature columns are missing, try to infer them
    feature_cols = [c for c in feature_cols if c in df_encoded.columns]

    X = df_encoded[feature_cols]

    # Load scaler and models if available
    try:
        scaler = joblib.load("sc.pkl")
    except Exception as e:
        st.error(f"Scaler 'sc.pkl' not found or failed to load: {e}")
        return None, None, None, None

    X_scaled = scaler.transform(X)

    models = {}
    # Load models with the filenames inferred from notebook
    def safe_load(fn):
        try:
            return joblib.load(fn)
        except Exception:
            return None

    models["KMeans"] = safe_load("km.pkl")
    models["DBSCAN"] = safe_load("dbscan.pkl")
    models["MeanShift"] = safe_load("mean.pkl")
    # fuzzy.pkl in your notebook was joblib.dump(fuzz.cmeans, "fuzzy.pkl")
    # So we won't expect a fitted model file for fuzzy; we'll call fuzz.cluster.cmeans when needed.
    models["Fuzzy C-Means"] = None

    return df_original, X_scaled, scaler, models, le_map, feature_cols

loaded = load_data_and_models()
if loaded[0] is None:
    st.stop()

df_original, X_scaled, scaler, models, le_map, feature_cols = loaded

# ----- Sidebar: visualization choice -----
st.sidebar.header("Visualization & Settings")
vis_algo = st.sidebar.selectbox("Choose algorithm to visualize", ["KMeans", "DBSCAN", "MeanShift", "Fuzzy C-Means"])
fcm_clusters = st.sidebar.number_input("FCM: Number of clusters (only for Fuzzy C-Means)", min_value=2, max_value=10, value=3, step=1)

st.subheader(f"📊 Cluster Visualization — {vis_algo}")

# compute cluster labels for visualization
if vis_algo == "Fuzzy C-Means":
    # run fuzzy c-means on dataset's scaled features (transpose required)
    try:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_scaled.T, c=int(fcm_clusters), m=2.0, error=0.005, maxiter=1000, init=None)
        cluster_labels = np.argmax(u, axis=0)
    except Exception as e:
        st.error(f"Error running Fuzzy C-Means: {e}")
        cluster_labels = np.full(shape=(X_scaled.shape[0],), fill_value=-1)
else:
    model = models.get(vis_algo)
    if model is None:
        st.warning(f"No saved model found for {vis_algo}. Visualization will show an unclustered scatter.")
        cluster_labels = np.full(shape=(X_scaled.shape[0],), fill_value=-1)
    else:
        # many sklearn cluster models store .labels_, otherwise use predict
        if hasattr(model, "labels_") and getattr(model, "labels_") is not None:
            cluster_labels = model.labels_
        else:
            try:
                cluster_labels = model.predict(X_scaled)
            except Exception:
                cluster_labels = np.full(shape=(X_scaled.shape[0],), fill_value=-1)

# Plot Age vs Spending_Score colored by cluster label (if these columns exist)
if ("Age" in df_original.columns) and ("Spending_Score" in df_original.columns):
    fig, ax = plt.subplots(figsize=(9, 5))
    # Make palette automatic (do not force colors)
    scatter = ax.scatter(df_original["Age"], df_original["Spending_Score"], c=cluster_labels, cmap='tab10', s=40)
    ax.set_xlabel("Age")
    ax.set_ylabel("Spending Score")
    ax.set_title(f"Clusters by {vis_algo}")
    # create legend for cluster numbers if there are distinct clusters
    unique_labels = np.unique(cluster_labels)
    handles = []
    for ul in unique_labels:
        handles.append(plt.Line2D([], [], marker="o", linestyle="", label=f"Cluster {int(ul)}"))
    if len(handles) > 0:
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
else:
    st.info("Dataset does not contain both 'Age' and 'Spending_Score' columns — showing first 5 rows instead.")
    st.dataframe(df_original.head())

# ----- Prediction UI: explicit input columns -----
st.subheader("🔧 Input Features (enter values below)")

# We'll show the feature columns used and then provide inputs in two rows (4 inputs per row)
st.markdown("**Features used for prediction:** " + ", ".join(feature_cols))

# Initialize session state containers
if "batch" not in st.session_state:
    st.session_state.batch = pd.DataFrame(columns=feature_cols)
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=feature_cols + ["Algorithm", "Predicted_Cluster"])

# Mapping for professions (keeps same mapping used previously in your app code)
profession_map = {
    "Artist": 0, "Doctor": 1, "Engineer": 2, "Entertainment": 3,
    "Executive": 4, "Healthcare": 5, "Homemaker": 6, "Lawyer": 7,
    "Marketing": 8, "Other": 9
}

# Layout inputs across two rows (responsive)
row1_cols = st.columns(len(feature_cols[:4]) or 1)
row2_cols = st.columns(len(feature_cols[4:8]) or 1)

# create a dictionary to hold raw input values
input_vals = {}

# Row 1 inputs (first up to 4 features)
for i, col in enumerate(feature_cols[:4]):
    with row1_cols[i]:
        if col in ["Gender"]:
            input_vals[col] = st.selectbox(col, ["Male", "Female"], key=f"inp_{col}")
        elif col in ["Ever_Married", "Graduated"]:
            input_vals[col] = st.selectbox(col, ["Yes", "No"], key=f"inp_{col}")
        elif col in ["Age"]:
            input_vals[col] = st.number_input(col, value=30.0, step=1.0, key=f"inp_{col}")
        else:
            input_vals[col] = st.text_input(col, key=f"inp_{col}")

# Row 2 inputs (remaining up to 4 features)
for j, col in enumerate(feature_cols[4:8]):
    with row2_cols[j]:
        if col in ["Profession"]:
            input_vals[col] = st.selectbox(col, list(profession_map.keys()), key=f"inp_{col}")
        elif col in ["Work_Experience", "Family_Size"]:
            input_vals[col] = st.number_input(col, value=3 if col=="Work_Experience" else 2, step=1, key=f"inp_{col}")
        elif col in ["Spending_Score"]:
            input_vals[col] = st.number_input(col, value=60.0, step=1.0, key=f"inp_{col}")
        else:
            input_vals[col] = st.text_input(col, key=f"inp_{col}")

# Convert raw inputs to encoded numeric row matching feature_cols order
def encode_input_row(raw):
    row = []
    for c in feature_cols:
        val = raw.get(c, None)
        if c == "Gender":
            row.append(1 if str(val).lower() == "male" else 0)
        elif c in ["Ever_Married", "Graduated"]:
            row.append(1 if str(val).lower() == "yes" else 0)
        elif c == "Profession":
            row.append(int(profession_map.get(val, 9)))
        elif c in ["Work_Experience", "Family_Size"]:
            row.append(int(val))
        elif c == "Spending_Score":
            row.append(float(val))
        elif c == "Age":
            row.append(float(val))
        else:
            # fallback: try numeric conversion, otherwise 0
            try:
                row.append(float(val))
            except Exception:
                row.append(0.0)
    return row

# Buttons for adding to batch / predicting single
colA, colB, colC = st.columns([1,1,1])
with colA:
    if st.button("➕ Add to Batch"):
        row = encode_input_row(input_vals)
        df_row = pd.DataFrame([row], columns=feature_cols)
        st.session_state.batch = pd.concat([st.session_state.batch, df_row], ignore_index=True)
        st.success("Row added to batch. See 'Batch' below.")

with colB:
    pred_algo = st.selectbox("Prediction algorithm (single predict)", ["KMeans", "MeanShift"], key="pred_algo_select")

with colC:
    if st.button("🔮 Predict (single)"):
        # perform single prediction using pred_algo
        try:
            row = np.array(encode_input_row(input_vals)).reshape(1, -1)
            row_scaled = scaler.transform(row)
            model = models.get(pred_algo)
            if model is None:
                st.error(f"Model file for {pred_algo} not found. Make sure the corresponding .pkl is present.")
            else:
                pred = model.predict(row_scaled)[0]
                st.success(f"Predicted cluster: {pred} (algorithm: {pred_algo})")
                out = {c: input_vals.get(c) for c in feature_cols}
                out["Algorithm"] = pred_algo
                out["Predicted_Cluster"] = int(pred)
                st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([out])], ignore_index=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Show batch table and allow batch prediction
st.markdown("### 🧾 Batch (rows you added)")
st.dataframe(st.session_state.batch)

batch_col1, batch_col2 = st.columns(2)
with batch_col1:
    if st.button("⏹ Reset Batch"):
        st.session_state.batch = pd.DataFrame(columns=feature_cols)
        st.success("Batch cleared.")

with batch_col2:
    if st.button("▶ Predict Batch"):
        if st.session_state.batch.empty:
            st.warning("Batch is empty — add rows first.")
        else:
            # Predict for the entire batch using chosen algorithm
            pred_algo_batch = st.selectbox("Choose batch prediction algorithm", ["KMeans", "MeanShift"], key="batch_algo_select")
            model = models.get(pred_algo_batch)
            if model is None:
                st.error(f"Model for {pred_algo_batch} not found. Can't predict batch.")
            else:
                try:
                    batch_vals = st.session_state.batch[feature_cols].astype(float).values
                    batch_scaled = scaler.transform(batch_vals)
                    preds = model.predict(batch_scaled)
                    st.session_state.batch["Predicted_Cluster"] = preds.astype(int)
                    # Add to history as well
                    history_rows = st.session_state.batch.copy()
                    history_rows["Algorithm"] = pred_algo_batch
                    # Move Predicted_Cluster to end and convert to ints
                    history_rows["Predicted_Cluster"] = history_rows["Predicted_Cluster"].astype(int)
                    st.session_state.history = pd.concat([st.session_state.history, history_rows], ignore_index=True)
                    st.success("Batch predicted and added to history.")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

# ----- Prediction History & Download -----
st.markdown("## 📜 Prediction History")
if st.session_state.history.empty:
    st.info("No predictions yet — use the single predict or batch predict actions to populate history.")
else:
    st.dataframe(st.session_state.history)

    csv = st.session_state.history.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download history as CSV", data=csv, file_name="predictions_history.csv", mime="text/csv")

# ----- End -----
st.markdown("---")
st.caption("If any column is missing from the dataset or the models/scaler files are not present, the app will notify you. I added an explicit, visible input area (split into rows/columns) and buttons to Add to Batch / Predict single / Predict batch / Reset batch.")

