# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import skfuzzy as fuzz
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, MeanShift

st.set_page_config(page_title="Customer Clustering & Prediction (3D)", layout="wide")
st.title("🤖 Customer Clustering & Prediction — 3D Visualization")

# ---------------- Helpers ----------------
def safe_joblib_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

def basic_clean_and_encode(df):
    # Fill missing with reasonable defaults like in your notebook
    if "Ever_Married" in df.columns:
        df["Ever_Married"] = df["Ever_Married"].fillna("Yes")
    if "Graduated" in df.columns:
        df["Graduated"] = df["Graduated"].fillna("Yes")
    if "Profession" in df.columns:
        df["Profession"] = df["Profession"].fillna("Artist")
    if "Work_Experience" in df.columns:
        df["Work_Experience"] = df["Work_Experience"].fillna(3).astype(int)
    if "Family_Size" in df.columns:
        df["Family_Size"] = df["Family_Size"].fillna(2).astype(int)

    # drop known unused cols if present
    for c in ["Var_1", "ID"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Keep original copy
    df_original = df.copy()

    # Encode textual columns with LabelEncoder (store encoders for possible inverse)
    le_map = {}
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        le_map[col] = le

    # Desired feature columns (from your notebook)
    feature_cols = ["Gender", "Ever_Married", "Age", "Graduated",
                    "Profession", "Work_Experience", "Spending_Score", "Family_Size"]
    # Use only those present
    feature_cols = [c for c in feature_cols if c in df_enc.columns]

    X = df_enc[feature_cols].copy()

    return df_original, X, le_map, feature_cols

def ensure_scaler_and_scale(X, scaler_path="sc.pkl"):
    scaler = safe_joblib_load(scaler_path)
    if scaler is None:
        # fit a StandardScaler on X (fallback) — store in session_state to reuse during the app session
        scaler = StandardScaler()
        scaler.fit(X)
        st.warning("No saved scaler found ('sc.pkl'). Using a scaler fit on the provided dataset (fallback).")
    X_scaled = scaler.transform(X)
    return scaler, X_scaled

def fit_or_load_model(name, model_path, X_scaled, fallback_fit_fn):
    model = safe_joblib_load(model_path)
    if model is not None:
        return model, False  # loaded, not fitted here
    # fallback: fit model now
    try:
        model = fallback_fit_fn(X_scaled)
        st.info(f"No saved {name} model found. Fitted a {name} model on-the-fly for visualization/prediction.")
        return model, True
    except Exception as e:
        st.error(f"Failed to load or fit {name}: {e}")
        return None, False

# ---------------- Load dataset (or upload) ----------------
st.sidebar.header("Data & Model files")
uploaded_csv = st.sidebar.file_uploader("Upload dataset CSV (optional)", type=["csv"])
if uploaded_csv is not None:
    try:
        df_in = pd.read_csv(uploaded_csv)
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded CSV: {e}")
        st.stop()
else:
    # try common filenames (Test.csv etc.)
    df_in = None
    for name in ("Test.csv", "test.csv", "data.csv", "Test_data.csv"):
        try:
            df_in = pd.read_csv(name)
            break
        except Exception:
            continue
    if df_in is None:
        st.sidebar.warning("No dataset found in working directory. Upload a CSV to proceed.")
        df_in = pd.DataFrame()  # empty placeholder

# Model file uploaders (optional)
st.sidebar.markdown("**Optional: upload model .pkl files**")
uploaded_km = st.sidebar.file_uploader("KMeans (.pkl)", type=["pkl"], key="km_upload")
uploaded_db = st.sidebar.file_uploader("DBSCAN (.pkl)", type=["pkl"], key="db_upload")
uploaded_ms = st.sidebar.file_uploader("MeanShift (.pkl)", type=["pkl"], key="ms_upload")
uploaded_sc = st.sidebar.file_uploader("Scaler (.pkl)", type=["pkl"], key="sc_upload")

# If model files uploaded via uploader, save them to disk temporarily for joblib to load
def save_uploaded_tmp(uploaded_file, target_name):
    if uploaded_file is None:
        return False
    bytes_data = uploaded_file.getvalue()
    with open(target_name, "wb") as f:
        f.write(bytes_data)
    return True

if uploaded_km:
    save_uploaded_tmp(uploaded_km, "km_uploaded.pkl")
if uploaded_db:
    save_uploaded_tmp(uploaded_db, "dbscan_uploaded.pkl")
if uploaded_ms:
    save_uploaded_tmp(uploaded_ms, "mean_uploaded.pkl")
if uploaded_sc:
    save_uploaded_tmp(uploaded_sc, "sc_uploaded.pkl")

# ---------------- Prepare data & encoders ----------------
if df_in.empty:
    st.warning("No data available. You can still use the Prediction UI (enter features manually).")
    # prepare empty placeholders
    df_original = pd.DataFrame()
    X = pd.DataFrame()
    le_map = {}
    feature_cols = ["Gender", "Ever_Married", "Age", "Graduated",
                    "Profession", "Work_Experience", "Spending_Score", "Family_Size"]
    # Leave scaler None for now; prediction will error until scaler exists or user uploads one.
    scaler = None
    X_scaled = np.zeros((0, len(feature_cols)))
else:
    df_original, X, le_map, feature_cols = basic_clean_and_encode(df_in)
    # choose scaler path — prefer uploaded sc_uploaded.pkl, else sc.pkl
    scaler_path = "sc_uploaded.pkl" if uploaded_sc else "sc.pkl"
    scaler, X_scaled = ensure_scaler_and_scale(X, scaler_path=scaler_path)

# ---------------- Model load / fit fallback ----------------
# Decide which file names to try (prefer uploaded saved names)
km_path = "km_uploaded.pkl" if uploaded_km else "km.pkl"
db_path = "dbscan_uploaded.pkl" if uploaded_db else "dbscan.pkl"
ms_path = "mean_uploaded.pkl" if uploaded_ms else "mean.pkl"

# fit functions for fallback
def fit_kmeans(Xs): return KMeans(n_clusters=3, random_state=3).fit(Xs)
def fit_dbscan(Xs): return DBSCAN(eps=3, min_samples=2).fit(Xs)
def fit_meanshift(Xs): return MeanShift(bandwidth=2.6).fit(Xs)

# Only attempt to fit if we have non-empty X_scaled
models = {}
if X_scaled.shape[0] > 0:
    models["KMeans"], _ = fit_or_load_model("KMeans", km_path, X_scaled, fit_kmeans)
    models["DBSCAN"], _ = fit_or_load_model("DBSCAN", db_path, X_scaled, fit_dbscan)
    models["MeanShift"], _ = fit_or_load_model("MeanShift", ms_path, X_scaled, fit_meanshift)
else:
    models["KMeans"] = safe_joblib_load(km_path)
    models["DBSCAN"] = safe_joblib_load(db_path)
    models["MeanShift"] = safe_joblib_load(ms_path)

# Fuzzy C-Means: we will run cmeans on-the-fly if needed
# ---------------- Visualization controls ----------------
st.sidebar.header("Visualization settings")
vis_algo = st.sidebar.selectbox("Choose algorithm to visualize", ["KMeans", "DBSCAN", "MeanShift", "Fuzzy C-Means"])
fcm_clusters = st.sidebar.number_input("FCM clusters (if Fuzzy C-Means)", min_value=2, max_value=10, value=3, step=1)
pca_components = st.sidebar.number_input("PCA components for 3D", min_value=2, max_value=8, value=3)

st.subheader(f"3D Cluster Visualization — {vis_algo}")

# Prepare 3D coordinates via PCA (if we have data)
if X_scaled.shape[0] > 0:
    # If feature count < requested PCA dims, fall back
    n_features = X_scaled.shape[1]
    n_comp = min(pca_components, n_features)
    if n_comp < 2:
        n_comp = min(2, n_features)
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(X_scaled)
    # If PCA returned fewer dims than 3, pad zeros to make 3D plotable
    if coords.shape[1] < 3:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 3 - coords.shape[1]))])
else:
    coords = np.zeros((0, 3))

# Compute labels depending on algorithm
if X_scaled.shape[0] == 0:
    cluster_labels = np.array([], dtype=int)
else:
    if vis_algo == "Fuzzy C-Means":
        try:
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_scaled.T, c=int(fcm_clusters), m=2.0,
                                                            error=0.005, maxiter=1000, init=None)
            cluster_labels = np.argmax(u, axis=0)
        except Exception as e:
            st.error(f"Fuzzy C-Means failed: {e}")
            cluster_labels = np.full((X_scaled.shape[0],), -1)
    else:
        model = models.get(vis_algo)
        if model is None:
            st.warning(f"No saved/fitted {vis_algo} model available; all points will be shown as cluster -1.")
            cluster_labels = np.full((X_scaled.shape[0],), -1)
        else:
            if hasattr(model, "labels_") and getattr(model, "labels_") is not None:
                # DBSCAN often has labels_ attribute
                cluster_labels = model.labels_
            else:
                try:
                    cluster_labels = model.predict(X_scaled)
                except Exception:
                    # fallback: try .labels_ or mark -1
                    if hasattr(model, "labels_"):
                        cluster_labels = model.labels_
                    else:
                        cluster_labels = np.full((X_scaled.shape[0],), -1)

# Make DataFrame for plotting
if X_scaled.shape[0] > 0:
    plot_df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "cluster": cluster_labels
    })
    # Add some original columns for hover if they exist
    for col in ["Age", "Spending_Score"]:
        if col in df_original.columns:
            plot_df[col] = df_original[col].values
else:
    plot_df = pd.DataFrame(columns=["x", "y", "z", "cluster"])

# Plot interactive 3D scatter
if not plot_df.empty:
    fig = px.scatter_3d(plot_df, x="x", y="y", z="z", color=plot_df["cluster"].astype(str),
                        hover_data=[c for c in ["Age", "Spending_Score"] if c in plot_df.columns],
                        title=f"3D clusters ({vis_algo}) — PCA projection")
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available to visualize. Upload a dataset or place Test.csv in the app folder.")

# ---------------- Prediction UI ----------------
st.markdown("---")
st.subheader("🔮 Prediction — single & batch")

# Input fields (explicit 8 features)
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    married = st.selectbox("Ever Married", ["Yes", "No"], index=0)
    age = st.number_input("Age", value=30.0, step=1.0)
    graduated = st.selectbox("Graduated", ["Yes", "No"], index=0)
with col2:
    profession = st.selectbox("Profession",
                              ["Artist", "Doctor", "Engineer", "Entertainment", "Executive",
                               "Healthcare", "Homemaker", "Lawyer", "Marketing", "Other"], index=0)
    work_exp = st.number_input("Work Experience (Years)", value=3, step=1)
    spending_score = st.number_input("Spending Score", value=60.0, step=1.0)
    family_size = st.number_input("Family Size", value=2, step=1)

profession_map = {
    "Artist": 0, "Doctor": 1, "Engineer": 2, "Entertainment": 3,
    "Executive": 4, "Healthcare": 5, "Homemaker": 6, "Lawyer": 7,
    "Marketing": 8, "Other": 9
}

def encode_inputs_row(vals):
    # Ensure same order as feature_cols; if feature_cols missing columns, still return 8-value row for predict-time scaler
    ordered = ["Gender", "Ever_Married", "Age", "Graduated", "Profession", "Work_Experience", "Spending_Score", "Family_Size"]
    row = []
    for c in ordered:
        v = vals.get(c)
        if c == "Gender":
            row.append(1 if str(v).lower() == "male" else 0)
        elif c in ["Ever_Married", "Graduated"]:
            row.append(1 if str(v).lower() == "yes" else 0)
        elif c == "Profession":
            row.append(int(profession_map.get(v, 9)))
        elif c in ["Work_Experience", "Family_Size"]:
            row.append(int(v))
        elif c == "Spending_Score":
            row.append(float(v))
        elif c == "Age":
            row.append(float(v))
        else:
            row.append(0.0)
    return np.array(row).reshape(1, -1)

# Session state for batch/history
if "batch" not in st.session_state:
    st.session_state.batch = pd.DataFrame(columns=["Gender", "Ever_Married", "Age", "Graduated",
                                                  "Profession", "Work_Experience", "Spending_Score", "Family_Size"])
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Gender", "Ever_Married", "Age", "Graduated",
                                                    "Profession", "Work_Experience", "Spending_Score", "Family_Size",
                                                    "Algorithm", "Predicted_Cluster"])

# Buttons row
b1, b2, b3 = st.columns([1, 1, 1])
with b1:
    if st.button("➕ Add to Batch"):
        row = {
            "Gender": gender, "Ever_Married": married, "Age": age, "Graduated": graduated,
            "Profession": profession, "Work_Experience": work_exp,
            "Spending_Score": spending_score, "Family_Size": family_size
        }
        st.session_state.batch = pd.concat([st.session_state.batch, pd.DataFrame([row])], ignore_index=True)
        st.success("Row added to batch.")

with b2:
    pred_algo = st.selectbox("Prediction algorithm", ["KMeans", "MeanShift"], key="pred_algo")
with b3:
    if st.button("🔮 Predict (single)"):
        vals = {"Gender": gender, "Ever_Married": married, "Age": age, "Graduated": graduated,
                "Profession": profession, "Work_Experience": work_exp,
                "Spending_Score": spending_score, "Family_Size": family_size}
        row_enc = encode_inputs_row(vals)
        # Scale: prefer uploaded scaler if present
        scaler_to_use = None
        if uploaded_sc:
            scaler_to_use = safe_joblib_load("sc_uploaded.pkl")
        else:
            scaler_to_use = safe_joblib_load("sc.pkl") or scaler  # fallback to earlier fitted scaler
        if scaler_to_use is None:
            st.error("No scaler available. Upload sc.pkl or dataset to auto-create a scaler.")
        else:
            try:
                row_scaled = scaler_to_use.transform(row_enc)
                model_path = "km_uploaded.pkl" if uploaded_km else "km.pkl" if pred_algo == "KMeans" else ("mean_uploaded.pkl" if uploaded_ms else "mean.pkl")
                model = safe_joblib_load(model_path)
                if model is None:
                    # fallback: fit model quickly on dataset if we have X_scaled
                    if X_scaled.shape[0] > 0:
                        if pred_algo == "KMeans":
                            model = KMeans(n_clusters=3, random_state=3).fit(X_scaled)
                        else:
                            model = MeanShift(bandwidth=2.6).fit(X_scaled)
                        st.info(f"Used on-the-fly fitted {pred_algo} for prediction.")
                    else:
                        st.error(f"No model available and cannot fit {pred_algo} (no dataset).")
                        model = None
                if model is not None:
                    pred = model.predict(row_scaled)[0]
                    st.success(f"Predicted cluster: {pred} (algorithm: {pred_algo})")
                    out = vals.copy()
                    out["Algorithm"] = pred_algo
                    out["Predicted_Cluster"] = int(pred)
                    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([out])], ignore_index=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Batch controls
st.markdown("### 🧾 Batch")
st.dataframe(st.session_state.batch)

colA, colB = st.columns(2)
with colA:
    if st.button("⏹ Reset Batch"):
        st.session_state.batch = pd.DataFrame(columns=st.session_state.batch.columns)
        st.success("Batch cleared.")
with colB:
    if st.button("▶ Predict Batch"):
        if st.session_state.batch.empty:
            st.warning("Batch empty — add rows first.")
        else:
            alg = st.selectbox("Batch model", ["KMeans", "MeanShift"], key="batch_algo")
            # encode all rows
            enc_rows = []
            for _, r in st.session_state.batch.iterrows():
                vals = r.to_dict()
                enc_rows.append(encode_inputs_row(vals).reshape(-1))
            enc_arr = np.vstack(enc_rows)
            # scale
            scaler_to_use = safe_joblib_load("sc_uploaded.pkl") if uploaded_sc else (safe_joblib_load("sc.pkl") or scaler)
            if scaler_to_use is None:
                st.error("No scaler available to transform batch.")
            else:
                try:
                    enc_scaled = scaler_to_use.transform(enc_arr)
                    model_path = "km_uploaded.pkl" if uploaded_km else "km.pkl" if alg == "KMeans" else ("mean_uploaded.pkl" if uploaded_ms else "mean.pkl")
                    model = safe_joblib_load(model_path)
                    if model is None:
                        # fit on the fly if we can
                        if X_scaled.shape[0] > 0:
                            model = KMeans(n_clusters=3, random_state=3).fit(X_scaled) if alg == "KMeans" else MeanShift(bandwidth=2.6).fit(X_scaled)
                            st.info(f"Fitted {alg} on-the-fly for batch prediction.")
                        else:
                            st.error("No model available and cannot fit (no dataset).")
                            model = None
                    if model is not None:
                        preds = model.predict(enc_scaled)
                        st.session_state.batch["Predicted_Cluster"] = preds.astype(int)
                        # append to history
                        hist_rows = st.session_state.batch.copy()
                        hist_rows["Algorithm"] = alg
                        # ensure Predicted_Cluster exists as int
                        hist_rows["Predicted_Cluster"] = hist_rows["Predicted_Cluster"].astype(int)
                        st.session_state.history = pd.concat([st.session_state.history, hist_rows], ignore_index=True)
                        st.success("Batch predicted and added to history.")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

# ---------------- History & Download ----------------
st.markdown("---")
st.subheader("📜 Prediction History")
if st.session_state.history.empty:
    st.info("No predictions yet. Make single or batch predictions to see history.")
else:
    st.dataframe(st.session_state.history)
    csv = st.session_state.history.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download history as CSV", data=csv, file_name="predictions_history.csv", mime="text/csv")

st.markdown("---")
st.caption("This app is robust: if saved models/scaler are missing it will (a) warn you, (b) allow uploading .pkl files, or (c) fit clustering models on the provided dataset as a fallback. For 3D visualization the app uses PCA to project features to three dimensions.")
