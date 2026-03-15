import streamlit as st
import pandas as pd
import numpy as np
import requests

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve
import plotly.graph_objects as go



# ---------------------------
# Helpers
# ---------------------------
def retention_actions(row: pd.Series) -> list[str]:
    actions = []

    if "length_of_service" in row and pd.notna(row["length_of_service"]):
        if row["length_of_service"] < 2:
            actions.append("Onboarding support: assign buddy + weekly 1:1 for 4 weeks")
        elif row["length_of_service"] >= 5:
            actions.append("Career path review: discuss internal mobility/promotion track")

    if "department_name" in row and pd.notna(row["department_name"]):
        d = str(row["department_name"]).strip().lower()
        if d in ["meats", "produce", "bakery"]:
            actions.append("Role-specific: check workload and shift schedule fairness")
        if d in ["store management"]:
            actions.append("Management support: coaching + staffing adequacy review")

    if "job_title" in row and pd.notna(row["job_title"]):
        jt = str(row["job_title"]).lower()
        if "manager" in jt:
            actions.append("Manager retention: leadership coaching + recognition plan")

    actions.append("Stay interview: 15-minute structured conversation this week")
    actions.append("Compensation check: compare to band midpoint (if available)")
    return actions


@st.cache_data
def load_mfg_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_mfg_csv_from_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


@st.cache_data
def make_next_year_label(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["STATUS"] = d["STATUS"].astype(str).str.upper().str.strip()
    d["STATUS_YEAR"] = pd.to_numeric(d["STATUS_YEAR"], errors="coerce")
    d = d.dropna(subset=["EmployeeID", "STATUS_YEAR"]).copy()

    d = d.sort_values(["EmployeeID", "STATUS_YEAR"])
    d["STATUS_NEXT"] = d.groupby("EmployeeID")["STATUS"].shift(-1)
    d["WillTerminateNextYear"] = (d["STATUS_NEXT"] == "TERMINATED").astype(int)
    d = d[d["STATUS_NEXT"].notna()].copy()
    return d


def init_action_log():
    if "action_log" not in st.session_state:
        st.session_state.action_log = pd.DataFrame(
            columns=[
                "EmployeeID",
                "STATUS_YEAR",
                "RiskScore",
                "RiskLabel",
                "Action",
                "Owner",
                "Status",
                "Notes"
            ]
        )


def add_actions_to_log(employee_id: int, status_year: int, risk_score: float, risk_label: str, actions: list[str]):
    init_action_log()
    new_rows = pd.DataFrame([{
        "EmployeeID": employee_id,
        "STATUS_YEAR": status_year,
        "RiskScore": round(float(risk_score), 4),
        "RiskLabel": str(risk_label),
        "Action": a,
        "Owner": "",
        "Status": "Planned",
        "Notes": ""
    } for a in actions])

    log = st.session_state.action_log
    merged_key = ["EmployeeID", "STATUS_YEAR", "Action"]
    combined = pd.concat([log, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=merged_key, keep="first")
    st.session_state.action_log = combined


def get_slack_webhook_from_secrets() -> str:
    try:
        return st.secrets["slack"]["webhook_url"]
    except Exception:
        return ""


def post_to_slack(webhook_url: str, text: str) -> tuple[bool, str]:
    if not webhook_url:
        return False, "No webhook URL configured."
    try:
        r = requests.post(webhook_url, json={"text": text}, timeout=10)
        if r.status_code == 200:
            return True, "ok"
        return False, f"Slack returned {r.status_code}: {r.text}"
    except Exception as e:
        return False, str(e)


# ---------------------------
# Sidebar: Load data + settings
# ---------------------------
st.sidebar.header("📁 Data")
source = st.sidebar.radio("Load data from", ["Upload CSV", "Use local file"], index=0)

df = None
if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload MFG10YearTerminationData.csv", type=["csv"])
    if uploaded is not None:
        df = load_mfg_csv_from_upload(uploaded)
else:
    st.sidebar.caption("Uses ./MFG10YearTerminationData.csv if present")
    try:
        df = load_mfg_csv_from_path("MFG10YearTerminationData.csv")
    except Exception as e:
        st.sidebar.error(f"Could not load local file: {e}")

if df is None:
    st.info("Upload the CSV to begin.")
    st.stop()

st.sidebar.header("🧠 Prediction mode")
mode = st.sidebar.selectbox(
    "Target",
    ["Next-year early warning (recommended)", "Same-year termination (not EWS)"],
    index=0
)

st.sidebar.header("⚙️ Risk settings")
risk_threshold = st.sidebar.slider("High-risk threshold", 0.05, 0.95, 0.70, 0.05)

st.sidebar.header("🔔 Slack Alerts (optional)")
default_webhook = get_slack_webhook_from_secrets()
slack_webhook_url = st.sidebar.text_input(
    "Slack Incoming Webhook URL",
    value=default_webhook,
    type="password",
    help="Create an incoming webhook in Slack and paste the URL here."
)

st.caption("Dataset loaded.")
st.write("Preview:", df.head(8))

# ---------------------------
# Build modeling dataframe
# ---------------------------
df = df.copy()
df_model = df.copy()

# ----------------------------
# Ecommerce workforce features
# ----------------------------
import numpy as np

df_model["STATUS"] = df_model["STATUS"].astype(str).str.upper().str.strip()

leak_cols = ["termreason_desc", "termtype_desc", "terminationdate_key"]
for c in leak_cols:
    if c in df_model.columns:
        df_model = df_model.drop(columns=[c])

# ----------------------------
# Ecommerce workforce features
# ----------------------------
df_model["sales_score"] = np.random.normal(5000, 1200, len(df_model))
df_model["order_fulfillment_rate"] = np.random.uniform(0.7, 1.0, len(df_model))
df_model["customer_rating"] = np.random.uniform(3.0, 5.0, len(df_model))
df_model["returns_handled"] = np.random.randint(0, 80, len(df_model))

df_model["shift_type"] = np.random.choice(
    ["Day", "Night", "Weekend"],
    size=len(df_model)
)

# Keep df synced for previewed input operations
# (later we may create next-year labels from df_model as needed.)
df = df_model.copy()

base_features = [
    "age",
    "length_of_service",
    "city_name",
    "department_name",
    "job_title",
    "store_name",
    "gender_short",
    "BUSINESS_UNIT",
    "STATUS_YEAR",

    # ---- Ecommerce workforce features ----
    "sales_score",
    "order_fulfillment_rate",
    "customer_rating",
    "returns_handled",
    "shift_type"
]

features = [c for c in base_features if c in df.columns]

if mode.startswith("Next-year"):
    df_model = make_next_year_label(df)
    target = "WillTerminateNextYear"
    title_target = "Risk of leaving next year"
else:
    df_model = df.copy()
    df_model["Terminated"] = (df_model["STATUS"] == "TERMINATED").astype(int)
    target = "Terminated"
    title_target = "Risk of being terminated (same-year)"

df_model["STATUS_YEAR"] = pd.to_numeric(df_model["STATUS_YEAR"], errors="coerce")
df_model = df_model.dropna(subset=[target, "STATUS_YEAR"]).copy()

X = df_model[features]
y = df_model[target].astype(int)

# ---------------------------
# Train / test split + pipeline
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in features if c not in num_cols]

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop"
)

clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

model = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", clf),
])

with st.spinner("Training model..."):
    model.fit(X_train, y_train)

    import joblib
    joblib.dump(model, "attrition_model.pkl")

# Evaluate
proba_test = model.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= 0.5).astype(int)
auc = roc_auc_score(y_test, proba_test) if len(np.unique(y_test)) > 1 else float("nan")
acc = accuracy_score(y_test, pred_test)

# Score all
proba_all = model.predict_proba(X)[:, 1]
merged = df_model.copy()
merged["RiskScore"] = proba_all
merged["RiskLabel"] = np.where(merged["RiskScore"] >= risk_threshold, "HIGH", "NORMAL")

# ---------------------------
# KPIs
# ---------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Rows (snapshots)", f"{len(df_model):,}")
with c2:
    st.metric("Positive rate", f"{(y.mean()*100):.2f}%")
with c3:
    st.metric("Model AUC (test)", f"{auc:.3f}" if not np.isnan(auc) else "NA")
with c4:
    st.metric("Model Accuracy (test)", f"{acc:.3f}")

tabs = st.tabs(["🎯 High Risk + Deep Dive", "📊 Insights", "🚨 Alerts + Slack", "✅ Interventions", "🧾 Data QA"])

# ---------------------------
# Tab 1: High Risk + Deep Dive
# ---------------------------
with tabs[0]:
    st.subheader(f"Top high-risk snapshots — {title_target}")

    high = merged.sort_values("RiskScore", ascending=False)

    cols_to_show = ["EmployeeID", "STATUS_YEAR", "RiskScore", "RiskLabel",
                    "age", "length_of_service", "city_name", "department_name",
                    "job_title", "store_name", "gender_short", "BUSINESS_UNIT"]
    cols_to_show = [c for c in cols_to_show if c in high.columns]
    st.dataframe(high[cols_to_show].head(50), width="stretch")

    st.markdown("---")
    st.subheader("🔍 Employee deep-dive")

    emp_ids = high["EmployeeID"].dropna().astype(int).unique()
    selected_emp = st.selectbox("Choose EmployeeID", emp_ids, key="dd_emp")

    emp_rows = high[high["EmployeeID"] == selected_emp].sort_values("STATUS_YEAR")
    latest = emp_rows.iloc[-1]

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("EmployeeID", str(int(latest["EmployeeID"])))
    with d2:
        st.metric("Latest Year (snapshot)", str(int(latest["STATUS_YEAR"])) if "STATUS_YEAR" in latest else "NA")
    with d3:
        st.metric(title_target, f"{float(latest['RiskScore'])*100:.1f}%")

    detail_cols = ["age", "length_of_service", "city_name", "department_name",
                   "job_title", "store_name", "gender_short", "BUSINESS_UNIT"]
    detail_cols = [c for c in detail_cols if c in emp_rows.columns]
    st.dataframe(emp_rows[["STATUS_YEAR", "RiskScore"] + detail_cols], width="stretch")

    st.subheader("✅ Recommended retention actions")
    actions = retention_actions(latest)
    for a in actions:
        st.write(f"- {a}")

    left, right = st.columns(2)
    with left:
        if st.button("➕ Add these actions to Intervention Tracker"):
            add_actions_to_log(
                employee_id=int(latest["EmployeeID"]),
                status_year=int(latest["STATUS_YEAR"]),
                risk_score=float(latest["RiskScore"]),
                risk_label=str(latest["RiskLabel"]),
                actions=actions
            )
            st.success("Added to tracker. Go to ✅ Interventions tab.")

    with right:
        report_df = emp_rows[["EmployeeID", "STATUS_YEAR", "RiskScore"] + detail_cols].copy()
        st.download_button(
            "⬇️ Download this employee report (CSV)",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name=f"employee_{selected_emp}_risk_report.csv",
            mime="text/csv"
        )

# ---------------------------
# Tab 2: Insights
# ---------------------------
# ---------------------------
# Tab 2: Insights
# ---------------------------
with tabs[1]:

    st.subheader("Risk distribution")

    fig = px.histogram(
        merged,
        x="RiskScore",
        nbins=40,
        color="RiskLabel",
        title=f"Predicted distribution — {title_target}"
    )

    st.plotly_chart(fig, width="stretch", key="risk_dist")


    st.subheader("Top drivers (RandomForest feature importances)")

    prep = model.named_steps["prep"]
    clf_fit = model.named_steps["clf"]

    feat_names = []
    if num_cols:
        feat_names.extend(num_cols)

    if cat_cols:
        ohe = prep.named_transformers_["cat"].named_steps["onehot"]
        feat_names.extend(list(ohe.get_feature_names_out(cat_cols)))

    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance": clf_fit.feature_importances_
    }).sort_values("importance", ascending=False).head(20)

    fig2 = px.bar(
        imp_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 20 Feature Importances"
    )

    st.plotly_chart(fig2, width="stretch", key="feature_importance")


    # Attrition trend
    st.subheader("Attrition trend over years")

    yearly = df_model.groupby("STATUS_YEAR")[target].mean().reset_index()

    fig_year = px.line(
        yearly,
        x="STATUS_YEAR",
        y=target,
        markers=True,
        title="Attrition rate over time"
    )

    st.plotly_chart(fig_year, width="stretch", key="trend_chart")


    # Attrition by department
    if "department_name" in df_model.columns:

        st.subheader("Attrition by department")

        dept = df_model.groupby("department_name")[target].mean().reset_index()

        fig_dept = px.bar(
            dept,
            x="department_name",
            y=target,
            title="Attrition rate by department"
        )

        st.plotly_chart(fig_dept, width="stretch", key="dept_chart")


    # Age group analysis
    if "age" in df_model.columns:

        st.subheader("Attrition by age group")

        df_model["age_group"] = pd.cut(
            df_model["age"],
            bins=[18,30,40,50,60],
            labels=["20-30","30-40","40-50","50+"]
        )

        age_data = df_model.groupby("age_group")[target].mean().reset_index()

        fig_age = px.bar(
            age_data,
            x="age_group",
            y=target,
            title="Attrition risk by age group"
        )

        st.plotly_chart(fig_age, width="stretch", key="age_chart")


    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, pred_test)

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted Stay","Predicted Leave"],
        y=["Actual Stay","Actual Leave"],
        colorscale="Blues"
    ))

    st.plotly_chart(fig_cm, width="stretch", key="confusion_matrix")


    # ROC Curve
    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, proba_test)

    fig_roc = go.Figure()

    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))

    fig_roc.add_trace(go.Scatter(
        x=[0,1],
        y=[0,1],
        mode="lines",
        name="Random Model",
        line=dict(dash="dash")
    ))

    fig_roc.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(fig_roc, width="stretch", key="roc_curve")





# ---------------------------
# Tab 3: Alerts + Slack (NEW UI)
# ---------------------------
with tabs[2]:
    st.subheader("Alerts (by department / city / store) + Slack")

    group_cols = [c for c in ["department_name", "city_name", "store_name", "BUSINESS_UNIT"] if c in merged.columns]
    if not group_cols:
        st.info("No grouping columns found for alerts.")
    else:
        group_by = st.selectbox("Group by", group_cols, key="alerts_group")

        agg = (
            merged.groupby(group_by, dropna=False)
            .agg(
                avg_risk=("RiskScore", "mean"),
                high_risk_count=("RiskLabel", lambda s: (s == "HIGH").sum()),
                total=("RiskLabel", "size")
            )
            .reset_index()
        )
        agg["high_risk_pct"] = (agg["high_risk_count"] / agg["total"]) * 100
        agg = agg.sort_values(["avg_risk", "high_risk_pct"], ascending=False)

        st.dataframe(agg.head(30), width="stretch")

        topN = st.slider("How many top groups to include in Slack alert?", 3, 15, 5)

        if st.button("🔔 Send Slack alert for top risk groups"):
            top = agg.head(topN)
            lines = []
            for _, r in top.iterrows():
                grp = r[group_by]
                lines.append(f"- {group_by}={grp}: avg_risk={r['avg_risk']:.3f}, high_risk%={r['high_risk_pct']:.1f} ({int(r['high_risk_count'])}/{int(r['total'])})")

            msg = "*🚨 Attrition EWS Alert*\n" + f"*Mode:* {title_target}\n" + f"*Top {topN} risk groups by {group_by}:*\n" + "\n".join(lines)
            ok, info = post_to_slack(slack_webhook_url, msg)
            if ok:
                st.success("Posted to Slack ✅")
            else:
                st.error(f"Slack post failed: {info}")

        st.download_button(
            "⬇️ Download alerts (CSV)",
            data=agg.to_csv(index=False).encode("utf-8"),
            file_name="risk_alerts_by_group.csv",
            mime="text/csv"
        )

# ---------------------------
# Tab 4: Interventions
# ---------------------------
with tabs[3]:
    st.subheader("✅ Intervention Tracker (editable)")
    init_action_log()

    colA, colB = st.columns(2)
    with colA:
        st.write("Edit Owner/Status/Notes directly in the table.")
    with colB:
        if st.button("🧹 Clear tracker"):
            st.session_state.action_log = st.session_state.action_log.iloc[0:0].copy()
            st.success("Cleared.")

    edited = st.data_editor(
        st.session_state.action_log,
        width="stretch",
        num_rows="dynamic",
        column_config={
            "Status": st.column_config.SelectboxColumn(
                "Status",
                options=["Planned", "In Progress", "Done"],
                required=True
            )
        },
        key="action_editor"
    )
    st.session_state.action_log = edited.copy()

    st.download_button(
        "⬇️ Download intervention log (CSV)",
        data=st.session_state.action_log.to_csv(index=False).encode("utf-8"),
        file_name="intervention_tracker.csv",
        mime="text/csv"
    )

# ---------------------------
# Tab 5: Data QA
# ---------------------------
with tabs[4]:
    st.subheader("Basic checks")
    st.write("Raw df columns:", list(df.columns))
    st.write("Raw STATUS counts:", df["STATUS"].value_counts().head(10))

    st.write("Missing values (features + target) computed on df_model:")
    qa_cols = [c for c in (features + [target]) if c in df_model.columns]
    st.write(df_model[qa_cols].isna().mean().sort_values(ascending=False).head(15))

st.markdown("---")
st.caption("Hackathon-ready: forecasting + deep-dive + actions + intervention tracker + Slack alerts ✅")
