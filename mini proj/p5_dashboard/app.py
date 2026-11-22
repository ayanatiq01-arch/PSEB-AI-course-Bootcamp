# app.py: live training dashboard.  Run:  streamlit run app.py
import pandas as pd
import streamlit as st
import trainer

COLORS = ["#E8710A", "#1A73E8"]            # training = orange, validation = blue
st.set_page_config(page_title="Live training dashboard", layout="wide")
st.title("Live training dashboard · City Hospital")
st.caption("Breast cancer screening model: watch training and validation loss, fix the curves, count the cancers found.")

if "runs" not in st.session_state:
    st.session_state.runs = []             # the experiments table survives button clicks
if "model" not in st.session_state:
    st.session_state.model = None

# ---- 1 · Sidebar: settings ----
with st.sidebar:
    st.header("Settings")
    size = st.select_slider("Hidden neurons", options=[4, 8, 16, 32, 64, 128, 256], value=256)
    dropout = st.slider("Dropout", 0.0, 0.7, 0.0, 0.1)
    decay = st.select_slider("Weight decay", options=[0.0, 0.001, 0.01, 0.05, 0.1], value=0.0)
    lr = st.select_slider("Learning rate", options=[0.0001, 0.001, 0.01, 0.1], value=0.001)
    early = st.checkbox("Early stopping", value=False)
    patience = st.slider("Patience (epochs)", 5, 50, 20, disabled=not early)
    scale = st.checkbox("Scale the measurements", value=True)
    epochs = st.slider("Max epochs", 50, 400, 300, 10)
    note = st.text_input("Run name (optional)", "")
    go = st.button("Train", type="primary")

cfg = {"size": size, "dropout": dropout, "decay": decay, "lr": lr,
       "patience": patience if early else 0, "scale": scale, "epochs": epochs}

# ---- 2 · Train, redrawing the chart every 5 epochs ----
st.subheader("Training and validation loss")
chart, bar = st.empty(), st.progress(0.0)


def redraw(epoch, hist):                   # the trainer calls this after every epoch
    bar.progress((epoch + 1) / cfg["epochs"])
    if epoch % 5 == 0:
        chart.line_chart(pd.DataFrame({"training loss": hist["train"], "validation loss": hist["val"]}), color=COLORS)


if go:
    data = trainer.load_data(scale=scale)
    model, hist = trainer.train(cfg, data, on_epoch=redraw)
    bar.progress(1.0)
    chart.line_chart(pd.DataFrame({"training loss": hist["train"], "validation loss": hist["val"]}), color=COLORS)
    verdict = trainer.diagnose(hist)                       # ---- 3 · diagnosis ----
    val = trainer.counts(model, data["X_va"], data["y_va"])
    best_ep = hist["val"].index(min(hist["val"])) + 1
    st.session_state.model, st.session_state.data = model, data
    st.session_state.last = {"cfg": cfg, "hist": hist, "verdict": verdict, "val": val, "best_ep": best_ep}
    st.session_state.runs.append({
        "run": note or f"run {len(st.session_state.runs) + 1}", **cfg, "diagnosis": verdict,
        "epochs_run": len(hist["train"]), "best_val_loss": round(min(hist["val"]), 3),
        "cancers_found": f'{val["cancers_found"]} of {val["cancers_total"]}', "false_alarms": val["false_alarms"]})
elif "last" in st.session_state:
    h = st.session_state.last["hist"]
    chart.line_chart(pd.DataFrame({"training loss": h["train"], "validation loss": h["val"]}), color=COLORS)
    bar.progress(1.0)

if "last" in st.session_state:
    L = st.session_state.last
    icon = {"healthy": "✅", "overfitting": "⚠️", "underfitting": "⚠️"}.get(L["verdict"], "")
    msg = {"healthy": "Healthy: both losses fall together.",
           "overfitting": "Overfitting: validation loss rose well above its best. Try a smaller model, dropout, weight decay or early stopping.",
           "underfitting": "Underfitting: training loss is still high. Train longer, scale the data or use a bigger model."}[L["verdict"]]
    (st.success if L["verdict"] == "healthy" else st.warning)(f"{icon} {msg}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Epochs run", len(L["hist"]["train"]))
    c2.metric("Best validation loss", f'{min(L["hist"]["val"]):.3f}', f'epoch {L["best_ep"]}', delta_color="off")
    c3.metric("Cancers found (validation)", f'{L["val"]["cancers_found"]} of {L["val"]["cancers_total"]}')
    c4.metric("False alarms", L["val"]["false_alarms"])

# ---- 4 · every run, downloadable ----
st.subheader("Experiments")
if st.session_state.runs:
    table = pd.DataFrame(st.session_state.runs)
    st.dataframe(table, hide_index=True)
    st.download_button("Download experiments as CSV", table.to_csv(index=False), "experiments.csv", "text/csv")
else:
    st.info("No runs yet. Change a setting in the sidebar and press Train. Change one thing at a time.")

# ---- 5 · check a patient + one-time final test ----
if st.session_state.model is not None:
    model, data = st.session_state.model, st.session_state.data
    st.subheader("Check a patient the model never saw")
    i = st.slider("Test patient", 0, len(data["X_te"]) - 1, 0)
    import torch
    with torch.no_grad():
        p = torch.softmax(model(data["X_te"][i:i + 1]), 1)[0, 1].item()
    truth = "malignant" if int(data["y_te"][i]) == 1 else "benign"
    st.write(f"Model: **{'malignant' if p >= 0.5 else 'benign'}** (probability of malignant {p:.2f}) · True label: **{truth}**")
    st.caption("A doctor must review every result: this model is not cleared for real patients.")

    st.subheader("Final test (run once)")
    if st.checkbox("I have chosen my settings: run the final test now"):
        r = trainer.counts(model, data["X_te"], data["y_te"])
        st.write(f'Final test: accuracy **{r["accuracy"]}** · **{r["cancers_found"]} of {r["cancers_total"]}** cancers found · '
                 f'**{r["false_alarms"]}** false alarms')
