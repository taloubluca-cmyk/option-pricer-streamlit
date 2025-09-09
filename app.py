
# app.py
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="Option Pricer on Futures", layout="wide")

# =========================
# Black on Futures (pricing & greeks)
# =========================
def _tau(t, T):
    return max(T - t, 0.0)

def _d1(Ft, K, t, T, sigma):
    tau = _tau(t, T)
    if tau == 0 or sigma == 0 or Ft <= 0 or K <= 0:
        return np.inf
    return (np.log(Ft / K) + 0.5 * sigma * sigma * tau) / (sigma * np.sqrt(tau))

def _d2(d1, t, T, sigma):
    tau = _tau(t, T)
    return d1 - sigma * np.sqrt(tau) if np.isfinite(d1) else np.inf

def price_call_on_future(Ft, K, t, T, r, sigma):
    tau = _tau(t, T)
    if tau == 0:
        return max(Ft - K, 0.0)
    d1 = _d1(Ft, K, t, T, sigma); d2 = _d2(d1, t, T, sigma)
    return np.exp(-r * tau) * (Ft * norm.cdf(d1) - K * norm.cdf(d2))

def price_put_on_future(Ft, K, t, T, r, sigma):
    tau = _tau(t, T)
    if tau == 0:
        return max(K - Ft, 0.0)
    d1 = _d1(Ft, K, t, T, sigma); d2 = _d2(d1, t, T, sigma)
    return np.exp(-r * tau) * (K * norm.cdf(-d2) - Ft * norm.cdf(-d1))

def delta_call_on_future(Ft, K, t, T, r, sigma):
    tau = _tau(t, T)
    if tau == 0:
        return 1.0 if Ft > K else 0.0
    d1 = _d1(Ft, K, t, T, sigma)
    return np.exp(-r * tau) * norm.cdf(d1)

def delta_put_on_future(Ft, K, t, T, r, sigma):
    tau = _tau(t, T)
    if tau == 0:
        return -1.0 if Ft < K else 0.0
    d1 = _d1(Ft, K, t, T, sigma)
    return -np.exp(-r * tau) * norm.cdf(-d1)

def vega_on_future(Ft, K, t, T, r, sigma):
    tau = _tau(t, T)
    if tau == 0 or sigma == 0:
        return 0.0
    d1 = _d1(Ft, K, t, T, sigma)
    return np.exp(-r * tau) * Ft * np.sqrt(tau) * norm.pdf(d1)

def gamma_on_future(Ft, K, t, T, r, sigma):
    tau = _tau(t, T)
    if tau == 0 or sigma == 0 or Ft <= 0:
        return 0.0
    d1 = _d1(Ft, K, t, T, sigma)
    return np.exp(-r * tau) * norm.pdf(d1) / (Ft * sigma * np.sqrt(tau))

def theta_call_on_future(Ft, K, t, T, r, sigma):
    tau = _tau(t, T)
    if tau == 0:
        return 0.0
    d1 = _d1(Ft, K, t, T, sigma)
    return -r * price_call_on_future(Ft, K, t, T, r, sigma) + np.exp(-r * tau) * Ft * norm.pdf(d1) * sigma / (2 * np.sqrt(tau))

def theta_put_on_future(Ft, K, t, T, r, sigma):
    tau = _tau(t, T)
    if tau == 0:
        return 0.0
    d1 = _d1(Ft, K, t, T, sigma)
    return -r * price_put_on_future(Ft, K, t, T, r, sigma) + np.exp(-r * tau) * Ft * norm.pdf(d1) * sigma / (2 * np.sqrt(tau))

def rho_call_on_future(Ft, K, t, T, r, sigma):
    tau = _tau(t, T)
    return -tau * price_call_on_future(Ft, K, t, T, r, sigma)

def rho_put_on_future(Ft, K, t, T, r, sigma):
    tau = _tau(t, T)
    return -tau * price_put_on_future(Ft, K, t, T, r, sigma)

def vanna_on_future(Ft, K, t, T, r, sigma):
    tau = _tau(t, T)
    if tau == 0 or sigma == 0:
        return 0.0
    d1 = _d1(Ft, K, t, T, sigma); d2 = _d2(d1, t, T, sigma)
    return -np.exp(-r * tau) * norm.pdf(d1) * d2 / sigma

def vomma_on_future(Ft, K, t, T, r, sigma):
    tau = _tau(t, T)
    if tau == 0 or sigma == 0:
        return 0.0
    d1 = _d1(Ft, K, t, T, sigma); d2 = _d2(d1, t, T, sigma)
    return vega_on_future(Ft, K, t, T, r, sigma) * d1 * d2 / sigma

def price_fn(opt_type):
    return price_call_on_future if opt_type == "Call" else price_put_on_future

def delta_fn(opt_type):
    return delta_call_on_future if opt_type == "Call" else delta_put_on_future

def theta_fn(opt_type):
    return theta_call_on_future if opt_type == "Call" else theta_put_on_future

def rho_fn(opt_type):
    return rho_call_on_future if opt_type == "Call" else rho_put_on_future

def greek_value(opt, Ft, which):
    K, t, T, r, sigma = opt["K"], opt["t"], opt["T"], opt["r"], opt["sigma"]
    tp = opt["type"]
    if which == "Price":  return price_fn(tp)(Ft, K, t, T, r, sigma)
    if which == "Delta":  return delta_fn(tp)(Ft, K, t, T, r, sigma)
    if which == "Gamma":  return gamma_on_future(Ft, K, t, T, r, sigma)
    if which == "Vega":   return vega_on_future(Ft, K, t, T, r, sigma)
    if which == "Theta":  return theta_fn(tp)(Ft, K, t, T, r, sigma)
    if which == "Rho":    return rho_fn(tp)(Ft, K, t, T, r, sigma)
    if which == "Vanna":  return vanna_on_future(Ft, K, t, T, r, sigma)
    if which == "Vomma":  return vomma_on_future(Ft, K, t, T, r, sigma)
    raise ValueError("Unknown metric")

def payoff_at_maturity(opt, FT):
    K, tp = opt["K"], opt["type"]
    return np.maximum(FT - K, 0.0) if tp == "Call" else np.maximum(K - FT, 0.0)

def signed(val, side, qty):
    return (1.0 if side == "Buy" else -1.0) * qty * val

# =========================
# UI (FR)
# =========================
st.title("📈 Option Pricer on Futures — Courbe unique (interactive)")

with st.sidebar:
    st.header("Paramètres globaux")
    Ft0 = st.number_input("Prix futur courant Ft", min_value=0.0001, value=5000.0, step=50.0, format="%.6f")
    r_global = st.number_input("Taux sans risque r (annuel)", value=0.05, step=0.005, format="%.6f")
    t_now = st.number_input("Temps courant t (années)", value=0.0, step=0.01, format="%.6f")

    st.markdown("---")
    st.subheader("Affichage")
    # >>> ICI: un seul sélecteur qui inclut Payoff <<<
    metric = st.selectbox("Courbe à afficher", ["Price","Delta","Gamma","Vega","Theta","Rho","Vanna","Vomma","Payoff"], index=0)

    st.subheader("Réglages du plot")
    pct = st.number_input("Auto range autour de Ft (±%)", min_value=1.0, max_value=90.0, value=30.0, step=1.0)
    use_manual = st.checkbox("Forcer une plage manuelle", value=False)
    if use_manual:
        x_min = st.number_input("Ft min", value=max(0.0001, Ft0 * (1 - pct / 100.0)), step=10.0, format="%.6f")
        x_max = st.number_input("Ft max", value=Ft0 * (1 + pct / 100.0), step=10.0, format="%.6f")
    else:
        x_min = Ft0 * (1 - pct / 100.0)
        x_max = Ft0 * (1 + pct / 100.0)
        st.caption(f"Auto range: [{x_min:.2f}, {x_max:.2f}] (±{pct:.0f}% autour de Ft)")

    n_pts = st.slider("Nombre de points", min_value=50, max_value=1000, value=400, step=10)

st.markdown("### Options")
n = st.number_input("Nombre d'options", min_value=1, max_value=20, value=2, step=1)

options = []
# --- dans la boucle des options ---
for i in range(int(n)):
    with st.expander(f"Option #{i+1}", expanded=(i == 0)):
        c1, c2, c3, c4 = st.columns(4)
        tp   = c1.selectbox("Type", ["Call", "Put"], key=f"type_{i}")
        side = c2.selectbox("Sens", ["Buy", "Sell"], key=f"side_{i}")
        qty  = c3.number_input("Quantité (lots)", value=1.0, step=1.0, key=f"qty_{i}")
        sigma= c4.number_input("Volatilité sigma (annuelle)", value=0.30, step=0.01, format="%.6f", key=f"sigma_{i}")

        c5, c6, c7 = st.columns(3)
        K = c5.number_input("Strike K", value=5000.0, step=50.0, key=f"K_{i}")
        # --- MODIF: T en jours ---
        T_days = c6.number_input("Maturité T (jours)", value=180, step=5, key=f"Tdays_{i}")
        T = T_days / 365.0  # conversion en années
        r_i = c7.number_input("Taux sans risque r (override)", value=r_global, step=0.005, format="%.6f", key=f"r_{i}")

        options.append({"type": tp, "side": side, "qty": qty, "K": K, "t": t_now, "T": T, "r": r_i, "sigma": sigma})


# =========================
# Grille de Ft
# =========================
Ft_grid = np.linspace(x_min, x_max, int(n_pts))

# =========================
# Tableau snapshot @ Ft0 (signé × qty)
# =========================
metrics = ["Price","Delta","Gamma","Vega","Theta","Rho","Vanna","Vomma"]  # on garde Payoff hors snapshot
rows = []
for i, opt in enumerate(options, start=1):
    row = {"Option": f"#{i} ({opt['side']} {opt['type']})", "K": opt["K"], "T": opt["T"], "σ": opt["sigma"], "Qty": opt["qty"]}
    for m in metrics:
        row[m] = signed(greek_value(opt, Ft0, m), opt["side"], opt["qty"])
    rows.append(row)

df_options = pd.DataFrame(rows)
st.markdown("### Snapshot à Ft courant (signé × quantité)")
st.dataframe(
    df_options.style.format({
        "K": "{:.4f}", "T": "{:.4f}", "σ": "{:.6f}", "Qty": "{:.2f}",
        **{m: "{:.6f}" for m in metrics}
    }),
    use_container_width=True,
    hide_index=True
)

portfolio_snapshot = {m: float(np.sum([signed(greek_value(opt, Ft0, m), opt["side"], opt["qty"]) for opt in options]))
                      for m in metrics}
df_port = pd.DataFrame([portfolio_snapshot], index=["Portfolio (signé)"])
st.dataframe(df_port.style.format({m: "{:.6f}" for m in metrics}), use_container_width=True)

# =========================
# Courbe unique: soit Price/Greek, soit Payoff
# =========================
def plot_selected_curve(metric_name):
    """
    Trace les courbes individuelles des options ainsi que la courbe du portefeuille,
    selon le metric choisi (Payoff ou Greek).
    """

    # Initialisation de la figure avec un double axe Y
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    y_port = np.zeros_like(Ft_grid)

    # --- Cas 1 : Payoff pur à maturité ---
    if metric_name == "Payoff":
        x_vals = Ft_grid  # FT à maturité sur la grille

        for i, opt in enumerate(options, start=1):
            y = signed(
                payoff_at_maturity(opt, x_vals),
                opt["side"],
                opt["qty"]
            )
            y_port += y

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y,
                    mode="lines",
                    name=f"Opt #{i} ({opt['side']} {opt['type']})"
                ),
                secondary_y=False
            )

        y_label_left = "Payoff (signé × qty)"
        x_label = "FT à maturité"

    # --- Cas 2 : Price ou Greeks ---
    else:
        x_vals = Ft_grid

        for i, opt in enumerate(options, start=1):
            y = np.array([
                signed(
                    greek_value(opt, Ft, metric_name),
                    opt["side"],
                    opt["qty"]
                )
                for Ft in x_vals
            ])
            y_port += y

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y,
                    mode="lines",
                    name=f"Opt #{i} ({opt['side']} {opt['type']})"
                ),
                secondary_y=False
            )

        y_label_left = f"{metric_name} (signé × qty)"
        x_label = "Ft"

    # --- Trace du portefeuille (axe secondaire) ---
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_port,
            mode="lines",
            name="Portfolio (signé)",
            line=dict(width=3)
        ),
        secondary_y=True
    )

    # --- Mise en forme du graphe ---
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        hovermode="x unified"
    )

    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label_left, secondary_y=False)
    fig.update_yaxes(title_text="Portfolio (signé × qty)", secondary_y=True)

    # --- Affichage dans Streamlit ---
    st.plotly_chart(fig, use_container_width=True)



st.markdown("### Courbe")
plot_selected_curve(metric)


