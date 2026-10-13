"""
Aplikasi Streamlit: Analisis Peramalan Permintaan (Forecasting)
Kasus Community Centre Aqua Fit - Bab 3 Operations Management.

Jalankan:
    pip install -r requirements.txt
    streamlit run app.py
"""
import re

import numpy as np
import pandas as pd
import streamlit as st

# ==================================================================
# 1. FUNGSI PERAMALAN
#    Setiap fungsi mengembalikan array panjang n+1: forecast untuk
#    periode 1..n+1 (indeks 0..n). NaN = forecast tidak tersedia.
# ==================================================================

def naive(d):
    f = np.full(len(d) + 1, np.nan)
    f[1:] = d                      # forecast t = aktual t-1
    return f


def sma(d, k):
    n = len(d)
    f = np.full(n + 1, np.nan)
    for t in range(k, n + 1):
        f[t] = np.mean(d[t - k:t])
    return f


def wma(d, weights):
    """weights: urut dari data paling baru ke paling lama."""
    k, n = len(weights), len(d)
    f = np.full(n + 1, np.nan)
    w = np.array(weights, dtype=float)
    for t in range(k, n + 1):
        f[t] = np.dot(w, d[t - k:t][::-1])
    return f


def exp_smoothing(d, alpha, start_period, init_forecast):
    """start_period: nomor periode (mulai 1) yang forecast-nya diberikan."""
    n = len(d)
    f = np.full(n + 1, np.nan)
    s = int(start_period) - 1
    if s < 0 or s > n:
        return f
    f[s] = init_forecast
    for t in range(s + 1, n + 1):
        f[t] = f[t - 1] + alpha * (d[t - 1] - f[t - 1])
    return f


def accuracy(actual, forecast, mask):
    """MAD, MSE, MAPE, bias pada periode dengan mask True. Error = aktual - forecast."""
    a = np.asarray(actual, dtype=float)[mask]
    f = np.asarray(forecast[:len(actual)], dtype=float)[mask]
    e = a - f
    return {
        "MAD": float(np.mean(np.abs(e))),
        "MSE": float(np.mean(e ** 2)),
        "MAPE (%)": float(np.mean(np.abs(e) / a) * 100),
        "Bias": float(np.mean(e)),
    }


def parse_weights(text):
    """'0.6, 0.3, 0.1' -> [0.6, 0.3, 0.1]; None jika tidak valid (pakai titik desimal)."""
    parts = [p for p in re.split(r"[;,\s]+", text.strip()) if p]
    try:
        w = [float(p) for p in parts]
    except ValueError:
        return None
    return w if len(w) >= 2 else None


def run_methods(demand, methods, p):
    """Hitung forecast semua metode terpilih. p = dict parameter."""
    out = {}
    if "Naive" in methods:
        out["Naive"] = naive(demand)
    if "SMA" in methods:
        out[f"SMA-{p['k']}"] = sma(demand, p["k"])
    if "WMA" in methods and p["weights"]:
        out[f"WMA-{len(p['weights'])}"] = wma(demand, p["weights"])
    if "Exponential Smoothing" in methods:
        out[f"ES (a={p['alpha']:.2f})"] = exp_smoothing(
            demand, p["alpha"], p["es_start"], p["es_init"])
    return out


def common_mask(demand, forecasts):
    """Periode yang punya forecast untuk SEMUA metode."""
    mask = np.ones(len(demand), dtype=bool)
    for f in forecasts.values():
        mask &= ~np.isnan(f[:len(demand)])
    return mask


def build_conclusion(acc, n_eval):
    """Kesimpulan otomatis seperti Bagian B.4-B.5 laporan."""
    best = {m: acc[m].idxmin() for m in ["MAD", "MSE", "MAPE (%)"]}
    lines = []
    if len(set(best.values())) == 1:
        lines.append(f"**{best['MAD']}** paling akurat menurut MAD, MSE, dan MAPE.")
    else:
        lines.append("Ukuran akurasi tidak sepakat: "
                     f"MAD -> {best['MAD']}, MSE -> {best['MSE']}, "
                     f"MAPE -> {best['MAPE (%)']}.")
    if (acc["Bias"] > 0).all():
        lines.append("Semua metode **under-forecast** (bias positif): "
                     "forecast cenderung lebih rendah dari demand aktual.")
    elif (acc["Bias"] < 0).all():
        lines.append("Semua metode **over-forecast** (bias negatif): "
                     "forecast cenderung lebih tinggi dari demand aktual.")
    if n_eval < 6:
        lines.append(f"Evaluasi hanya memakai {n_eval} periode, sehingga "
                     "kesimpulan belum kuat. Gunakan data lebih panjang.")
    return best["MAD"], lines


# ==================================================================
# 2. DATA PRESET (sesuai isi laporan)
# ==================================================================
PRESETS = {
    "Kasus Aqua Fit (Problem #2)": dict(
        labels=[f"Minggu {i}" for i in range(1, 7)],
        demand=[162, 158, 138, 190, 182, 177],
        next_label="Minggu 7", methods=["SMA", "WMA", "Exponential Smoothing"],
        k=2, weights="0.6, 0.3, 0.1", alpha=0.3, es_start=3, es_init=130.0),
    "Latihan 1: Bola Lampu (Problem #1)": dict(
        labels=["Jan", "Feb", "Mar", "Apr", "Mei"],
        demand=[255, 298, 357, 319, 360],
        next_label="Juni",
        methods=["Naive", "SMA", "WMA", "Exponential Smoothing"],
        k=3, weights="0.5, 0.3, 0.2", alpha=0.2, es_start=5, es_init=350.0),
    "Latihan 2: Gudang Pertanian (Problem #3)": dict(
        labels=["2015", "2016", "2017", "2018", "2019"],
        demand=[348, 372, 311, 371, 365],
        next_label="2020", methods=["Exponential Smoothing"],
        k=2, weights="0.5, 0.3, 0.2", alpha=0.4, es_start=1, es_init=360.0),
    "Data sendiri": dict(
        labels=[f"Periode {i}" for i in range(1, 7)],
        demand=[100, 110, 105, 120, 125, 130],
        next_label="Periode 7", methods=["SMA", "WMA", "Exponential Smoothing"],
        k=2, weights="0.6, 0.3, 0.1", alpha=0.3, es_start=3, es_init=100.0),
}

# Nilai yang tertulis di laporan (Bagian B) untuk uji kebenaran
EXPECTED_AQUA = {
    "SMA-2": dict(next=179.5, MAD=23.0),
    "WMA-3": dict(next=179.8, MAD=19.1),
    "ES (a=0.30)": dict(next=164.7, MAD=35.8),
}

# ==================================================================
# 3. ANTARMUKA
# ==================================================================
st.set_page_config(page_title="Forecasting Demand", layout="wide")
st.title("Analisis Peramalan Permintaan (Forecasting)")
st.caption("Berdasarkan Bab 3 Operations Management. Forecast adalah titik "
           "awal rantai: forecast -> aggregate planning/S&OP -> MPS -> MRP "
           "-> production order.")

tab1, tab2, tab3 = st.tabs(
    ["Forecast & Akurasi", "Bandingkan 2 Forecast (Latihan 3)",
     "Rumus & Kaitan ERP"])

# ------------------------------------------------------------------
# TAB 1
# ------------------------------------------------------------------
with tab1:
    preset_name = st.selectbox("Pilih kasus", list(PRESETS.keys()))
    ps = PRESETS[preset_name]
    key = preset_name  # kunci widget berubah saat preset berubah -> default ikut berubah

    st.sidebar.header("Parameter metode")
    methods = st.sidebar.multiselect(
        "Metode", ["Naive", "SMA", "WMA", "Exponential Smoothing"],
        default=ps["methods"], key=f"m_{key}")
    k = st.sidebar.number_input("SMA: jumlah periode", 2, 12, ps["k"],
                                key=f"k_{key}")
    w_text = st.sidebar.text_input(
        "WMA: bobot terbaru -> terlama (pisah koma, desimal pakai titik)", ps["weights"],
        key=f"w_{key}")
    alpha = st.sidebar.slider("ES: alpha", 0.01, 1.0, ps["alpha"], 0.01,
                              key=f"a_{key}")
    es_start = st.sidebar.number_input("ES: forecast awal ada pada periode ke-",
                                       1, 100, ps["es_start"], key=f"s_{key}")
    es_init = st.sidebar.number_input("ES: nilai forecast awal",
                                      value=ps["es_init"], key=f"i_{key}")

    st.subheader("1. Data demand historis (bisa diedit)")
    df0 = pd.DataFrame({"Periode": ps["labels"], "Demand": ps["demand"]})
    df = st.data_editor(df0, num_rows="dynamic", key=f"de_{key}")
    df = df.dropna().reset_index(drop=True)
    demand = df["Demand"].astype(float).to_numpy()
    labels = df["Periode"].astype(str).tolist()
    n = len(demand)

    if n < 3:
        st.warning("Isi minimal 3 periode data demand.")
        st.stop()

    weights = parse_weights(w_text)
    if "WMA" in methods:
        if weights is None:
            st.warning("Format bobot WMA tidak valid. Contoh: 0.6, 0.3, 0.1")
        elif abs(sum(weights) - 1) > 1e-6:
            st.warning(f"Jumlah bobot WMA = {sum(weights):.2f}, seharusnya 1.")

    params = dict(k=int(k), weights=weights, alpha=alpha,
                  es_start=es_start, es_init=es_init)
    forecasts = run_methods(demand, methods, params)
    if not forecasts:
        st.info("Pilih minimal satu metode di sidebar.")
        st.stop()

    next_label = (ps["next_label"] if n == len(ps["demand"])
                  else f"Periode ke-{n + 1}")

    table = pd.DataFrame({"Periode": labels + [next_label],
                          "Aktual": list(demand) + [np.nan]})
    for name, f in forecasts.items():
        table[name] = f
    table = table.set_index("Periode")

    st.subheader("2. Hasil forecast")
    st.dataframe(table.round(1))
    chart = table.copy()
    chart.index = range(1, n + 2)
    chart.index.name = "Urutan periode"
    st.line_chart(chart)

    st.subheader("3. Ukuran akurasi")
    mask = common_mask(demand, forecasts)
    if mask.sum() == 0:
        st.warning("Tidak ada periode yang punya forecast untuk semua metode "
                   "terpilih. Kurangi metode atau tambah data.")
        st.stop()

    eval_labels = [lb for lb, m in zip(labels, mask) if m]
    st.caption(f"Dihitung pada periode yang sama untuk semua metode: "
               f"{eval_labels[0]} s.d. {eval_labels[-1]} ({int(mask.sum())} "
               "periode). Error = Aktual - Forecast; bias positif = "
               "cenderung under-forecast.")
    acc = pd.DataFrame({nm: accuracy(demand, f, mask)
                        for nm, f in forecasts.items()}).T
    st.dataframe(acc.round(1))

    st.subheader("4. Kesimpulan")
    best, lines = build_conclusion(acc, int(mask.sum()))
    c1, c2 = st.columns(2)
    c1.metric("Metode terbaik (MAD terendah)", best)
    c2.metric(f"Forecast {next_label}", f"{forecasts[best][n]:.1f}")
    for ln in lines:
        st.markdown(f"- {ln}")

    if preset_name.startswith("Kasus Aqua Fit"):
        with st.expander("Uji kebenaran terhadap jawaban di laporan"):
            rows = []
            for nm, exp in EXPECTED_AQUA.items():
                if nm in forecasts:
                    got_next = round(float(forecasts[nm][n]), 1)
                    got_mad = round(float(acc.loc[nm, "MAD"]), 1)
                    rows.append({
                        "Metode": nm,
                        "Forecast minggu 7 (aplikasi)": got_next,
                        "Di laporan": exp["next"],
                        "MAD (aplikasi)": got_mad,
                        "MAD di laporan": exp["MAD"],
                        "Cocok": "Ya" if (got_next == exp["next"]
                                          and got_mad == exp["MAD"]) else "Tidak",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows))
            st.caption("Cocok hanya jika parameter sama dengan default kasus "
                       "(SMA 2 periode, WMA 0.6/0.3/0.1, ES alpha 0.3 dengan "
                       "forecast awal minggu 3 = 130).")

# ------------------------------------------------------------------
# TAB 2 - Latihan 3 (Problem #4): dua forecast sudah diberikan
# ------------------------------------------------------------------
with tab2:
    st.subheader("Latihan 3: Permintaan Rontgen")
    st.write("Dua metode forecast sudah diberikan. Bandingkan akurasinya.")
    ex = pd.DataFrame({"Minggu": [1, 2, 3, 4],
                       "Aktual": [48, 65, 58, 79],
                       "Forecast 1": [50, 55, 60, 70],
                       "Forecast 2": [50, 56, 55, 85]})
    ed = st.data_editor(ex, num_rows="dynamic", key="rontgen").dropna()
    if len(ed) >= 1:
        a = ed["Aktual"].to_numpy(dtype=float)
        mask2 = np.ones(len(a), dtype=bool)
        res = {nm: accuracy(a, ed[nm].to_numpy(dtype=float), mask2)
               for nm in ["Forecast 1", "Forecast 2"]}
        r = pd.DataFrame(res).T
        st.dataframe(r.round(2))
        b = r["MAD"].idxmin()
        st.success(f"{b} lebih akurat menurut MAD (MAD terendah).")
        st.caption("Pada data bawaan: MAD 5,75 vs 5,0; MSE 47,25 vs 32,5; "
                   "MAPE 8,6% vs 7,7%.")

# ------------------------------------------------------------------
# TAB 3 - Rumus dan ERP
# ------------------------------------------------------------------
with tab3:
    st.subheader("Rumus kunci")
    st.markdown(
        "- **Naive:** forecast t = aktual t-1\n"
        "- **SMA (n periode):** jumlah demand n periode terakhir / n\n"
        "- **WMA:** jumlah (bobot x demand); bobot terbesar pada data terbaru; "
        "jumlah bobot = 1\n"
        "- **Exponential smoothing:** F(t) = F(t-1) + alpha x (A(t-1) - F(t-1))\n"
        "- **Error** = Aktual - Forecast\n"
        "- **MAD** = jumlah |error| / n\n"
        "- **MSE** = jumlah error^2 / n\n"
        "- **MAPE** = (jumlah |error| / aktual) / n x 100%\n"
        "- **Bias** = jumlah error / n (positif = under-forecast)")

    st.subheader("Kaitan dengan ERP (modul manufacturing)")
    st.table(pd.DataFrame({
        "Kegiatan di kasus": [
            "Data demand historis", "SMA, WMA, ES",
            "MAD, MSE, MAPE, bias", "Memilih metode terbaik",
            "Forecast final minggu 7"],
        "Konsep di ERP": [
            "Riwayat penjualan/konsumsi (sales history)",
            "Model forecast statistik pada modul demand planning",
            "Pemantauan akurasi forecast",
            "Pemilihan model forecast (manual atau otomatis)",
            "Planned independent requirement yang masuk ke MPS dan MRP"],
    }))
    st.info("Alur: forecast -> aggregate planning/S&OP -> MPS -> MRP -> "
            "production order. Pada kasus Jason Enterprises, demand 500, 600, "
            "650, 800, 900, 800 adalah output tahap peramalan seperti ini.")
