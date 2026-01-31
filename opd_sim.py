import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

def show():
    """Main function to display the OPD Simulator page"""
    
    st.title("OPD Queue Simulator (Poisson–Exponential Model)")
    
    uploaded_file = st.file_uploader("Upload OPD Excel File", type=["xlsx"])

    # ---------- Chi-Square: Poisson (Arrival) ----------
    def chi_square_poisson(arrival_counts, lam):
        observed = np.bincount(arrival_counts)
        expected = np.array([
            len(arrival_counts) * stats.poisson.pmf(i, lam)
            for i in range(len(observed))
        ])

        mask = expected >= 5
        if not np.any(mask):
            return np.nan, 0, 0, False

        chi2 = np.sum((observed[mask] - expected[mask])**2 / expected[mask])
        df = max(1, np.sum(mask) - 1)
        critical = stats.chi2.ppf(0.95, df)

        return chi2, df, critical, chi2 < critical

    # ---------- Chi-Square: Exponential (Service) ----------
    def chi_square_exponential(data, rate):
        data = data[data > 0]
        n = len(data)
        if n < 5:
            return np.nan, 0, 0, False

        k = int(np.sqrt(n))
        bins = np.linspace(0, data.max(), k + 1)
        observed, _ = np.histogram(data, bins=bins)

        expected = []
        for i in range(k):
            a, b = bins[i], bins[i + 1]
            prob = np.exp(-rate * a) - np.exp(-rate * b)
            expected.append(n * prob)
        expected = np.array(expected)

        mask = expected >= 5
        if not np.any(mask):
            return np.nan, 0, 0, False

        chi2 = np.sum((observed[mask] - expected[mask])**2 / expected[mask])
        df = max(1, np.sum(mask) - 2)
        critical = stats.chi2.ppf(0.95, df)

        return chi2, df, critical, chi2 < critical

    # ---------- MAIN ----------
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df = df.dropna(subset=["InterArrival (Min)", "Duration (Min)"])

        inter_arrival = df["InterArrival (Min)"].values
        service = df["Duration (Min)"].values
        service = service[service > 0]

        # ---------- ARRIVAL COUNTS (Poisson) ----------
        interval = 10  # minutes
        arrival_times = np.cumsum(inter_arrival)
        bins = int(arrival_times.max() // interval) + 1
        arrival_counts = np.zeros(bins, dtype=int)

        for t in arrival_times:
            idx = int(t // interval)
            if idx < bins:
                arrival_counts[idx] += 1

        lam_interval = np.mean(arrival_counts)      # arrivals per interval
        lam_per_min = lam_interval / interval       # arrivals per minute
        mu_hat = 1 / np.mean(service)              # services per minute
        rho = lam_per_min / mu_hat                  # utilization

        # ---------- DISPLAY QUEUE MODEL ----------
        st.subheader("Identified Queueing Model")
        st.markdown(
            f"""
**Arrival Process:** Poisson distribution  
**Service Process:** Exponential distribution  

**Estimated Parameters:**  
- Arrival rate (λ) = `{lam_per_min:.3f}` per minute  
- Service rate (μ) = `{mu_hat:.3f}` per minute  
- Utilization (ρ) = `{rho:.3f}`  

Queue Model: **M/M/1**
"""
        )

        # ---------- CHI-SQUARE TESTS ----------
        st.subheader("Goodness-of-Fit Tests (Chi-Square, α = 0.05)")
        chi_a, df_a, crit_a, acc_a = chi_square_poisson(arrival_counts, lam_interval)
        chi_s, df_s, crit_s, acc_s = chi_square_exponential(service, mu_hat)

        col1, col2 = st.columns(2)

        col1.markdown(
            f"""
**Arrival ~ Poisson**  
χ² = {chi_a:.2f}  
df = {df_a}  
critical = {crit_a:.2f}  
Result: {'Accepted' if acc_a else 'Rejected'}
"""
        )

        col2.markdown(
            f"""
**Service ~ Exponential**  
χ² = {chi_s:.2f}  
df = {df_s}  
critical = {crit_s:.2f}  
Result: {'Accepted' if acc_s else 'Rejected'}
"""
        )

        # ---------- VISUALIZATION ----------
        st.subheader("Distribution Visualization")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(arrival_counts, discrete=True, ax=ax[0], color="#1f77b4")
        ax[0].set_title("Arrivals per Interval (Poisson)")

        sns.histplot(service, stat="density", ax=ax[1], color="#ff7f0e")
        x = np.linspace(0, service.max(), 100)
        ax[1].plot(x, stats.expon.pdf(x, scale=1/mu_hat), "r", lw=2)
        ax[1].set_title("Service Time (Exponential)")

        st.pyplot(fig)

        # ---------- SIMULATION ----------
        st.subheader("Future Customer Simulation")
        st.markdown(
            """
Simulate future OPD customers using the M/M/1 queue model.
"""
        )

        n_sim = st.number_input("Number of future patients to simulate", 1, 1000, 10)

        if st.button("Run Simulation"):
            # arrivals in minutes using per-minute λ
            arrivals = np.cumsum(np.random.exponential(1 / lam_per_min, n_sim))
            services = np.random.exponential(1 / mu_hat, n_sim)

            start, end, wait = [], [], []
            server_free = 0

            for i in range(n_sim):
                s = max(arrivals[i], server_free)
                e = s + services[i]
                start.append(s)
                end.append(e)
                wait.append(s - arrivals[i])
                server_free = e

            sim_df = pd.DataFrame({
                "Patient": range(1, n_sim + 1),
                "Arrival Time (min)": arrivals,
                "Service Start (min)": start,
                "Service End (min)": end,
                "Waiting Time (min)": wait
            }).round(2)

            st.dataframe(sim_df, use_container_width=True)

            # Download
            output = BytesIO()
            sim_df.to_excel(output, index=False)
            st.download_button(
                "Download Simulation Results",
                data=output.getvalue(),
                file_name="opd_simulation.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )