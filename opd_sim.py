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
        original_len = len(df)
        df = df.dropna(subset=["InterArrival (Min)", "Duration (Min)"])
        cleaned_len = len(df)

        # ---------- DATA PREVIEW ----------
        st.subheader("Uploaded Data Preview")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            max_rows = min(100, len(df))
            rows_to_show = st.slider("Number of rows to display", min_value=10, max_value=max_rows, value=min(10, max_rows), step=10)
        
        with col2:
            st.info(f"Total records: {original_len} → {cleaned_len} (after cleanup)")
        
        st.dataframe(df.head(rows_to_show), use_container_width=True)
        
        # Optional: Show column info
        with st.expander("📋 Column Information"):
            st.write(df.dtypes.to_frame("Data Type"))

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
Result: {'Accepted' if acc_a else '❌ Rejected'}
"""
        )

        col2.markdown(
            f"""
**Service ~ Exponential**  
χ² = {chi_s:.2f}  
df = {df_s}  
critical = {crit_s:.2f}  
Result: {'Accepted' if acc_s else '❌ Rejected'}
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

        # ---------- MONTE CARLO SIMULATION (ONLY IF BOTH TESTS ACCEPTED) ----------
        if acc_a and acc_s:
            st.success("Both distributions validated! Monte Carlo simulation available.")
            
            st.subheader("Monte Carlo Simulation with CP Lookup")
            st.markdown("""
            Simulate future OPD customers using Monte Carlo method.  
            **Note:** Row 1 = System start (I.A = 0, Arrival = 0). Actual patients start from Row 2.
            """)
            
            n_sim = st.number_input("Number of future patients to simulate (excluding system start row)", 1, 1000, 10)

            if st.button("Run Monte Carlo Simulation"):
                # Initialize lists for all columns
                ids = []
                cp_values = []
                cp_lookup_values = []
                ia_values = []
                poisson_arrivals = []
                service_times_list = []
                start_times = []
                end_times = []
                tat_values = []
                wt_values = []
                rt_values = []

                n_total = n_sim  # number of actual patients

                current_time = 0.0
                server_free = 0.0

                for i in range(n_total):
                    ids.append(i + 1)
                    
                    # CP value (just random for simulation)
                    cp = np.random.rand()
                    cp_values.append(cp)
                    
                    # CP Lookup = previous CP, for first row = 0
                    cp_lookup_values.append(cp_values[i - 1] if i > 0 else 0.0)
                    
                    # Inter-arrival time (exponential)
                    ia = -mu_hat * np.log(np.random.rand())
                    ia_values.append(ia if i > 0 else 0.0)  # first I.A = 0
                    
                    # Poisson arrival = cumulative sum of inter-arrivals
                    current_time += ia if i > 0 else 0.0
                    poisson_arrivals.append(current_time)  # first arrival = 0
                    
                    # Service time
                    service_time = -mu_hat * np.log(np.random.rand())
                    service_times_list.append(service_time)
                    
                    # Start and end times
                    start = max(current_time, server_free)
                    end = start + service_time
                    start_times.append(start)
                    end_times.append(end)
                    
                    # Turnaround, waiting, response
                    tat = end - current_time
                    wt = start - current_time
                    rt = start - current_time
                    tat_values.append(tat)
                    wt_values.append(wt)
                    rt_values.append(rt)
                    
                    # Update server free
                    server_free = end

                # Build DataFrame
                sim_df = pd.DataFrame({
                    "ID": ids,
                    "CP": np.round(cp_values, 4),
                    "CP Lookup": np.round(cp_lookup_values, 4),
                    "I.A": np.round(ia_values, 2),
                    "Poisson Arrival": np.round(poisson_arrivals, 2),
                    "Service Time": np.round(service_times_list, 2),
                    "Start": np.round(start_times, 2),
                    "End": np.round(end_times, 2),
                    "T.A.T": np.round(tat_values, 2),
                    "W.T": np.round(wt_values, 2),
                    "R.T": np.round(rt_values, 2)
                })



                st.dataframe(sim_df, use_container_width=True)

                # Download button
                output = BytesIO()
                sim_df.to_excel(output, index=False)
                st.download_button(
                    "📥 Download Simulation Results (Excel)",
                    data=output.getvalue(),
                    file_name="opd_monte_carlo_simulation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # Summary statistics (exclude row 1 which is system start)
                st.subheader("Simulation Summary (Actual Patients Only)")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Patients", n_sim)
                col2.metric("Avg. Waiting Time", f"{np.mean(wt_values[1:]):.2f} min")
                col3.metric("Avg. Turnaround Time", f"{np.mean(tat_values[1:]):.2f} min")
                col4.metric("Server Utilization", f"{rho:.1%}")
        else:
            st.warning("Simulation requires both distributions to pass goodness-of-fit tests. Please review your data or adjust significance level.")
    else:
        st.info("Please upload an Excel file containing OPD data with columns 'InterArrival (Min)' and 'Duration (Min)'")