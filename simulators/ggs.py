import random
import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from modules.fig_func import *

def show():
    """G/G/S Queue Simulator page"""
    
    st.title("Simulation of G/G/S")

    def ggn(lembda, meu, sigma, n_servers):
        # Lists
        s_no = []
        cp = []      # actual cumulative probability
        cpl = [0]    # CP lookup for mapping random numbers
        int_arrival = []
        arrival = []
        service = []
        TA = []
        WT = []
        RT = []

        # Generate CP and CP Lookup
        x = 0
        while True:
            value = norm.cdf(x, meu, sigma)  # cumulative probability for customer x
            if value >= 0.999999:
                break
            s_no.append(x)
            cp.append(round(value, 6))      # actual CP
            cpl.append(round(value, 6))     # CP Lookup
            x += 1
        num_customers = len(s_no)

        # Inter-arrival times using CP Lookup
        for i in range(num_customers):
            ran_var = random.uniform(0, 1)
            for j in range(num_customers):
                if cpl[j] <= ran_var <= cp[j]:
                    int_arrival.append(j)
                    break
        if len(int_arrival) < num_customers:
            int_arrival.append(0)

        # Arrival times
        arrival.append(0)
        for i in range(1, num_customers):
            arrival.append(arrival[i - 1] + int_arrival[i])

        # Service times
        for i in range(num_customers):
            service.append(math.ceil(-meu * math.log(random.uniform(0, 1))))

        # Server assignment
        server_start = [[] for _ in range(n_servers)]
        server_end = [[] for _ in range(n_servers)]
        S = []
        E = []
        server_id = []

        for i in range(num_customers):
            earliest_server = min(range(n_servers), key=lambda s: server_end[s][-1] if server_end[s] else 0)
            start_time = max(arrival[i], server_end[earliest_server][-1] if server_end[earliest_server] else 0)
            end_time = start_time + service[i]

            server_start[earliest_server].append(start_time)
            server_end[earliest_server].append(end_time)

            S.append(start_time)
            E.append(end_time)
            server_id.append(earliest_server)

        # Performance metrics
        for i in range(num_customers):
            TA.append(E[i] - arrival[i])
            WT.append(TA[i] - service[i])
            RT.append(S[i] - arrival[i])

        # Inter Arrival Range
        ia_range = [f"{cpl[i]:.6f}-{cp[i]:.6f}" for i in range(num_customers)]

        # Create DataFrame
        df = pd.DataFrame({
            "Customer": s_no,
            "CP Lookup": cpl[:-1],          # lookup values
            "Cumulative Probability": cp,   # actual CP
            "I.A Range": ia_range,
            "Inter Arrival Time": int_arrival,
            "Arrival Time": arrival,
            "Service Time": service,
            "Server": server_id,
            "Start Time": S,
            "End Time": E,
            "Turn Around Time": TA,
            "Response Time": RT,
            "Wait Time": WT
        })

        return df

    # Inputs
    lembda = st.number_input("Mean arrival rate (λ)", step=0.1, format="%.2f", value=1.5)
    meu = st.number_input("Mean service rate (μ)", step=0.1, format="%.2f", value=5.0)
    sigma = st.number_input("Standard deviation (σ)", min_value=1.0, max_value=50.0, step=0.1, value=7.0)
    num_servers = st.number_input("Number of servers", min_value=1, max_value=10, step=1, value=3)

    if st.button("Generate Simulation"):
        df = ggn(lembda, meu, sigma, num_servers)
        st.write("### Simulation Results")
        st.dataframe(df.drop(["I.A Range"], axis=1), hide_index=True)

        # Averages
        st.write(f"**Average Inter-Arrival Time**: {df['Inter Arrival Time'].mean():.2f}")
        st.write(f"**Average Service Time**: {df['Service Time'].mean():.2f}")
        st.write(f"**Average Turn-Around Time**: {df['Turn Around Time'].mean():.2f}")
        st.write(f"**Average Wait Time**: {df['Wait Time'].mean():.2f}")
        st.write(f"**Average Response Time**: {df['Response Time'].mean():.2f}")

        st.write("### Gantt Chart for Servers")
        plot_gantt_chart(df, num_servers)

        st.write("### Wait Time vs Customers")
        entVsWT(df["Customer"], df["Wait Time"])

        st.write("### Turnaround Time vs Customers")
        entVsTA(df["Customer"], df["Turn Around Time"])

        st.write("### Arrival Time vs Customers")
        entVsArrival(df["Customer"], df["Arrival Time"])

        st.write("### Service Time vs Customers")
        entVsService(df["Customer"], df["Service Time"])

        st.write("### Model Utilization")
        server_util = calculate_server_utilization(df)
        OverallUtilization(np.sum(list(server_util.values())))

        st.write("### Server Utilization")
        i = 1
        for server, utilization in server_util.items():
            ServerUtilization(utilization, server_no=i)
            i += 1
