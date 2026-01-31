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
        # Initializing required lists
        s_no = []
        cp = []
        cpl = [0]
        int_arrival = [0]
        arrival = [0]
        service = []
        TA = []
        WT = []
        RT = []
        value = 0

        # Generating values for serial number, cumulative probability, and cumulative probability lookup
        x = 0
        while cpl[-1] < 1:
            s_no.append(x)
            value = norm.cdf(x, meu, sigma)
            cp.append(float("%.4f" % value))
            cpl.append(cp[-1])
            x += 1
        cpl.pop(-1)

        # Generating customers dynamically
        ran_var = 0
        arrival_time = 0
        service.append(math.ceil(-meu * math.log(random.uniform(0, 1))))
        for i in range(len(cp)-1):
            ran_var = float("%.4f" % random.uniform(0, 1))
            for j in range(len(cp)):
                if cpl[j] < ran_var <= cp[j]:
                    inter_arrival = j
                    int_arrival.append(inter_arrival)
                    break
            
            arrival_time += inter_arrival
            arrival.append(arrival_time)
            service.append(math.ceil(-meu * math.log(random.uniform(0, 1))))

        # Total number of entries generated
        num_entries = len(arrival)

        # Initializing servers
        server_start = [[] for _ in range(n_servers)]
        server_end = [[] for _ in range(n_servers)]
        S = []
        E = []
        server_id = []

        # Assigning customers to servers
        for i in range(num_entries):
            # Find the next available server or the server that becomes free the earliest
            earliest_server = min(range(n_servers), key=lambda s: server_end[s][-1] if server_end[s] else 0)
            if server_end[earliest_server] and arrival[i] < server_end[earliest_server][-1]:
                start_time = server_end[earliest_server][-1]
            else:
                start_time = arrival[i]

            end_time = start_time + service[i]

            # Update server records
            server_start[earliest_server].append(start_time)
            server_end[earliest_server].append(end_time)

            # Append to overall records
            S.append(start_time)
            E.append(end_time)
            server_id.append(earliest_server)

        # Calculating performance metrics
        for i in range(num_entries):
            TA.append(E[i] - arrival[i])
            WT.append(TA[i] - service[i])
            RT.append(S[i] - arrival[i])

        # Creating the "Inter Arrival Range" column
        ia_range = [f"{cp[i]:.4f}-{cp[i+1]:.4f}" for i in range(len(cp)-1)]
        ia_range.append(f"{cp[-1]:.4f}-1.0000")

        # Generating the DataFrame with the required columns
        result = [
            s_no,
            cp,
            ia_range,
            int_arrival,
            arrival,
            service,
            server_id,
            S,
            E,
            TA,
            RT,
            WT
        ]

        df = pd.DataFrame(result, index=["Customer", "Cumulative Probability", "I.A Range", "Inter Arrival Time", "Arrival Time", 
                                         "Service Time", "Server", "Start Time", "End Time", "Turn Around Time", 
                                         "Response Time", "Wait Time"])
        df = df.dropna(axis=1)
        pd.set_option('display.max_columns', None)
        df = df.transpose()
        return df

    lembda = st.number_input("Mean arrival rate (λ)", step=0.1, format="%.2f", value=1.5)
    meu = st.number_input("Mean service rate (μ) - Minimum", step=0.1, format="%.2f", value=5.0)
    sigma = st.number_input("Standard deviation (σ) - Maximum", min_value=1.0, max_value=50.0, step=0.1, value=7.0)
    num_servers = st.number_input("Number of servers", min_value=1, max_value=10, step=1, value=3)

    if st.button("Generate Simulation"):
        df = ggn(lembda, meu, sigma, num_servers)
        st.write("### Simulation Results")
        st.dataframe(df.drop(["Cumulative Probability", "I.A Range"],axis=1), hide_index=True)

        avg_interarrival = df["Inter Arrival Time"].mean()
        avg_service = df["Service Time"].mean()
        avg_TA = df["Turn Around Time"].mean()
        avg_WT = df["Wait Time"].mean()
        avg_RT = df["Response Time"].mean()

        st.write(f"**Average Inter-Arrival Time**: {avg_interarrival:.2f}")
        st.write(f"**Average Service Time**: {avg_service:.2f}")
        st.write(f"**Average Turn-Around Time**: {avg_TA:.2f}")
        st.write(f"**Average Wait Time**: {avg_WT:.2f}")
        st.write(f"**Average Response Time**: {avg_RT:.2f}")

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