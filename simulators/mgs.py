import math
import random
import numpy as np
import pandas as pd
import streamlit as st
from modules.fig_func import *

def show():
    """M/G/S Queue Simulator page"""
    
    st.title("Simulation of M/G/S")

    def mgn(lembda, meuMin, meuMax, n):
        meu = (meuMin + meuMax) / 2
        s_no = []
        cp = []
        cpl = [0]
        int_arrival = [0]
        arrival = [0]
        service = []
        TA = []
        WT = []
        RT = []
        server_assigned = []
        start_times = []
        end_times = []
        value = 0

        # Generating serial numbers and probabilities
        x = 0
        while cpl[-1] < 1:
            s_no.append(x)
            value = value + (((math.exp(-lembda)) * (lembda ** x)) / math.factorial(x))
            cp.append(float("%.4f" % value))
            cpl.append(cp[-1])
            x += 1
        cpl.pop(-1)

        # Inter-arrival time generation
        for i in range(len(cp)-1):
            ran_var = float("%.4f" % random.uniform(0, 1))
            for j in range(len(cp)):
                if cpl[j] < ran_var and ran_var < cp[j]:
                    int_arrival.append(j)

        # Arrival time generation
        for i in range(1, len(cp)):
            arrival.append(int_arrival[i] + arrival[i - 1])

        # Service time generation
        for i in range(len(cp)):
            service.append(math.ceil(-meu * math.log(random.uniform(0, 1))))

        # Server assignments
        ends = [0] * n
        for i in range(len(cp)):
            min_end_time_server = min(range(n), key=lambda x: ends[x])
            start_time = max(arrival[i], ends[min_end_time_server])
            end_time = start_time + service[i]
            ends[min_end_time_server] = end_time
            server_assigned.append(min_end_time_server)
            start_times.append(start_time)
            end_times.append(end_time)

        # Calculate turnaround time, wait time, and response time
        for i in range(len(cp)):
            TA.append(end_times[i] - arrival[i])
            WT.append(TA[-1] - service[i])
            RT.append(start_times[i] - arrival[i])

        # Create result DataFrame
        result = pd.DataFrame({
            "Customer": s_no,
            "Cumulative Probability": cp,
            "Inter Arrival Time": int_arrival,
            "Arrival Time": arrival,
            "Service Time": service,
            "Server": server_assigned,
            "Start Time": start_times,
            "End Time": end_times,
            "Turn Around Time": TA,
            "Response Time": RT,
            "Wait Time": WT
        })
        return result

    # Input fields
    lambda_value = st.number_input("Enter the value of lambda", min_value=0.1, value=1.5, step=0.1)
    meu_min = st.number_input("Enter the minimum value for meu", min_value=1.0, value=3.0, step=0.1)
    meu_max = st.number_input("Enter the maximum value for meu", min_value=1.0, value=5.0, step=0.1)
    num_servers = st.number_input("Enter the number of servers", min_value=1, value=1, step=1)

    # Run simulation
    if st.button("Generate Simulation"):
        df = mgn(lambda_value, meu_min, meu_max, num_servers)

        # Truncate DataFrame based on Cumulative Probability
        if (np.isclose(df['Cumulative Probability'], 1, atol=1e-4)).any():
            first_index = df[np.isclose(df['Cumulative Probability'], 1, atol=1e-4)].index[0]
            df = df.iloc[:first_index + 1]
        else:
            st.warning("No rows where 'Cumulative Probability' equals 1. Using the entire DataFrame.")

        st.write("### Simulation Results")
        st.dataframe(df.drop(["Cumulative Probability"],axis=1), hide_index=True)

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