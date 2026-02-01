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
        cpl = [0]  # CP Lookup starts with 0
        int_arrival = []
        arrival = []
        service = []
        TA = []
        WT = []
        RT = []
        server_assigned = []
        start_times = []
        end_times = []

        # Generate cumulative probability
        x = 0
        cum_prob = 0
        while cum_prob < 0.999999:
            s_no.append(x)
            cum_prob += (math.exp(-lembda) * (lembda ** x)) / math.factorial(x)
            cp.append(round(cum_prob, 6))
            cpl.append(round(cp[-1], 6))
            x += 1
        cpl.pop(-1)  # remove last extra element

        num_customers = len(s_no)

        # Inter-arrival times using CP Lookup
        for i in range(num_customers):
            ran_var = random.uniform(0, 1)
            for j in range(num_customers):
                if cpl[j] <= ran_var < cp[j]:
                    int_arrival.append(j)
                    break
        if len(int_arrival) < num_customers:
            int_arrival.append(0)  # padding if needed

        # Arrival times
        arrival.append(0)
        for i in range(1, num_customers):
            arrival.append(arrival[i - 1] + int_arrival[i])

        # Service times
        for i in range(num_customers):
            service.append(math.ceil(-meu * math.log(random.uniform(0, 1))))

        # Server assignment
        ends = [0] * n
        for i in range(num_customers):
            min_server = min(range(n), key=lambda x: ends[x])
            start_time = max(arrival[i], ends[min_server])
            end_time = start_time + service[i]
            ends[min_server] = end_time
            server_assigned.append(min_server)
            start_times.append(start_time)
            end_times.append(end_time)

        # Calculate TA, WT, RT
        for i in range(num_customers):
            TA.append(end_times[i] - arrival[i])
            WT.append(TA[i] - service[i])
            RT.append(start_times[i] - arrival[i])

        # Create DataFrame
        result = pd.DataFrame({
            "Customer": s_no,
            "CP Lookup": cpl,
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

        st.write("### Simulation Results")
        st.dataframe(df, hide_index=True)

        # Averages
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
