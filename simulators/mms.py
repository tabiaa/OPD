import math
import pandas as pd
import numpy as np
import streamlit as st
from modules.fig_func import (
    plot_gantt_chart,
    entVsArrival,
    entVsService,
    entVsWT,
    entVsTA,
    calculate_server_utilization,
    ServerUtilization,
    OverallUtilization
)

def show():
    """M/M/S Queue Simulator page"""
    
    st.title("Simulation of M/M/S Queue System")

    # Function to calculate cumulative probability
    def calculate_CP(x, lambda_rate, prev_cp):
        if x == 0:
            return prev_cp + math.exp(-lambda_rate)  # Base case for x = 0
        else:
            # Use logarithms to prevent overflow
            log_term = x * math.log(lambda_rate) - math.log(math.factorial(x))
            return prev_cp + math.exp(-lambda_rate + log_term)

    # Function to calculate service time
    def calculate_service_time(mu_rate):
        return math.ceil(-mu_rate * np.log(np.random.rand()))

    # M/M/S Simulation Function
    def mmn(lambda_rate, mu_rate, num_servers):
        service_times = []
        cp_values = []
        inter_arrival_times = [0]
        arrival_times = [0]
        turn_around_times = []
        wait_times = []
        response_times = []

        prev_cp = 0
        customers = []
        num_entries = 0

        # Generate cumulative probabilities until CP reaches or exceeds 1
        max_entries = 500  # Set an upper limit to prevent overflow
        while prev_cp < 0.999999 and num_entries < max_entries:
            cp = calculate_CP(num_entries, lambda_rate, prev_cp)
            cp_values.append(cp)
            prev_cp = cp
            service_times.append(calculate_service_time(mu_rate))
            customers.append(num_entries)
            num_entries += 1

        cp_values.append(1)  # Ensure cumulative probability ends at 1

        df = pd.DataFrame({
            "Customer": customers,
            "Cumulative Probability": cp_values[:-1],
            "Service Time": service_times
        })

        # Format cumulative probability to 6 decimals
        df["Cumulative Probability"] = df["Cumulative Probability"].apply(lambda x: float(f"{x:.6f}"))

        # Create CP Lookup column (shifted cumulative probability, first row = 0)
        df["CP Lookup"] = [0] + df["Cumulative Probability"].tolist()[:-1]

        # Generate inter-arrival and arrival times
        arrival = 0
        for i in range(num_entries - 1):
            random = np.random.random()
            for j in range(len(cp_values)):
                if random < cp_values[j]:
                    inter_arrival_times.append(j)
                    arrival += j
                    arrival_times.append(arrival)
                    break

        df["Inter Arrival Time"] = inter_arrival_times
        df["Arrival Time"] = arrival_times

        # Assign servers and calculate times
        servers = [0] * num_servers
        start_times = []
        assigned_servers = []
        for i in range(len(df)):
            server = servers.index(min(servers))
            assigned_start_time = max(df.loc[i, "Arrival Time"], servers[server])
            servers[server] = assigned_start_time + df.loc[i, "Service Time"]

            start_times.append(assigned_start_time)
            assigned_servers.append(server)

        df["Start Time"] = start_times
        df["End Time"] = df["Start Time"] + df["Service Time"]
        df["Server"] = assigned_servers

        for i, row in df.iterrows():
            turn_around = row["End Time"] - row["Arrival Time"]
            wait = turn_around - row["Service Time"]
            response = row["Start Time"] - row["Arrival Time"]

            turn_around_times.append(turn_around)
            wait_times.append(wait)
            response_times.append(response)

        df["Turn Around Time"] = turn_around_times
        df["Wait Time"] = wait_times
        df["Response Time"] = response_times

        return df

    # Inputs for Mean Arrival and Service Rates
    lembda = st.number_input("Enter the arrival rate (λ)", min_value=0.1, value=2.0, step=0.1)
    meu = st.number_input("Enter the service rate (μ)", min_value=0.1, value=3.0, step=0.1)
    servers = st.number_input("Enter the number of servers (n)", min_value=1, value=2, step=1)

    if st.button("Generate Simulation"):
        df = mmn(lembda, meu, servers)
        if np.isclose(df['Cumulative Probability'], 1).any():
            first_index = df[np.isclose(df['Cumulative Probability'], 1)].index[0]
            df = df.iloc[:first_index]
        else:
            st.warning("No rows where 'Cumulative Probability' is approximately 1.")
        
        st.write("### Simulation Results")
        st.dataframe(df[[
            "Customer",
            "CP Lookup",
            "Cumulative Probability",
            "Inter Arrival Time",
            "Arrival Time",
            "Service Time",
            "Start Time",
            "End Time",
            "Server",
            "Turn Around Time",
            "Wait Time",
            "Response Time"
        ]], hide_index=True)

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
        plot_gantt_chart(df, servers)

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
