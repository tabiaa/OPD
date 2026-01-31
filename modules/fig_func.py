import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.colors as mcolors


def calculate_bin_gap(s_no):

    if isinstance(s_no, int):
        length = s_no 
    else:
        length = len(s_no)
    if length <= 10:
        return 2 
    elif length <= 50:
        return 5 
    else:
        return 10 


# Function to plot Turn Around Time vs Customers
def entVsTA(s_no, TA):
    bin_gap = calculate_bin_gap(s_no)
    fig, ax = plt.subplots()
    ax.bar(s_no, TA, align='center', alpha=0.7)
    ax.set_xlabel('Customers')
    ax.set_ylabel('Turn Around Time')
    # ax.set_title('Turn Around Time in Queue for Each Customer')

    # Create custom x-tick positions with dynamic bin gap
    x_ticks = range(0, len(s_no) + 1, bin_gap)
    ax.set_xticks(x_ticks)

    st.pyplot(fig)


# Function to plot Arrival Time vs Customers
def entVsArrival(s_no, arrival):
    bin_gap = calculate_bin_gap(s_no)
    fig, ax = plt.subplots()
    ax.bar(s_no, arrival, align='center', alpha=0.7)
    ax.set_xlabel('Customers')
    ax.set_ylabel('Arrival Time')
    # ax.set_title('Arrival Time in Queue for Each Customer')

    # Create custom x-tick positions with dynamic bin gap
    x_ticks = range(0, len(s_no) + 1, bin_gap)
    ax.set_xticks(x_ticks)

    st.pyplot(fig)


# Function to plot Service Time vs Customers
def entVsService(s_no, service):
    bin_gap = calculate_bin_gap(s_no)
    fig, ax = plt.subplots()
    ax.bar(s_no, service, align='center', alpha=0.7)
    ax.set_xlabel('Customers')
    ax.set_ylabel('Service Time')
    # ax.set_title('Service Time in Queue for Each Customer')

    # Create custom x-tick positions with dynamic bin gap
    x_ticks = range(0, len(s_no) + 1, bin_gap)
    ax.set_xticks(x_ticks)

    st.pyplot(fig)


# Function to plot Wait Time vs Customers
def entVsWT(s_no, WT):
    bin_gap = calculate_bin_gap(s_no)
    fig, ax = plt.subplots()
    ax.bar(s_no, WT, align='center', alpha=0.7)
    ax.set_xlabel('Customers')
    ax.set_ylabel('Wait Time')
    # ax.set_title('Wait Time in Queue for Each Customer')

    # Create custom x-tick positions with dynamic bin gap
    x_ticks = range(0, len(s_no) + 1, bin_gap)
    ax.set_xticks(x_ticks)

    st.pyplot(fig)


# Function to visualize Server Utilization
def ServerUtilization(Server_util,server_no):
    idleTime = 1 - Server_util
    y = np.array([Server_util, idleTime])
    mylabels = ["Utilized Server", "Idle Time"]

    fig, ax = plt.subplots()
    ax.pie(y, labels=mylabels, autopct='%1.1f%%')
    ax.set_title(f"Server {server_no}")
    st.pyplot(fig)


# Function to visualize Server Utilization
def OverallUtilization(overall_util):
    idleTime = 1 - overall_util
    y = np.array([overall_util, idleTime])
    mylabels = ["Utilized Server", "Idle Time"]

    fig, ax = plt.subplots()
    ax.pie(y, labels=mylabels, autopct='%1.1f%%')
    ax.set_title("Overall Model Utilization")
    st.pyplot(fig)



# Function to calculate server utilization
def calculate_server_utilization(df):

    server_service_times = df.groupby('Server')['Service Time'].sum()

    # total_time = df['End Time'].max() - 0
    total_service_time = np.sum(df["Service Time"])

    # Calculate server utilization
    server_utilization = {server: service_time / total_service_time for server, service_time in server_service_times.items()}

    return server_utilization


def plot_gantt_chart(df, num_servers):
    # Map servers to specific colors
    server_color_map = {server: color for server, color in zip(df['Server'].unique(), mcolors.TABLEAU_COLORS.values())}

    fig, ax = plt.subplots(figsize=(13, 10))

    for i, row in df.iterrows():
        server = row["Server"]
        server_color = server_color_map[server]  # Use the predefined color mapping
        ax.barh(f"Server {server+1}", width=row["Service Time"], left=row["Start Time"],
                color=server_color, edgecolor="black")
        text = f'C:{row["Customer"]}\nST: {row["Service Time"]}'
        ax.text(row["Start Time"] + row["Service Time"] / 2, server, text, ha="center", va="center",
                color="black", fontsize=9)

    ax.set_xlabel("Time")
    ax.set_ylabel("Servers")
    # ax.set_title("Gantt Chart for Customers with Multiple Servers")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Dynamically calculate bin_gap for x-axis
    max_time = int(df["End Time"].max()) + 2
    bin_gap = calculate_bin_gap(max_time)
    x_ticks = list(range(0, max_time + 1, bin_gap))
    ax.set_xticks(x_ticks)

    plt.tight_layout()
    st.pyplot(fig)