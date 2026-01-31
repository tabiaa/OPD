import streamlit as st
import sys
import os

# ---------- PAGE CONFIG - ONLY HERE ----------
st.set_page_config(
    page_title="Queue Simulator Hub",
    layout="wide",
    initial_sidebar_state="auto"
)

# Add pages folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "simulators"))

# Import page modules
from opd_sim import show as opd_show
from simulators.mms import show as mms_show
from simulators.mgs import show as mgs_show
from simulators.ggs import show as ggs_show

# Initialize session state for page
if 'page' not in st.session_state:
    st.session_state['page'] = 'opd'

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.title("Navigation")
selection = st.sidebar.radio(
    "Select Page",
    ["OPD Simulator", "M/M/S Simulator", "M/G/S Simulator", "G/G/S Simulator"]
)

# Map sidebar selection to page
mapping = {
    "OPD Simulator": "opd",
    "M/M/S Simulator": "mms",
    "M/G/S Simulator": "mgs",
    "G/G/S Simulator": "ggs"
}
st.session_state['page'] = mapping[selection]

# ---------- PAGE ROUTING ----------
if st.session_state['page'] == 'opd':
    opd_show()
elif st.session_state['page'] == 'mms':
    mms_show()
elif st.session_state['page'] == 'mgs':
    mgs_show()
elif st.session_state['page'] == 'ggs':
    ggs_show()