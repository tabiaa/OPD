import streamlit as st
import sys
import os

# ---------- ADD CURRENT DIRECTORY TO PATH (CRITICAL FOR CLOUD DEPLOYMENT) ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add simulators directory to path
simulators_dir = os.path.join(current_dir, "simulators")
if simulators_dir not in sys.path:
    sys.path.insert(0, simulators_dir)

# ---------- PAGE CONFIG - ONLY HERE ----------
st.set_page_config(
    page_title="Queue Simulator Hub",
    layout="wide",
    initial_sidebar_state="auto"
)

# Import page modules SAFELY
try:
    from opd_sim import show as opd_show
except ImportError as e:
    st.error(f"Failed to import opd_sim: {e}")
    st.stop()

try:
    from simulators.mms import show as mms_show
except ImportError as e:
    st.error(f"Failed to import mms: {e}")
    st.stop()

try:
    from simulators.mgs import show as mgs_show
except ImportError as e:
    st.error(f"Failed to import mgs: {e}")
    st.stop()

try:
    from simulators.ggs import show as ggs_show
except ImportError as e:
    st.error(f"Failed to import ggs: {e}")
    st.stop()

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