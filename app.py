import streamlit as st
import sys
import os

# ---------- CRITICAL: FIX PATH RESOLUTION ----------
# Get the absolute path to the root directory
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Add modules directory
modules_dir = os.path.join(root_dir, "modules")
if modules_dir not in sys.path:
    sys.path.insert(0, modules_dir)

# Add simulators directory
simulators_dir = os.path.join(root_dir, "simulators")
if simulators_dir not in sys.path:
    sys.path.insert(0, simulators_dir)

# ---------- PAGE CONFIG (MUST BE FIRST) ----------
st.set_page_config(
    page_title="Queue Simulator Hub",
    layout="wide",
    initial_sidebar_state="auto"
)

# ---------- IMPORT PAGES ----------
try:
    from opd_sim import show as opd_show
except Exception as e:
    st.error(f"❌ Failed to import opd_sim: {str(e)}")
    st.stop()

try:
    from simulators.mms import show as mms_show
except Exception as e:
    st.error(f"❌ Failed to import mms: {str(e)}")
    st.stop()

try:
    from simulators.mgs import show as mgs_show
except Exception as e:
    st.error(f"❌ Failed to import mgs: {str(e)}")
    st.stop()

try:
    from simulators.ggs import show as ggs_show
except Exception as e:
    st.error(f"❌ Failed to import ggs: {str(e)}")
    st.stop()

# ---------- SESSION STATE ----------
if 'page' not in st.session_state:
    st.session_state['page'] = 'opd'

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.title("Navigation")
selection = st.sidebar.radio(
    "Select Page",
 ["OPD Simulator", "M/M/S Simulator", "M/G/S Simulator", "G/G/S Simulator"],
    index=["opd", "mms", "mgs", "ggs"].index(st.session_state['page'])
)

# Map selection to page key
page_map = {
    "OPD Simulator": "opd",
    "M/M/S Simulator": "mms",
    "M/G/S Simulator": "mgs",
    "G/G/S Simulator": "ggs"
}
st.session_state['page'] = page_map[selection]

# ---------- RENDER SELECTED PAGE ----------
if st.session_state['page'] == 'opd':
    opd_show()
elif st.session_state['page'] == 'mms':
    mms_show()
elif st.session_state['page'] == 'mgs':
    mgs_show()
elif st.session_state['page'] == 'ggs':
    ggs_show()