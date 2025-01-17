# Add path to sys.path
import os, sys
app_root = os.getcwd()
sys.path.append(app_root)


# Imports
import streamlit as st
from components.home_text import HTEXT


# README
st.markdown(HTEXT)