"""
live_app.py

Entry point of the Streamlit app:

    streamlit run live_app.py

It only names the two pages and puts them in the sidebar. The pages
themselves are in app_pages/ (CLAUDE.md 5 and 12.7):

    app_pages/single_delivery.py   one delivery from a source to a destination:
                                   live GNN, the six agents, "On the road"
                                   (disruptions and re-planning), the
                                   Dijkstra vs Tsinghua comparison
    app_pages/fleet_rounds.py      several deliveries per vehicle, planned
                                   with OR-Tools

(Until 2026-10-14 the single-delivery page was this file, and Streamlit's
automatic page list showed it only as "live app", which was easy to miss.)
"""

import streamlit as st

st.set_page_config(page_title="Bengaluru Logistics", page_icon=":material/alt_route:", layout="wide")

SINGLE_DELIVERY = st.Page("app_pages/single_delivery.py", title="Single delivery",
                          icon=":material/alt_route:", default=True)
FLEET_ROUNDS = st.Page("app_pages/fleet_rounds.py", title="Multi-stop fleet", icon=":material/local_shipping:")

st.navigation([SINGLE_DELIVERY, FLEET_ROUNDS]).run()
