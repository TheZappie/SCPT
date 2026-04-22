import streamlit as st

# logging.basicConfig(format="Conversion %(levelname)s: %(message)s", level=logging.WARNING)
page1 = st.Page("pages/home.py", title="Home", icon=":material/home:")
page2 = st.Page("pages/ray_paths.py", title="Ray tracing", icon=":material/sensors:")
st.logo("assets/fugro.png", link="https://www.fugro.com/")
pg = st.navigation([page1, page2])
pg.run()
