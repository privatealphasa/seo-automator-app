import streamlit as st
import os

def require_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔒 SEO Automator — Login")

    password = st.text_input("Password", type="password")

    expected = os.getenv("APP_PASSWORD", "")

    if st.button("Login"):
        if password == expected and expected:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")

    st.stop()
