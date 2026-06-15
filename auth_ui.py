import streamlit as st
from auth import init_db, signup_user, login_user

# Initialize database on first run
init_db()

st.set_page_config(page_title = "Clarix Login", page_icon = "✦", layout = "centered")

# ── Session state for auth ───────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None

# ── If logged in, show welcome screen ─────────────────────────
if st.session_state.user:
    st.title(f"✦ Welcome, {st.session_state.user['name']}!")
    st.caption(f"Logged in as: {st.session_state.user['email']}")
    st.success("You are logged in!")

    st.markdown("---")
    st.markdown(f"**Your user ID:** {st.session_state.user['id']}")
    st.markdown(f"**Email:** {st.session_state.user['email']}")
    st.markdown(f"**Name:** {st.session_state.user['name']}")

    st.markdown("---")
    if st.button("🚪 Logout", type = "primary"):
        st.session_state.user = None
        st.rerun()

# ── Not logged in — show login/signup tabs ────────────────────
else:
    st.title("✦ Clarix")
    st.caption("AI-powered intelligence for your data and documents")

    tab1, tab2 = st.tabs(["🔐 Login", "✨ Sign Up"])

    # ── LOGIN TAB ─────────────────────────────────────────────
    with tab1:
        st.markdown("### Welcome back")
        login_email = st.text_input("Email", key = "login_email", placeholder = "you@example.com")
        login_password = st.text_input("Password", type = "password", key = "login_password")

        if st.button("Login", type = "primary", use_container_width = True):
            if not login_email or not login_password:
                st.error("Please fill in both fields")
            else:
                with st.spinner("Logging in..."):
                    success, result = login_user(login_email, login_password)

                if success:
                    st.session_state.user = result
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error(result)

    # ── SIGNUP TAB ────────────────────────────────────────────
    with tab2:
        st.markdown("### Create your account")
        signup_name = st.text_input("Full Name", key = "signup_name", placeholder = "Ayush Ladha")
        signup_email = st.text_input("Email", key = "signup_email", placeholder = "you@example.com")
        signup_password = st.text_input("Password", type = "password", key = "signup_password", help="At least 6 characters")
        signup_password_confirm = st.text_input("Confirm Password", type = "password", key = "signup_password_confirm")

        if st.button("Create Account", type = "primary", use_container_width = True):
            if not signup_name or not signup_email or not signup_password:
                st.error("Please fill in all fields")
            elif signup_password != signup_password_confirm:
                st.error("Passwords do not match")
            elif len(signup_password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                with st.spinner("Creating account..."):
                    success, msg = signup_user(signup_email, signup_name, signup_password)

                if success:
                    st.success(msg)
                    st.info("Now switch to the Login tab to sign in!")
                else:
                    st.error(msg)