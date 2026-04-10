from __future__ import annotations

import hmac

import streamlit as st

ADMIN_ROLE = "admin"
VIEWER_ROLE = "general_viewer"
SESSION_ROLE_KEY = "auth_role"
SESSION_AUTH_KEY = "auth_authenticated"
SESSION_NAME_KEY = "auth_display_name"


def auth_is_configured() -> bool:
    try:
        passwords = st.secrets.get("passwords", {})
    except Exception:
        passwords = {}
    return bool(passwords.get("admin")) and bool(passwords.get("viewer"))


def _get_passwords() -> tuple[str | None, str | None]:
    try:
        passwords = st.secrets.get("passwords", {})
    except Exception:
        passwords = {}
    admin_password = passwords.get("admin")
    viewer_password = passwords.get("viewer")
    return (str(admin_password) if admin_password else None, str(viewer_password) if viewer_password else None)


def is_logged_in() -> bool:
    return bool(st.session_state.get(SESSION_AUTH_KEY))


def get_role() -> str | None:
    if not is_logged_in():
        return None
    return st.session_state.get(SESSION_ROLE_KEY)


def get_user_email() -> str | None:
    return None


def get_display_name() -> str:
    role = get_role()
    if role == ADMIN_ROLE:
        return "Admin"
    if role == VIEWER_ROLE:
        return "General Viewer"
    return "Guest"


def logout() -> None:
    st.session_state.pop(SESSION_AUTH_KEY, None)
    st.session_state.pop(SESSION_ROLE_KEY, None)
    st.session_state.pop(SESSION_NAME_KEY, None)


def _check_password(candidate: str, expected: str | None) -> bool:
    if not expected:
        return False
    return hmac.compare_digest(candidate, expected)


def require_login() -> None:
    if not auth_is_configured():
        st.error(
            "Passwords are not configured. Add .streamlit/secrets.toml locally or set app secrets in Streamlit Cloud."
        )
        st.stop()

    if is_logged_in():
        return

    st.subheader("Sign in")
    st.write("Enter the general viewer password or admin password.")

    with st.form("password_login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            admin_password, viewer_password = _get_passwords()
            entered = password.strip()

            if _check_password(entered, admin_password):
                st.session_state[SESSION_AUTH_KEY] = True
                st.session_state[SESSION_ROLE_KEY] = ADMIN_ROLE
                st.session_state[SESSION_NAME_KEY] = "Admin"
                st.rerun()
            elif _check_password(entered, viewer_password):
                st.session_state[SESSION_AUTH_KEY] = True
                st.session_state[SESSION_ROLE_KEY] = VIEWER_ROLE
                st.session_state[SESSION_NAME_KEY] = "General Viewer"
                st.rerun()
            else:
                st.error("Incorrect password.")

    st.stop()


def require_role(allowed_roles: list[str] | tuple[str, ...] | set[str]) -> None:
    role = get_role()
    if role not in set(allowed_roles):
        st.error("You do not have permission to perform this action.")
        st.stop()


def require_editor() -> None:
    require_role([ADMIN_ROLE])
