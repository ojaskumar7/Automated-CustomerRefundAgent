"""Streamlit frontend: customer submission + status; admin queue."""
from __future__ import annotations

import os

import requests
import streamlit as st

# API base URL
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")


def api_get(path: str):
    resp = requests.get(f"{API_BASE}{path}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, json_data: dict):
    resp = requests.post(f"{API_BASE}{path}", json=json_data, timeout=10)
    resp.raise_for_status()
    return resp.json()


st.set_page_config(page_title="Refund Resolution Agent", page_icon="💰", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Mode", ["Customer", "Admin"])

if mode == "Customer":
    st.title("💰 Submit a Refund Request")

    with st.form("grievance_form"):
        order_id = st.text_input("Order ID", placeholder="e.g., ORD-12345")
        grievance_text = st.text_area(
            "Describe your issue",
            placeholder=(
                "Explain what happened (e.g., 'screen cracked on arrival', "
                "'item delivered 2 weeks late', 'charged twice')"
            ),
            height=150,
        )
        submitted = st.form_submit_button("Submit Request")

        if submitted:
            if not order_id or not grievance_text:
                st.error("Please fill in both Order ID and description.")
            else:
                try:
                    result = api_post("/grievances", {
                        "order_id": order_id,
                        "grievance_text": grievance_text,
                    })
                    st.success(f"Submitted! Grievance ID: `{result['grievance_id']}`")
                    st.info(f"Status: **{result['status']}** — {result['message']}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to submit: {e}")

    st.divider()
    st.subheader("Check Status")
    check_id = st.text_input("Grievance ID", placeholder="Paste your grievance ID here")
    if st.button("Check Status") and check_id:
        try:
            status = api_get(f"/grievances/{check_id}")
            st.json(status)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                st.error("Grievance not found")
            else:
                st.error(f"Error: {e}")

elif mode == "Admin":
    st.title("🛠️ Admin: Escalation Queue")

    try:
        queue = api_get("/escalations")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load queue: {e}")
        queue = []

    if not queue:
        st.info("No escalated grievances pending review.")
    else:
        for item in queue:
            title = f"📋 {item['id']} — {item['category']} — {item['grievance_text'][:60]}..."
            with st.expander(title):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Order ID:** {item['order_id']}")
                    st.write(f"**Customer:** {item['customer_id']}")
                    st.write(f"**Category:** {item['category']}")
                    st.write(f"**Extracted Reason:** {item['extracted_reason']}")
                    req_amt = item['requested_amount']
                    if req_amt:
                        req_str = f"**Requested Amount:** ${req_amt:.2f}"
                    else:
                        req_str = "**Requested Amount:** Not specified"
                    st.write(req_str)
                with col2:
                    st.write(f"**Order Amount:** ${item['order_amount']:.2f}")
                    st.write(f"**Already Refunded:** ${item['refunded_amount']:.2f}")
                    max_refund = item['order_amount'] - item['refunded_amount']
                    st.write(f"**Max Refundable:** ${max_refund:.2f}")
                    st.write(f"**Escalation Reason:** {item['decision_reason']}")
                    st.write(f"**Submitted:** {item['created_at']}")

                st.write("**Grievance Text:**")
                st.write(item['grievance_text'])

                max_refundable = item['order_amount'] - item['refunded_amount']

                with st.form(f"decide_{item['id']}"):
                    action = st.radio(
                        "Decision",
                        ["approve", "reject", "adjust"],
                        horizontal=True,
                        key=f"action_{item['id']}",
                    )
                    amount = None
                    if action == "adjust":
                        amount = st.number_input(
                            "Adjusted Amount ($)",
                            min_value=0.0,
                            max_value=float(max_refundable * 2) if max_refundable > 0 else 1000.0,
                            value=float(max_refundable),
                            step=0.01,
                            key=f"amount_{item['id']}",
                        )
                    override = False
                    if action == "adjust" and amount and amount > max_refundable:
                        override = st.checkbox(
                            f"Override max refundable (${max_refundable:.2f})",
                            key=f"override_{item['id']}",
                        )
                    reason = st.text_area(
                        "Reason (required)",
                        placeholder="Explain your decision...",
                        key=f"reason_{item['id']}",
                    )
                    if st.form_submit_button("Submit Decision"):
                        if not reason:
                            st.error("Reason is required")
                        else:
                            try:
                                payload = {"action": action, "reason": reason}
                                if amount is not None:
                                    payload["amount"] = amount
                                if override:
                                    payload["override_max"] = True
                                result = api_post(f"/escalations/{item['id']}/decide", payload)
                                st.success(f"Decision recorded: {result['message']}")
                                st.rerun()
                            except requests.exceptions.HTTPError as e:
                                st.error(f"Failed: {e.response.json().get('detail', e)}")
                            except requests.exceptions.RequestException as e:
                                st.error(f"Failed: {e}")

