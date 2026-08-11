"""Phase 1 service skeleton for chatbot-ui (architecture.md §17 Phase 1
deliverable: "5 service skeletons + health checks"). No chat UI
functionality yet -- that is Phase 6. This exists only to prove the
container builds, binds, and is health-checkable inside Docker Compose.
"""

import streamlit as st

st.set_page_config(page_title="VoyagerAI Istanbul — Phase 1 skeleton")
st.title("VoyagerAI Istanbul")
st.write("Phase 1 skeleton. Chat UI is not implemented yet (Phase 6).")
