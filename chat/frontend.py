import streamlit as st
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.title("RAG Chatbot")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "interrupted" not in st.session_state:
    st.session_state.interrupted = False
if "interrupt_value" not in st.session_state:
    st.session_state.interrupt_value = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_user_123" # Fixed ID for demo persistence

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# If interrupted, show the feedback form INSTEAD of the chat input
if st.session_state.interrupted:
    with st.form("feedback_form"):
        st.subheader("Action Required")
        st.write("The agent has paused for your review.")

        # Display details from the interrupt snapshot
        value = st.session_state.interrupt_value
        if value:
            st.json(value)
        print(value)
        print("-"*100)
                         
        action = st.radio("Review Decision", ["approve", "edit", "reject"])
        query = st.text_input("Query (required for edit)", value="")
        filter = st.text_input("Filter (required for edit)", value="")
        step = st.text_input("Step (required for reject)", value="")
        
        submitted = st.form_submit_button("Submit Decision")
        
        if submitted:
            # map UI action to backend expected strings
            backend_action = ""
            if action == "approve":
                backend_action = "approve"
            elif action == "edit":
                backend_action = "edit"
            else:
                backend_action = "reject"
            
            with st.spinner("Resuming agent..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/chat/feedback", 
                        json={
                            "thread_id": st.session_state.thread_id, 
                            "action": backend_action, 
                            "query": query,
                            "filter": [x.strip() for x in filter.split(",")],
                            "step": step
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if data.get("status") == "interrupted":
                        st.session_state.interrupted = True
                        st.session_state.interrupt_value = data.get("value")
                        st.warning("Agent paused again.")
                        st.rerun()
                    else:
                        # Finished
                        st.session_state.interrupted = False
                        st.session_state.interrupt_value = None
                        
                        bot_reply = data.get("response", "No response.")
                        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error resuming chat: {e}")

# Only show chat input if NOT interrupted
else:
    if prompt := st.chat_input("What is your question?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/chat", 
                    json={
                        "messages": [{"role": "user", "content": prompt}], 
                        "thread_id": st.session_state.thread_id
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") == "interrupted":
                    st.session_state.interrupted = True
                    st.session_state.interrupt_value = data.get("value")
                    st.rerun() # Rerun to show the form
                else:
                    bot_reply = data.get("response", "No response.")
                    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                    st.chat_message("assistant").markdown(bot_reply)
                    
            except Exception as e:
                st.error(f"Error: {e}")
