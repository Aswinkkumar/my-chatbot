# import streamlit as st
# from huggingface_hub import InferenceClient

# MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# st.title("🤖 My Chatbot")

# client = InferenceClient(model=MODEL, token=st.secrets["HF_TOKEN"])

# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "system", "content": "You are a helpful assistant."}
#     ]

# for m in st.session_state.messages:
#     if m["role"] != "system":
#         with st.chat_message(m["role"]):
#             st.write(m["content"])

# user_input = st.chat_input("Ask me something")
# if user_input:
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.write(user_input)

#     with st.chat_message("assistant"):
#         try:
#             with st.spinner("Thinking..."):
#                 out = client.chat_completion(
#                     messages=st.session_state.messages,
#                     max_tokens=256,
#                 )
#             reply = out.choices[0].message.content
#         except Exception as e:
#             reply = "Sorry, something went wrong."
#             st.error(str(e)[:200])
#         st.write(reply)

#     st.session_state.messages.append({"role": "assistant", "content": reply})




import datetime
import streamlit as st
from huggingface_hub import InferenceClient
from ddgs import DDGS

MODEL = "meta-llama/Llama-3.1-8B-Instruct"   # use whichever model works for you
MAX_INPUT_CHARS = 500

st.title("🤖 My Chatbot")
use_search = st.sidebar.checkbox("Use web search (recent info)", value=True)

client = InferenceClient(model=MODEL, provider="auto", token=st.secrets["HF_TOKEN"])

def web_search(query, n=5):
    try:
        results = DDGS().text(query, max_results=n)
        return "\n\n".join(
            f"[{i+1}] {r['title']}\n{r['body']}\nSource: {r['href']}"
            for i, r in enumerate(results)
        )
    except Exception:
        return ""

def system_prompt():
    today = datetime.date.today().strftime("%B %d, %Y")
    return (
        f"You are a helpful assistant. Today's date is {today}. "
        "Your training data is outdated, so for recent events rely on the "
        "provided web search results. Cite sources by number like [1]. "
        "If the results don't contain the answer, say you don't know. "
        "Never invent facts."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_input = st.chat_input("Ask me something")
if user_input and len(user_input) <= MAX_INPUT_CHARS:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and thinking..."):
                context = web_search(user_input) if use_search else ""

                # Send history, but attach search results only to the latest question
                msgs = [{"role": "system", "content": system_prompt()}]
                msgs += st.session_state.messages[:-1][-6:]   # last few turns
                if context:
                    msgs.append({
                        "role": "user",
                        "content": f"Web search results:\n{context}\n\nQuestion: {user_input}",
                    })
                else:
                    msgs.append({"role": "user", "content": user_input})

                out = client.chat_completion(messages=msgs, max_tokens=500, temperature=0.2)
            reply = out.choices[0].message.content
        except Exception as e:
            reply = "Sorry, something went wrong."
            st.error(str(e)[:200])
        st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})