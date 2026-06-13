import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ---------- CSS ----------
st.markdown("""
<style>

.main{
background:#0F172A;
}

.title{
text-align:center;
font-size:42px;
font-weight:bold;
color:#4F46E5;
}

.chat{

display:inline-block;

max-width:70%;

padding:12px 18px;

border-radius:18px;

margin:8px 0;

word-wrap:break-word;

font-size:16px;
}

.user{

background:#4F46E5;

color:white;

margin-left:auto;

display:block;

width:fit-content;

text-align:left;

border-bottom-right-radius:5px;
}

.stTextInput input{

border-radius:25px;

height:50px;

padding-left:20px;

font-size:16px;
}

div.stButton > button{

height:50px;

border-radius:50%;

font-size:24px;

background:#4F46E5;

color:white;

border:none;
}

.bot{

background:#E2E8F0;

color:black;

display:block;

width:fit-content;

border-bottom-left-radius:5px;
}

div.stButton > button{
width:100%;
height:50px;
border-radius:15px;
background:#4F46E5;
color:white;
font-size:18px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
"<div class='title'>🤖 AI FAQ Chatbot</div>",
unsafe_allow_html=True
)

# ---------- Load Data ----------
data = pd.read_csv("faq.csv")

questions = data["question"]

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(
questions
)

# ---------- Session ----------
if "messages" not in st.session_state:
    st.session_state.messages=[]

# ---------- Function ----------
def get_answer(query):

    query_vector = vectorizer.transform(
        [query]
    )

    similarity = cosine_similarity(
        query_vector,
        vectors
    )

    index = similarity.argmax()

    score = similarity[0][index]

    if score > 0.15:
        return data.iloc[index]["answer"]

    return "Sorry, I don't know this yet."

# ---------- Show Messages ----------
for role,msg in st.session_state.messages:

    cls="user" if role=="user" else "bot"

    st.markdown(
f"""
<div class="chat {cls}">
{msg}
</div>
""",
unsafe_allow_html=True
)

# ---------- Input ----------
user_input = st.chat_input(
"Type your message..."
)

send = st.button("Send")

if user_input or send:

    if user_input:

        st.session_state.messages.append(
            ("user",user_input)
        )

        with st.spinner(
            "🤖 Typing..."
        ):
            time.sleep(1.5)

        answer = get_answer(
            user_input
        )

        st.session_state.messages.append(
            ("bot",answer)
        )

        st.rerun()