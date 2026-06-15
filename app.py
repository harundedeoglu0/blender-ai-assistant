import streamlit as st
import chromadb
from openai import OpenAI

# OpenAI API Key
client_openai = OpenAI(
     api_key=st.secrets["OPENAI_API_KEY"]
)

# Sayfa ayarları
st.set_page_config(
    page_title="Blender AI Assistant",
    page_icon="🤖"
)

st.title("🤖 Blender AI Assistant")
st.caption("RAG Tabanlı Blender Dokümantasyon Asistanı")

# Mod seçimi
mode = st.radio(
    "Çalışma Modu",
    ["Saf RAG", "RAG + GPT"]
)

# Sohbet geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

try:

    client_db = chromadb.PersistentClient(path="vector_db")
    collection = client_db.get_collection("blender_docs")

    st.success("Sistem hazır.")

    question = st.text_input("Blender hakkında soru sor:")

    if st.button("Sor"):

        if question.strip() == "":
            st.warning("Lütfen bir soru giriniz.")

        else:

            results = collection.query(
                query_texts=[question],
                n_results=1
            )

            context = results["documents"][0][0]

            # SAF RAG
            if mode == "Saf RAG":

                answer = context

            # RAG + GPT
            else:

                prompt = f"""
Sen Blender konusunda uzman bir asistansın.

Aşağıdaki dokümanı kullanarak cevap ver.

Doküman:
{context}

Kullanıcı Sorusu:
{question}

Türkçe ve açıklayıcı cevap ver.
"""

                response = client_openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                answer = response.choices[0].message.content

            st.session_state.messages.append(
                {
                    "mode": mode,
                    "question": question,
                    "answer": answer,
                    "context": context
                }
            )

    if st.session_state.messages:

        st.subheader("💬 Sohbet Geçmişi")

        for msg in reversed(st.session_state.messages):

            st.markdown(f"**Mod:** {msg['mode']}")
            st.markdown(f"**❓ Soru:** {msg['question']}")
            st.markdown(f"**🤖 Cevap:** {msg['answer']}")

            with st.expander("Kullanılan Doküman"):
                st.write(msg["context"])

            st.divider()

except Exception as e:
    st.error(f"Hata: {e}")