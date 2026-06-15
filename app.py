import streamlit as st
import chromadb




st.set_page_config(
    page_title="Blender AI Assistant",
    page_icon="🤖"
)

st.title("🤖 Blender AI Assistant")
st.write("TEST")
st.caption("RAG Tabanlı Blender Dokümantasyon Asistanı")

mode = st.radio(
    "Çalışma Modu",
    ["Saf RAG", "RAG + GPT"]
)

if "messages" not in st.session_state:
    st.session_state.messages = []

try:

    client_db = chromadb.PersistentClient(path="vector_db")
    collection = client_db.get_collection("blender_docs")

    st.success("Sistem hazır.")

    question = st.text_input("Blender hakkında soru sor:")

    if st.button("Sor"):

        if question.strip():

            results = collection.query(
                query_texts=[question],
                n_results=1
            )

            # Güvenli doküman alma
            context = ""

            if (
                "documents" in results
                and len(results["documents"]) > 0
                and len(results["documents"][0]) > 0
            ):
                context = results["documents"][0][0]

            else:
                st.error("Doküman bulunamadı.")
                st.stop()

            if mode == "Saf RAG":

                answer = context

            else:

                prompt = f"""
Aşağıdaki Blender dokümanını kullanarak cevap ver.

Doküman:
{context}

Soru:
{question}

Türkçe ve açıklayıcı cevap ver.
"""

                response = client_openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "Sen Blender konusunda uzman bir asistansın."
                        },
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
    st.exception(e)