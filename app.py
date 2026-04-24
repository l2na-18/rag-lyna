import os
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import load_and_vectorize_data, get_conversational_rag_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
from gtts import gTTS
import io

# Set up page configurations
st.set_page_config(page_title="Hunter x Hunter RAG Bot", page_icon="🎣")

# Load environment variables
load_dotenv()
print(f"DEBUG APP: GOOGLE_API_KEY from env: {os.environ.get('GOOGLE_API_KEY', 'MISSING')[:10]}...")

# Initialize API Key
if not os.environ.get("GOOGLE_API_KEY"):
    st.sidebar.error("Please add your GOOGLE_API_KEY to the .env file")
    st.stop()

# Initialize session state for memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Truncate memory to the last 10 interactions (to satisfy the 5-10 memory requirement)
def get_chat_history():
    recent_messages = st.session_state.messages[-10:]
    langchain_history = []
    for msg in recent_messages:
        if msg["role"] == "user":
            langchain_history.append(HumanMessage(content=msg["content"]))
        else:
            langchain_history.append(AIMessage(content=msg["content"]))
    return langchain_history

@st.cache_resource(show_spinner=False)
def initialize_lore_search():
    # This now returns a TF-IDF retriever, not a vectorstore
    retriever = load_and_vectorize_data()
    return get_conversational_rag_chain(retriever)

st.title("🎣 Hunter x Hunter AI Chatbot")
st.markdown("Ask me anything about HxH or about the team that built me! I'll refuse to talk about anything else.")

with st.spinner("Initializing Knowledge Base..."):
    rag_chain = initialize_lore_search()

# Sidebar for multimodal bonus features
st.sidebar.header("Bonus Features")

st.sidebar.subheader("Image Recognition")
uploaded_image = st.sidebar.file_uploader("Upload a Character Image", type=["png", "jpg", "jpeg"])
if uploaded_image:
    st.sidebar.image(uploaded_image, caption="Image ready for analysis.", use_column_width=True)

st.sidebar.subheader("Voice Input (STT)")
audio_input = st.sidebar.audio_input("Record your question")

st.sidebar.subheader("Voice Output (TTS)")
enable_tts = st.sidebar.checkbox("Enable voice responses")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input widget
user_input = st.chat_input("Ask a question about HxH...")

# Handle Text Input or Audio input
if user_input or audio_input:
    # Determine the context of the user message
    if audio_input and not user_input:
        user_msg_display = "🎤 (Audio Recording Submitted)"
        user_query = "Please answer the question asked in this audio clip. You are an expert on Hunter x Hunter."
    else:
        user_msg_display = user_input
        user_query = user_input

    # IDENTITY INTERCEPTION: Ensure the team always gets credit
    identity_keywords = ["who built you", "who developed you", "who created you", "who are your developers", "who is your creator", "who developed u", "who built u"]
    is_identity_query = any(kw in user_query.lower() for kw in identity_keywords)

    # Display the user message immediately
    with st.chat_message("user"):
        st.markdown(user_msg_display)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            if is_identity_query:
                # Bypass LLM and pull directly from our source of truth
                with open("data/team_info.txt", "r") as f:
                    answer = f.read()
            # Use raw Gemini if Image or Audio is provided
            elif uploaded_image is not None or audio_input is not None:
                llm = ChatGoogleGenerativeAI(
                    model="models/gemini-flash-lite-latest",
                    google_api_key=os.environ["GOOGLE_API_KEY"],
                    request_timeout=120
                )
                
                # Refined identity prompt: Share team info ONLY if asked.
                multimodal_identity_prompt = (
                    "You are an expert Hunter x Hunter AI assistant created by a student team from Esi sba (Higher School of Computer Science Sidi Bel Abbes).\n"
                    "Team Members (Only list these if the user explicitly asks who created you or who is on your team):\n"
                    "- Lyna Lakehal\n"
                    "- Mouffokes Mohamed El Habibe\n"
                    "- Kacimi Nadjiba\n"
                    "- Naila Sirine Achour\n"
                    "- Reda Bouhennouche\n"
                    "Do NOT mention Google/OpenAI. Answer only based on HxH lore and your development team context.\n"
                    "If the question is about an image or audio, identify the HxH content first.\n\n"
                    "User Question: " + user_query
                )
                
                content_parts = [
                    {"type": "text", "text": multimodal_identity_prompt}
                ]
                
                if uploaded_image:
                    encoded_image = base64.b64encode(uploaded_image.getvalue()).decode('utf-8')
                    content_parts.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"})
                    
                if audio_input:
                    encoded_audio = base64.b64encode(audio_input.getvalue()).decode('utf-8')
                    content_parts.append({
                        "type": "media", 
                        "mime_type": "audio/wav", 
                        "data": encoded_audio
                    })
                    
                multimodal_message = HumanMessage(content=content_parts)
                try:
                    response = llm.invoke([multimodal_message])
                    answer = response.content
                    
                    # Clean up multimodal response (remove brackets/metadata)
                    if isinstance(answer, list):
                        answer = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in answer])
                except Exception as e:
                    # Fallback to a clearer debug message
                    answer = f"I'm sorry, I encountered an error analyzing your multimodal input. (Error: {str(e)})"
            else:
                # Ordinary RAG text chain
                try:
                    history = get_chat_history()
                    response = rag_chain.invoke({"input": user_query, "chat_history": history})
                    answer = response["answer"]
                    if not answer:
                        answer = "I searched the lore but couldn't find a specific answer for that. Tell me more about what you're looking for!"
                except Exception as e:
                    answer = f"I'm sorry, I encountered a temporary mental block. (Error: {str(e)})"
            
            st.markdown(answer)
            # Append BOTH messages to state at the end so history is clean for the RAG call
            st.session_state.messages.append({"role": "user", "content": user_msg_display})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Generate Audio TTS
            if enable_tts:
                try:
                    tts = gTTS(text=answer, lang='en', slow=False)
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    st.audio(fp.getvalue(), format="audio/mp3", autoplay=True)
                except Exception as e:
                    st.error("Failed to generate audio.")
