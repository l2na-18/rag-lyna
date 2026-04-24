# Hunter × Hunter RAG Chatbot: Architecture & Implementation Report

## Executive Summary
This project presents a Retrieval-Augmented Generation (RAG) conversational AI specialized in the *Hunter × Hunter* (HxH) universe. The system integrates deep domain lore with high-performance character statistics and development team metadata, providing an expert-level interface for fans and judges alike. Built using a modern technical stack—including LangChain, Streamlit, and Google Gemini—the chatbot features native multimodal capabilities (STT/TTS and Vision) and strict domain guardrails.

## 1. Technical Architecture
The system follows a classic RAG pattern, augmented for multimodal interaction:

*   **Logic Layer**: LangChain orchestrates the retrieval and generation phases.
*   **Vector Engine**: FAISS (Facebook AI Similarity Search) provides high-speed semantic retrieval.
*   **Embeddings**: Google `gemini-embedding-2` transforms text chunks into 3072-dimensional vector space.
*   **Reasoning Model**: Google `gemini-flash-lite-latest` handles contextual synthesis and conversation flow.
*   **User Interface**: Streamlit provides a responsive, production-ready frontend for web browsers.

## 2. Knowledge Base & Data Ingestion
The chatbot's expertise is derived from three primary data pillars:
1.  **General Domain Lore**: Comprehensive summaries of Nen systems, world geography (The Dark Continent), and arc histories.
2.  **Character Combat Stats**: A specialized dataset (sourced from HxH-stats) containing numerical values for STR, SPD, INT, and NEN for over 80 major and minor characters.
3.  **Team Metadata**: Specific profiles for the development team at **Esi sba** (Higher School of Computer Science Sidi Bel Abbes).

### Processing Pipeline
*   **Chunking**: Documents were split using `RecursiveCharacterTextSplitter` into 1000-token segments with a 100-token overlap to maintain table consistency and context.
*   **History Awareness**: A standalone question generator reformulates user queries based on chat memory to ensure follow-up questions (e.g., "What about his Nen type?") remain semantically accurate.

## 3. Bonus Features & Multimodal Integration
To exceed the standard RAG requirements, the following modules were added:

*   **🖼️ LLM Vision Recognition**: Users can upload images of HxH characters. The model performs zero-shot character identification before responding.
*   **🎙️ Native Speech-to-Text**: Integration with `st.audio_input` allows users to record voice queries, which are processed by Gemini’s native audio understanding.
*   **🔊 Automatic Text-to-Speech**: Responses are dynamically converted to MP3 format via the `gTTS` library and played back automatically, creating an immersive experience.

## 4. Privacy, Licensing & Alignment
*   **Domain Alignment**: A rigid system prompt acts as a hard guardrail, forcing the model to politely refuse non-HxH or non-team-related topics (e.g., general world history, other anime, or coding help).
*   **Data Integrity**: Used CC-licensed and fan-curated data sources with appropriate attribution.
*   **Team Privacy**: Only relevant academic and professional information regarding the Esi sba team is exposed through the RAG context.

## 5. Deployment Guide
The system is optimized for **Streamlit Community Cloud**.
1.  **Repository Structure**: Optimized for one-click deployment.
2.  **Environment**: Controlled via `.env` for secure API key management.
3.  **Persistence**: The FAISS index is stored locally but can be regenerated dynamically if the knowledge base expands.

---
**Prepared by the Esi sba Development Team**  
*Lyna Lakehal, Mouffokes Mohamed El Habibe, Kacimi Nadjiba, Naila Sirine Achour, Reda Bouhennouche.*
