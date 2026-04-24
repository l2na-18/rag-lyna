# Hunter x Hunter RAG Chatbot: Architecture & Implementation Report
**By: The Esi sba Student Team**

---

## 1. Project Overview
The **Hunter x Hunter AI Assistant** is a specialized, interactive application designed to serve as a high-fidelity knowledge base for the *Hunter x Hunter* universe. Unlike generic chatbots, this system was built to provide accurate, lore-specific answers while offering a multimodal experience that includes image recognition and voice interaction.

---

## 2. System Architecture
Our system follows a **Modular RAG (Retrieval-Augmented Generation)** architecture. This was designed to keep the application lightweight and responsive, specifically for hosting on Streamlit Community Cloud.

### A. Data Layer
The foundation of the bot is its knowledge base, consisting of structured text files containing:
- **Character Stats**: Detailed breakdowns of Nen types and combat ratings.
- **Lore Summaries**: Plot points, location details, and organization bios (like the Phantom Troupe).
- **Team Information**: Biographies of the student development team.

### B. The Search Engine (Local Intelligence)
To avoid the instability of external API search calls, we implemented a custom-built search algorithm:
- **Technology**: We used `Scikit-Learn` to convert our lore into a mathematical "search space" right on the server.
- **Precision**: When a user asks a question, the system uses **Cosine Similarity** to instantly find the most relevant paragraphs. This ensures the bot "reads" the right information before it even starts thinking of an answer.

---

## 3. Implementation Details

### A. Multimodal Processing (Images & Voice)
One of the core challenges was processing different types of information without slowing down the app:
1. **Image Recognition**: Using the `Gemini-1.5-Flash` vision engine, the bot analyzes character designs in real-time. It can identify subtle visual cues (like Killua’s electric sparks) to determine which specific power is being shown.
2. **Audio Pipeline**: We integrated `gTTS` for voice output and Streamlit’s audio widgets for input. This allows for a "hands-free" educational experience for the user.

### B. The Identity & Safety Framework
We implemented two layers of "Safety Checks" to keep the bot professional:
- **Identity Override**: The bot is programmed to prioritize our team biography whenever asked about its origins.
- **Domain Refusal**: To keep the project focused, we added a rule that politely blocks questions about real-world politics or non-HxH topics. This is handled by a specialized **System Prompt** that governs every interaction.

---

## 4. Deployment & Performance Optimization
Hosting an AI model on a free tier like Streamlit Cloud required significant optimization:
- **Memory Management**: We removed heavy libraries like Torch and Tensorflow, replacing them with lightweight alternatives like `Numpy` and `Scikit-Learn`. This dropped the app's boot time from minutes to under 30 seconds.
- **Error Resilience**: We added "Try-Except" blocks at every critical stage. If a specific API call fails, the bot is programmed to provide a helpful response instead of crashing, ensuring a smooth user experience.

---

## 5. Summary of Technologies
- **Front-End**: Streamlit (Python-based Web Framework)
- **Mind (LLM)**: Google Gemini 1.5 Flash
- **Memory (RAG)**: Scikit-Learn TF-IDF & Numpy
- **Senses**: Google Generative AI (Vision) & gTTS (Voice)

---
