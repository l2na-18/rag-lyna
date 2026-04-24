import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Set API Key
api_key = os.environ.get("GOOGLE_API_KEY")

# Safety settings
safety_settings = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
}

class ManualSearchEngine:
    """A high-performance TF-IDF search engine using Scikit-Learn and Numpy."""
    def __init__(self, documents):
        self.documents = documents
        self.texts = [doc.page_content for doc in documents]
        # Removed stop_words to ensure short HxH names aren't filtered
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query, k=5):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_indices]

def load_and_vectorize_data(data_dir="./data"):
    # Load and split documents
    loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    # Initialize our robust manual search engine
    return ManualSearchEngine(splits)

class ManualRAGChain:
    """A RAG chain that uses our custom ManualSearchEngine for 100% stability."""
    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-flash-lite-latest",
            temperature=0.3,
            google_api_key=api_key,
            safety_settings=safety_settings
        )
        self.system_prompt = (
            "You are a STRICT Hunter x Hunter lore specialist developed by the Esi sba student team (Lyna Lakehal, Mouffokes Mohamed El Habibe, Kacimi Nadjiba, Naila Sirine Achour, and Reda Bouhennouche).\n\n"
            "MANDATORY RULE: You ONLY discuss Hunter x Hunter or your creators from Esi sba. "
            "If a user asks about real-world politics, history, other anime, or anything unrelated to HxH, you must politely but firmly REFUSE to answer. "
            "Do not speculate on which Nen type a real-world person would have. Just say: 'I only discuss things related to the Hunter x Hunter universe and the Esi sba team.'\n\n"
            "Context for lore questions:\n{context}"
        )

    def invoke(self, inputs):
        user_input = inputs["input"]
        chat_history = inputs.get("chat_history", [])
        
        # Search the knowledge base using our manual engine
        docs = self.search_engine.search(user_input)
        context_str = "\n\n".join([d.page_content for d in docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt.format(context=context_str)),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"input": user_input, "chat_history": chat_history})
        
        # Ensure we return a clean string
        answer = response.content
        if isinstance(answer, list):
            answer = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in answer])
        
        return {"answer": str(answer)}

def get_conversational_rag_chain(search_engine):
    return ManualRAGChain(search_engine)
