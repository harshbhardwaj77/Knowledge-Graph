import os
import PyPDF2
from gensim.models import KeyedVectors
import networkx as nx
import matplotlib.pyplot as plt
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

# Load Word2Vec model
def load_word_embeddings(path):
    return KeyedVectors.load_word2vec_format(path, binary=True)

# Generate embeddings for the text
def get_word_embeddings(text, model):
    words = text.split()
    embeddings = {word: model[word] for word in words if word in model}
    return embeddings

# Create a knowledge graph with specific prominent nodes
def create_knowledge_graph(embeddings, important_terms):
    graph = nx.Graph()
    for word in important_terms:
        if word in embeddings:
            graph.add_node(word, embedding=embeddings[word])
    
    # Example: Connect prominent nodes with a dummy relationship
    for i in range(len(important_terms) - 1):
        if important_terms[i] in graph and important_terms[i+1] in graph:
            graph.add_edge(important_terms[i], important_terms[i+1])

    return graph

# Plot the knowledge graph
def plot_knowledge_graph(graph):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, with_labels=True, node_size=5000, node_color="lightblue", font_size=10, font_weight="bold")
    plt.show()

# Query Gemini API to enhance the graph
def query_gemini_api(prompt):
    # Load API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("API key is missing. Set the GOOGLE_API_KEY environment variable.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# Enhance the graph with information from Gemini API
def enhance_graph_with_gemini(graph, api_query_template):
    for node in graph.nodes():
        prompt = api_query_template.format(node=node)
        response_text = query_gemini_api(prompt)
        # Process the response_text to extract new nodes or edges and update the graph
        # Note: You need to parse the response_text and identify new nodes/edges to add to the graph.

if __name__ == "__main__":
    # Replace with the name of your PDF file
    pdf_path = "C:/Users/XX/Downloads/borg.pdf"  # Path to your PDF file
    
    # Word2Vec model file is in the current working directory
    word2vec_model_path = "GoogleNews-vectors-negative300.bin"  # Path to your Word2Vec model
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Load Word2Vec model
    model = load_word_embeddings(word2vec_model_path)
    
    # Tokenize and embed text
    embeddings = get_word_embeddings(text, model)
    
    # List of prominent terms to include in the knowledge graph
    important_terms = [
        "Borg", "Google", "Kubernetes", "cluster", "BorgMaster", "Borglet", "task", 
        "job", "scheduler", "cell", "resource", "allocation", "priority", "quota", 
        "admission control", "monitoring", "fault recovery"
    ]
    
    # Create and plot the initial knowledge graph with only prominent nodes
    graph = create_knowledge_graph(embeddings, important_terms)
    plot_knowledge_graph(graph)
    
    # Enhance the graph with the Gemini API
    api_query_template = "Tell me more about the relationship between {Borg} and other concepts."
    enhance_graph_with_gemini(graph, api_query_template)
    
    # Plot the enhanced knowledge graph
    plot_knowledge_graph(graph)
