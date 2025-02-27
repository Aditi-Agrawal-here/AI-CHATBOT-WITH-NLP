import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load corpus from external file
def load_corpus(file_path="corpus.txt"):
    corpus = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    key, value = line.strip().split("|", 1)
                    corpus[key.lower()] = value
    except FileNotFoundError:
        print("Error: corpus.txt not found! Make sure it's in the same directory.")
    return corpus

corpus = load_corpus()

# Preprocess corpus
corpus_keys = list(corpus.keys())
corpus_values = list(corpus.values())
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus_keys)

# Chat history for simple memory
chat_history = []
max_history = 3  # Number of interactions to remember

def chatbot_response(user_input):
    user_input = user_input.lower()
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, X)
    best_match_index = similarities.argmax()
    best_score = similarities[0, best_match_index]
    
    # Set a threshold for similarity matching
    if best_score > 0.3:
        response = corpus_values[best_match_index]
    else:
        response = "I'm not sure I understand. Can you rephrase?"
    
    # Store user input & response in chat history
    chat_history.append((user_input, response))
    if len(chat_history) > max_history:
        chat_history.pop(0)
    
    return response

# Main chat loop
def chat():
    print("Chatbot: Hello! Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['bye', 'exit', 'quit']:
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = chatbot_response(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    chat()
