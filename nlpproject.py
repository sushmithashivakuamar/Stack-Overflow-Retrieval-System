from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import json
import translatepy
import re
import os

# Establish the base directory for relative file paths
base_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to the Annoy index and JSON data file
qa_index_path = os.path.join(base_dir, 'qa_index.ann')
qa_data_path = os.path.join(base_dir, 'qa_data.json')

# Initialize the Flask application
app = Flask(__name__)

# Initialize the translator object for translating search queries
translator = translatepy.Translator()

# Load the Annoy index for fast similarity search
embedding_dim = 384  # Dimensionality of the embeddings used in the index
index = AnnoyIndex(embedding_dim, 'angular')  # Using angular distance for similarity
index.load(qa_index_path)  # Load the pre-built Annoy index

# Load the Sentence Transformer model for generating text embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load questions and answers data from a JSON file
with open(qa_data_path, 'r') as file:
    qa_data = [json.loads(line) for line in file]  # Deserialize each line in the JSON file into a Python object

# Define the route for the main page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':  # If the request method is POST, handle the form submission
        search_term = request.form['search']  # Get the search term from the form
        # Translate the search term to English
        translated_search_term = translator.translate(search_term, "English").result
        # Encode the translated search term to get the query embedding
        query_embedding = model.encode(translated_search_term)
        k = 5  # Number of nearest neighbors to find
        # Retrieve indices of the k nearest neighbors from the index
        nearest_indices = index.get_nns_by_vector(query_embedding, k)

        # Retrieve details from qa_data using the nearest indices found
        results = [qa_data[idx] for idx in nearest_indices]
        return render_template('index.html', results=results)  # Render the results in the index.html template
    return render_template('index.html')  # If not a POST request, simply render the page without results

# Define a custom template filter to remove newline characters from text
@app.template_filter('remove_newlines')
def remove_newlines(text):
    if isinstance(text, list):
        return ''.join(re.sub(r'\\n', '', str(item)) for item in text)  # Remove newlines from each item if text is a list
    return re.sub(r'\\n', '', text)  # Remove newlines from text if it is not a list

# Define a route for handling text feedback
@app.route('/feedback-text', methods=['POST'])
def handle_text_feedback():
    data = request.get_json()  # Get the JSON data sent with the POST request
    print(f"Text feedback received: {data['feedback']}")  # Log the feedback received
    # Process and store this feedback as needed (implementation not shown)
    return jsonify({'message': 'Your feedback has been received. Thank you!'})  # Return a confirmation message

# Entry point for running the Flask application
if __name__ == '__main__':
    app.run(debug=True)  # Run the app with debugging enabled
