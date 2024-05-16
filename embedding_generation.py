from bs4 import BeautifulSoup
import re
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer

# Define paths for the XML file and the Annoy index file
file_path = r'D:\NLP Project\stackoverflow.com-Posts\Posts.xml'
annoy_index_file = r'D:\NLP Project\Python\pythonProject3\index.ann'

def preprocess_text(text):
    # Remove HTML tags using BeautifulSoup
    text = BeautifulSoup(text, 'html.parser').get_text()
    # Remove all characters except alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert text to lowercase to maintain consistency
    return text.lower()


def create_annoy_index(file_path, limit=10000):
    qa_list = []

    # Parse the XML file and extract data from rows with PostTypeId = '1' (indicating questions)
    for event, elem in ET.iterparse(file_path, events=('end',)):
        if elem.tag == 'row' and elem.attrib.get('PostTypeId') == '1':
            # Stop processing if the limit of questions is reached
            if len(qa_list) >= limit:
                break
            # Concatenate title and body text for each question
            title = elem.attrib.get('Title', '')
            body = elem.attrib.get('Body', '')
            processed_text = preprocess_text(title + ' ' + body)
            qa_list.append(processed_text)
            # Free up memory by clearing the element from memory
            elem.clear()

    # Use TF-IDF to vectorize the text, setting the maximum number of features to 384
    vectorizer = TfidfVectorizer(max_features=384)
    vectors = vectorizer.fit_transform(qa_list)

    # Create an Annoy index with 384 dimensions using angular distance metric
    index = AnnoyIndex(384, 'angular')
    for i, vec in enumerate(vectors):
        # Add each vector to the Annoy index
        index.add_item(i, vec.toarray()[0])

    # Build the index with 10 trees for efficient querying
    index.build(10)
    # Save the index to the specified file
    index.save(annoy_index_file)
    print(f"Annoy index saved to {annoy_index_file}")

create_annoy_index(file_path, limit=10000)
