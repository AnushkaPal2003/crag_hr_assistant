# Run this ONCE to build the vectorstore from HR dataset
# python setup_vectorstore.py

import config
from vectorstore.setup import create_vectorstore

print("Building HR knowledge base vectorstore...")
create_vectorstore()
print("Done! Run: streamlit run app.py")
