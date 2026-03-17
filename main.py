import os
from google import genai
from google.genai import types

api_key = os.environ["GOOGLE_API_KEY"]
client = genai.Client(api_key=api_key)

with open("gel.jpg", "rb") as f:
    image_data = f.read()

result = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents=types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
)

embedding = result.embeddings[0].values
print(f"Embedding length: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
