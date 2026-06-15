from sentence_transformers import SentenceTransformer
import chromadb

# embedding modeli
model = SentenceTransformer("all-MiniLM-L6-v2")

# veritabanı
client = chromadb.PersistentClient(path="vector_db")

collection = client.get_or_create_collection(
    name="blender_docs"
)

# dosyayı oku
with open("data/blender_docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

# basit parçalama
chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

# embedding oluştur
embeddings = model.encode(chunks)

# kaydet
for i, chunk in enumerate(chunks):
    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embeddings[i].tolist()]
    )

print("Veriler başarıyla yüklendi.")