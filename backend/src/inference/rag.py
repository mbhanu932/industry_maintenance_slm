import os
import re

DOCUMENT_PATH = os.getenv("RAG_DOCUMENT_PATH", "knowledge_base/documents/engine_maintenance_sop.txt")

class Retriever:
    def __init__(self):
        self.documents = []
        if os.path.exists(DOCUMENT_PATH):
            with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
                text = f.read()
            self.documents = [x.strip() for x in re.split(r"\n(?=\d+\.)", text) if x.strip()]

    def search(self, query, k=4):
        words = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
        results = []
        for doc in self.documents:
            doc_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", doc.lower()))
            score = len(words.intersection(doc_words))
            if score:
                results.append({"text": doc, "score": float(score)})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
