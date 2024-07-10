import os
import json
import openai
from anthropic import Anthropic
from transformers import pipeline
import asyncio
import logging
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)


class RAG:
    def __init__(self, knowledge_base_path, sentence_transformer_model='all-MiniLM-L6-v2', max_retrieved_documents=5,
                 similarity_threshold=0.7):
        self.sentence_model = SentenceTransformer(sentence_transformer_model)
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        self.index = self.build_index()
        self.max_retrieved_documents = max_retrieved_documents
        self.similarity_threshold = similarity_threshold

    def load_knowledge_base(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def build_index(self):
        embeddings = self.sentence_model.encode([item['content'] for item in self.knowledge_base])
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def retrieve(self, query):
        query_vector = self.sentence_model.encode([query])
        distances, indices = self.index.search(query_vector, self.max_retrieved_documents)
        retrieved = [
            self.knowledge_base[i] for i, dist in zip(indices[0], distances[0])
            if dist <= self.similarity_threshold
        ]
        return retrieved


async def get_llm_response(prompt, config, rag=None):
    provider = config['provider']

    if rag:
        retrieved_context = rag.retrieve(prompt)
        context_str = "\n".join([f"- {item['content']}" for item in retrieved_context])
        augmented_prompt = f"Context from knowledge base:\n{context_str}\n\nPrompt: {prompt}"
    else:
        augmented_prompt = f"Prompt: {prompt}"

    if provider == 'openai':
        return await get_openai_response(augmented_prompt, config)
    elif provider == 'anthropic':
        return await get_anthropic_response(augmented_prompt, config)
    elif provider == 'huggingface':
        return await get_huggingface_response(augmented_prompt, config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


async def get_openai_response(prompt, config):
    openai.api_key = os.getenv(config['models']['default']['api_key'])
    model = config['models']['fine_tuned' if config['use_fine_tuned'] else 'default']['name']

    response = await openai.ChatCompletion.acreate(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config['models']['default']['max_tokens'],
        temperature=config['models']['default']['temperature']
    )
    return response.choices[0].message.content.strip()


async def get_anthropic_response(prompt, config):
    anthropic = Anthropic(api_key=os.getenv(config['alternative_providers']['anthropic']['api_key']))
    response = await anthropic.completions.create(
        model=config['alternative_providers']['anthropic']['model'],
        prompt=prompt,
        max_tokens_to_sample=config['alternative_providers']['anthropic']['max_tokens'],
        temperature=config['alternative_providers']['anthropic']['temperature']
    )
    return response.completion


async def get_huggingface_response(prompt, config):
    generator = pipeline('text-generation', model=config['alternative_providers']['huggingface']['model'])
    response = await asyncio.to_thread(
        generator,
        prompt,
        max_length=config['alternative_providers']['huggingface']['max_tokens'],
        temperature=config['alternative_providers']['huggingface']['temperature']
    )
    return response[0]['generated_text']
