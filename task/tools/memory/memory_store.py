import os
os.environ['OMP_NUM_THREADS'] = '1'

import json
from datetime import datetime, UTC, timedelta
import numpy as np
import faiss
from aidial_client import AsyncDial
from sentence_transformers import SentenceTransformer

from task.tools.memory._models import Memory, MemoryData, MemoryCollection


class LongTermMemoryStore:
    """
    Manages long-term memory storage for users.

    Storage format: Single JSON file per user in DIAL bucket
    - File: {user_id}/long-memories.json
    - Caching: In-memory cache with conversation_id as key
    - Deduplication: O(n log n) using FAISS batch search
    """

    DEDUP_INTERVAL_HOURS = 24

    def __init__(self, endpoint: str):
        #TODO:
        # 1. Set endpoint
        # 2. Create SentenceTransformer as model, model name is `all-MiniLM-L6-v2`
        # 3. Create cache, doct of str and MemoryCollection (it is imitation of cache, normally such cache should be set aside)
        # 4. Make `faiss.omp_set_num_threads(1)` (without this set up you won't be able to work in debug mode in `_deduplicate_fast` method
        self.endpoint = endpoint
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache: dict[str, MemoryCollection] = {}
        faiss.omp_set_num_threads(1)

    async def _get_memory_file_path(self, dial_client: AsyncDial) -> str:
        """Get the path to the memory file in DIAL bucket."""
        #TODO:
        # 1. Get DIAL app home path
        # 2. Return string with path in such format: `files/{bucket_with_app_home}/__long-memories/data.json`
        #    The memories will persist in appdata for this agent in `__long-memories` folder and `data.json` file
        #    (You will be able to check it also in Chat UI in attachments)
        home_path = await dial_client.my_appdata_home()
        return f"files/{home_path}/__long-memories/data.json"

    async def _load_memories(self, api_key: str) -> MemoryCollection:
        #TODO:
        # 1. Create AsyncDial client (api_version is 2025-01-01-preview)
        # 2. Get memory file path
        # 3. Check cache: cache is dict of str and MemoryCollection, for the key we will use `memory file path` to make
        #    it simple. Such key will be unique for user and will allow to access memories across different
        #    conversations and only user can access them. In case if cache is present return its MemoryCollection.
        # ---
        # Below is logic when cache is not present:
        # 4. Open try-except block:
        #   - in try:
        #       - download file content
        #       - in response get content and decode it with 'utf-8'
        #       - load content with `json`
        #       - create MemoryCollection (it is pydentic model, use `model_validate` method)
        #   - in except:
        #       - create MemoryCollection (it will have empty memories, set up time for updated_at, more detailed take
        #         a look at MemoryCollection pydentic model and it Fields)
        # 5. Return created MemoryCollection
        dial_client = AsyncDial(api_key=api_key, base_url=self.endpoint, api_version="2025-01-01-preview")
        memory_file_path = await self._get_memory_file_path(dial_client)
        
        if memory_file_path in self.cache:
            return self.cache[memory_file_path]
        
        try:
            file_content = await dial_client.files.download(memory_file_path)
            decoded_content = file_content.get_content().decode('utf-8')
            memories_data = json.loads(decoded_content)
            memory_collection = MemoryCollection.model_validate(memories_data)
        except Exception:
            memory_collection = MemoryCollection(memories=[], updated_at=datetime.now(UTC), last_deduplicated_at=None) 
            
        return memory_collection

    async def _save_memories(self, api_key: str, memories: MemoryCollection):
        """Save memories to DIAL bucket and update cache."""
        #TODO:
        # 1. Create AsyncDial client
        # 2. Get memory file path
        # 3. Update `updated_at` of memories (now)
        # 4. Converts memories to json string (it's pydentic model and it have model dump json method for this). Don't
        #    make any indentations because it will make file 'bigger'. Here is the point that we store all the memories
        #    in one file and 'one memory' with its embeddings takes ~6-8Kb, we expect that there are won't be more that
        #    1000 memories but anyway for 1000 memories it will be ~6-8Mb, so, we need to make at least these small
        #    efforts to make it smaller 😉
        # 5. Put to cache (kind reminder the key is memory file path)
        dial_client = AsyncDial(api_key=api_key, base_url=self.endpoint, api_version="2025-01-01-preview")
        memory_file_path = await self._get_memory_file_path(dial_client)
        memories.updated_at = datetime.now(UTC)
        memories_json = memories.model_dump_json()

        self.cache[memory_file_path] = memories
        await dial_client.files.upload(
            file=memories_json.encode('utf-8'), 
            url=memory_file_path
        )

    async def add_memory(self, api_key: str, content: str, importance: float, category: str, topics: list[str]) -> str:
        """Add a new memory to storage."""
        #TODO:
        # 1. Load memories
        # 2. Make encodings for content with embedding model.
        #    Hint: provide content as list, and after encoding get first result (encode wil return list) and convertit `tolist`
        # 3. Create Memory
        #    - for id use `int(datetime.now(UTC).timestamp())` it will provide time now as int, it will be super enough
        #      to avoid collisions. Also, we won't use id but we added it because maybe in future you will make enhanced
        #      version of long-term memory and after that it will be additional 'headache' to add such ids 😬
        # 4. Add to memories created memory
        # 5. Save memories (it is PUT request bzw, -> https://dialx.ai/dial_api#tag/Files/operation/uploadFile)
        # 6. Return information that content has benn successfully stored
        memory_collection = await self._load_memories(api_key)
        embedding = self.model.encode([content])[0].tolist()

        new_memory = MemoryData(
            id=int(datetime.now(UTC).timestamp()),
            content=content,
            importance=importance,
            category=category,
            topics=topics
        )

        memory_collection.memories.append(Memory(data=new_memory, embedding=embedding))
        await self._save_memories(api_key, memory_collection)
        return "Memory successfully stored."
          

    async def search_memories(self, api_key: str, query: str, top_k: int = 5) -> list[MemoryData]:
        """
        Search memories using semantic similarity.

        Returns:
            List of MemoryData objects (without embeddings)
        """
        #TODO:
        # 1. Load memories
        # 2. If they are empty return empty array
        # ---
        # 3. Check if they needs_deduplication, if yes then deduplicate_and_save (need to implements both of these methods)
        # 4. Make vector search (embeddings are part of memory)😈
        # 5. Return `top_k` MemoryData based on vector search
        memory_collection = await self._load_memories(api_key)

        if not memory_collection.memories:
            return []
   
        if self._needs_deduplication(memory_collection):
            memory_collection = await self._deduplicate_and_save(api_key, memory_collection)

        query_embedding = self.model.encode([query]).astype('float32')
        memory_embeddings = np.array(
            [memory.embedding for memory in memory_collection.memories]
        ).astype('float32') 

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(query_embedding)
        faiss.normalize_L2(memory_embeddings)
        
        index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
        index.add(memory_embeddings) # type: ignore

        k = min(top_k, len(memory_collection.memories))
        similarity, indices = index.search(query_embedding, k) # type: ignore

        return [memory_collection.memories[i].data for i in indices[0]]

    def _needs_deduplication(self, collection: MemoryCollection) -> bool:
        """Check if deduplication is needed (>24 hours since last deduplication)."""
        #TODO:
        # The criteria for deduplication (collection length > 10 and >24 hours since last deduplication) or
        # (collection length > 10 last deduplication is None)
        if len(collection.memories) <= 10:
            return False
        
        if collection.last_deduplicated_at is None:
            return True
        
        return datetime.now(UTC) - collection.last_deduplicated_at > timedelta(hours=self.DEDUP_INTERVAL_HOURS)

    async def _deduplicate_and_save(self, api_key: str, collection: MemoryCollection) -> MemoryCollection:
        """
        Deduplicate memories synchronously and save the result.
        Returns the updated collection.
        """
        #TODO:
        # 1. Make fast deduplication (need to implement)
        # 2. Update last_deduplicated_at as now
        # 3. Save deduplicated memories
        # 4. Return deduplicated collection
        deduplicated_memories = self._deduplicate_fast(collection.memories)
        collection.memories = deduplicated_memories
        collection.last_deduplicated_at = datetime.now(UTC)

        await self._save_memories(api_key, collection)
        return collection

    def _deduplicate_fast(self, memories: list[Memory]) -> list[Memory]:
        """
        Fast deduplication using FAISS batch search with cosine similarity.

        Strategy:
        - Find k nearest neighbors for each memory using cosine similarity
        - Mark duplicates based on similarity threshold (cosine similarity > 0.75)
        - Keep memory with higher importance
        """
        #TODO:
        # This is the hard part 🔥🔥🔥
        # You need to deduplicate memories, duplicates are the memories that have 75% similarity.
        # Among duplicates remember about `importance`, most important have more priorities to survive
        # It must be fast, it is possible to do for O(n log n), probably you can find faster way (share with community if do 😉)
        # Return deduplicated memories

        print(f"Deduplicating {len(memories)} memories...")
        memory_embeddings = np.array([memory.embedding for memory in memories]).astype('float32')
        
        faiss.normalize_L2(memory_embeddings)

        index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
        index.add(memory_embeddings) # type: ignore
        
        memory_entry_count = len(memories)
        similarity_score, indices = index.search(memory_embeddings, memory_entry_count) # type: ignore

        removed_indices = set()

        # Loop through each memory and its neighbors to identify duplicates
        for i in range(memory_entry_count):
            # Skip if this memory is already marked for removal
            if i in removed_indices:
                continue

            # Loop through neighbors of memory i
            for j in indices[i]:
                # Avoid comparing the same memory and already removed memories
                if i != j and j not in removed_indices:
                    # Calculate cosine similarity (since embeddings are normalized, we can use the similarity score directly)
                    if similarity_score[i][j] > 0.75:
                        # Mark the less important memory for removal
                        if memories[i].data.importance >= memories[j].data.importance:
                            removed_indices.add(j)
                        else:
                            removed_indices.add(i)
                        # Break after finding the first duplicate to avoid multiple removals of the same memory
                        break
        print(f"Found {len(removed_indices)} duplicates to remove.")
        deduplicated_memories = [memories[i] for i in range(memory_entry_count) if i not in removed_indices]
        return deduplicated_memories


    async def delete_all_memories(self, api_key: str, ) -> str:
        """
        Delete all memories for the user.

        Removes the memory file from DIAL bucket and clears the cache
        for the current conversation.
        """
        #TODO:
        # 1. Create AsyncDial client
        # 2. Get memory file path
        # 3. Delete file
        # 4. Return info about successful memory deletion
        dial_client = AsyncDial(api_key=api_key, base_url=self.endpoint, api_version="2025-01-01-preview")
        memory_file_path = await self._get_memory_file_path(dial_client)
        await dial_client.files.delete(memory_file_path)
        if memory_file_path in self.cache:
            del self.cache[memory_file_path]
        return "All memories have been successfully deleted."

