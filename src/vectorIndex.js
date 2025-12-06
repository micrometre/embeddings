/**
 * Browser-compatible Vector Index using IndexedDB for persistence
 * Implements FAISS-like functionality with cosine similarity search
 */

export class VectorIndex {
  constructor(dimensions) {
    this.dimensions = dimensions;
    this.vectors = [];
    this.metadata = [];
  }

  /**
   * Add a vector with associated metadata
   * @param {number[]} vector - The embedding vector
   * @param {object} meta - Associated metadata (id, text, etc.)
   */
  add(vector, meta = {}) {
    if (vector.length !== this.dimensions) {
      throw new Error(`Vector dimension mismatch: expected ${this.dimensions}, got ${vector.length}`);
    }
    this.vectors.push(new Float32Array(vector));
    this.metadata.push(meta);
  }

  /**
   * Compute cosine similarity between two vectors
   * @param {Float32Array|number[]} a 
   * @param {Float32Array|number[]} b 
   * @returns {number} Similarity score between -1 and 1
   */
  cosineSimilarity(a, b) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
  }

  /**
   * Search for the k most similar vectors
   * @param {number[]} queryVector - The query embedding
   * @param {number} k - Number of results to return
   * @returns {Array<{score: number, index: number, ...metadata}>}
   */
  search(queryVector, k = 5) {
    if (queryVector.length !== this.dimensions) {
      throw new Error(`Query dimension mismatch: expected ${this.dimensions}, got ${queryVector.length}`);
    }

    const query = new Float32Array(queryVector);
    const scores = [];

    for (let i = 0; i < this.vectors.length; i++) {
      const score = this.cosineSimilarity(query, this.vectors[i]);
      scores.push({
        score,
        index: i,
        ...this.metadata[i]
      });
    }

    // Sort by similarity (descending) and return top k
    scores.sort((a, b) => b.score - a.score);
    return scores.slice(0, k);
  }

  /**
   * Get the number of vectors in the index
   * @returns {number}
   */
  size() {
    return this.vectors.length;
  }

  /**
   * Save index to IndexedDB
   * @param {string} name - Index name
   */
  async save(name) {
    const data = {
      dimensions: this.dimensions,
      vectors: this.vectors.map(v => Array.from(v)),
      metadata: this.metadata
    };

    return new Promise((resolve, reject) => {
      const request = indexedDB.open('VectorIndexDB', 1);

      request.onerror = () => reject(request.error);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains('indices')) {
          db.createObjectStore('indices', { keyPath: 'name' });
        }
      };

      request.onsuccess = (event) => {
        const db = event.target.result;
        const transaction = db.transaction(['indices'], 'readwrite');
        const store = transaction.objectStore('indices');

        const putRequest = store.put({ name, ...data });
        putRequest.onsuccess = () => resolve(true);
        putRequest.onerror = () => reject(putRequest.error);
      };
    });
  }

  /**
   * Load index from IndexedDB
   * @param {string} name - Index name
   * @returns {boolean} Whether loading was successful
   */
  async load(name) {
    return new Promise((resolve) => {
      const request = indexedDB.open('VectorIndexDB', 1);

      request.onerror = () => resolve(false);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains('indices')) {
          db.createObjectStore('indices', { keyPath: 'name' });
        }
      };

      request.onsuccess = (event) => {
        const db = event.target.result;
        
        if (!db.objectStoreNames.contains('indices')) {
          resolve(false);
          return;
        }

        const transaction = db.transaction(['indices'], 'readonly');
        const store = transaction.objectStore('indices');
        const getRequest = store.get(name);

        getRequest.onsuccess = () => {
          const data = getRequest.result;
          if (data) {
            this.dimensions = data.dimensions;
            this.vectors = data.vectors.map(v => new Float32Array(v));
            this.metadata = data.metadata;
            resolve(true);
          } else {
            resolve(false);
          }
        };

        getRequest.onerror = () => resolve(false);
      };
    });
  }

  /**
   * Clear the index
   */
  clear() {
    this.vectors = [];
    this.metadata = [];
  }

  /**
   * Delete index from IndexedDB
   * @param {string} name - Index name
   */
  async delete(name) {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('VectorIndexDB', 1);

      request.onerror = () => reject(request.error);

      request.onsuccess = (event) => {
        const db = event.target.result;
        const transaction = db.transaction(['indices'], 'readwrite');
        const store = transaction.objectStore('indices');

        const deleteRequest = store.delete(name);
        deleteRequest.onsuccess = () => resolve(true);
        deleteRequest.onerror = () => reject(deleteRequest.error);
      };
    });
  }
}
