# Browser Embeddings & Vector Search

This project demonstrates how to perform sentence embeddings and vector similarity search entirely in the browser using `@huggingface/transformers` (Transformers.js) and a custom vector index implementation.

## Features

- **In-Browser Embeddings**: Uses Hugging Face Transformers.js to generate embeddings directly in the client, with no server-side inference required.
- **Vector Search**: Implements a custom vector index with cosine similarity search to find semantically similar text.
- **🎯 Interactive ML Quiz**: Take an interactive quiz with **semantic answer evaluation** - your answers are graded using AI embeddings, not just exact text matching!
- **Smart Answer Grading**: Paraphrased or differently-worded answers still get credit based on semantic similarity.
- **Interactive UI**: Visualizes the model loading process, indexing status, and search results.
- **ML Quiz Dataset**: Includes a sample dataset of Machine Learning quiz questions to demonstrate the search capabilities.

## Tech Stack

- **Vite**: Fast frontend build tool and development server.
- **@huggingface/transformers**: Library for running state-of-the-art machine learning models in the browser using WebAssembly (WASM).
- **Vanilla JavaScript**: Built with standard ES modules and modern JavaScript features.
- **IndexedDB**: Browser storage for caching embeddings and vector indices.
- **CSS**: Custom styling with a dark theme.

## Getting Started

### Prerequisites

- Node.js (v16 or higher recommended)
- npm or yarn

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd browser-embeddings
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

### Running the Application

Start the development server:

```bash
npm run dev
```

Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:5173`).

### Building for Production

To build the project for production:

```bash
npm run build
```

To preview the production build:

```bash
npm run preview
```

## Project Structure

- `src/main.js`: The main entry point. Handles application state, UI updates, model loading, quiz logic, and search interactions.
- `src/vectorIndex.js`: A custom class that manages vector storage and implements the cosine similarity algorithm for search.
- `index.html`: The main HTML file containing the application layout and styles.

## How It Works

1. **Model Loading**: When the application starts, it downloads a lightweight sentence embedding model (`Xenova/all-MiniLM-L6-v2`) using `@huggingface/transformers`. This model runs entirely in your browser using WebAssembly.

2. **Indexing**:
   - The application takes a predefined list of ML questions.
   - For each question, it generates a 384-dimensional embedding vector using the model.
   - These vectors, along with their metadata (the original text), are stored in the `VectorIndex`.
   - The index is also saved to `IndexedDB` so it doesn't need to be rebuilt on every page reload.

3. **Search Process**:
   - When a user enters a query, the model generates an embedding vector for that query.
   - The `VectorIndex` calculates the **cosine similarity** between the query vector and every vector in the index.
   - **Cosine Similarity** measures the cosine of the angle between two vectors. A value closer to 1 indicates high similarity (small angle), while a value closer to -1 indicates dissimilarity.
   - The results are sorted by their similarity score in descending order.
   - The top 3 most similar questions are returned and displayed to the user.

4. **Interactive Quiz with Semantic Grading**:
   - The quiz presents ML questions and accepts free-text answers.
   - When you submit an answer, it's converted to an embedding vector.
   - Your answer embedding is compared to the expected answer embedding using cosine similarity.
   - **Scoring System**:
     | Similarity | Rating | Points |
     |------------|--------|--------|
     | ≥ 85% | Excellent! | 1.0 |
     | ≥ 70% | Good! | 0.8 |
     | ≥ 55% | Partial | 0.5 |
     | < 55% | Try Again | 0 |
   - This allows paraphrased or differently-worded correct answers to receive appropriate credit.

## Why Semantic Answer Matching?

Traditional quiz systems require exact text matching, which is frustrating when you know the concept but phrase it differently. By using embeddings:

- ✅ "Neural networks learn by adjusting weights based on error gradients" matches well with the expected answer about backpropagation
- ✅ Answers in your own words get credit if they're semantically correct
- ✅ No need for predefined answer variants or regex patterns

## License

[MIT](LICENSE)
