# Semantic Quiz

An AI-powered quiz application that grades answers using semantic similarity, running 100% in your browser with Transformers.js.

## Features

- **🎯 Semantic Answer Grading**: Your answers are graded using AI embeddings, not just exact text matching - paraphrased answers get credit!
- **🧠 In-Browser AI**: Uses Hugging Face Transformers.js to run the `all-MiniLM-L6-v2` model entirely in your browser via WebAssembly.
- **🔒 Privacy-First**: No data is sent to any server - everything runs locally.
- **⚡ Fast & Cached**: The model is cached after first load for instant startup.
- **📊 Smart Scoring**: Tiered scoring system based on cosine similarity (Excellent/Good/Partial/Try Again).
- **💡 Hints & Retry**: Get hints when stuck, retry incorrect answers to improve.

## Tech Stack

- **Vite**: Fast frontend build tool and development server.
- **@huggingface/transformers**: Library for running ML models in the browser using WebAssembly (WASM).
- **Vanilla JavaScript**: Built with standard ES modules and modern JavaScript features.
- **CSS**: Custom styling with a dark theme.

## Getting Started

### Prerequisites

- Node.js (v16 or higher recommended)
- npm or yarn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/micrometre/embeddings 
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

- `src/main.js`: Main application - handles model loading, quiz logic, semantic grading, and UI.
- `index.html`: The HTML file containing the application layout, styles, and educational content.

## How It Works

1. **Model Loading**: When the app starts, it downloads a lightweight sentence embedding model (`Xenova/all-MiniLM-L6-v2`) using Transformers.js. This model runs entirely in your browser using WebAssembly.

2. **Answer Embedding**: 
   - Quiz answers are pre-computed into 384-dimensional embedding vectors.
   - When you submit an answer, your text is also converted to an embedding.

3. **Semantic Comparison**:
   - Your answer embedding is compared to the expected answer using **cosine similarity**.
   - Cosine similarity measures the angle between two vectors - a value closer to 1 means high similarity.
   - This allows "Paris" and "The capital is Paris" to both match correctly!

4. **Scoring System**:
   | Similarity | Rating | Points |
   |------------|--------|--------|
   | ≥ 85% | Excellent! | 1.0 |
   | ≥ 70% | Good! | 0.8 |
   | ≥ 55% | Partial | 0.5 |
   | < 55% | Try Again | 0 |

## Example Quiz Questions

| Question | Example Answers That Score Well |
|----------|----------------------------------|
| Capital of France? | "Paris", "It's Paris", "The capital is Paris" |
| Days in a year? | "365", "There are 365 days", "365 days in a year" |
| How many continents? | "7", "Seven continents", "There are 7 continents" |

## Why Semantic Answer Matching?

Traditional quiz systems require exact text matching, which is frustrating when you know the concept but phrase it differently. By using embeddings:

- ✅ "Paris" and "The capital of France is Paris" both match
- ✅ "365" and "There are 365 days in a year" both score well
- ✅ Answers in your own words get credit if semantically correct
- ✅ No need for predefined answer variants or regex patterns

## License

[MIT](LICENSE)
