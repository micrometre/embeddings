/**
 * Browser-based ML Quiz Embeddings with Vector Search
 * Uses @huggingface/transformers for embeddings
 * Uses a custom FAISS-like index for similarity search (browser compatible)
 */

import { pipeline, env } from '@huggingface/transformers';
import { VectorIndex } from './vectorIndex.js';

// Suppress informational warnings from the library
env.allowLocalModels = false;

// ML Quiz questions to index
const mlQuizzes = [
  "What is the difference between supervised and unsupervised learning?",
  "Explain the concept of overfitting in machine learning models.",
  "What is gradient descent and how does it optimize neural networks?",
  "Describe the bias-variance tradeoff in machine learning.",
  "What are the main differences between classification and regression tasks?",
  "How does a convolutional neural network process images?",
  "What is backpropagation and how does it train neural networks?",
  "Explain the concept of regularization in machine learning.",
  "What are hyperparameters and how do you tune them?",
  "Describe how random forests work for classification."
];

// Interactive Quiz with answers for semantic matching
const quizData = [
  {
    question: "What is the main goal of supervised learning?",
    answer: "Supervised learning aims to learn a mapping from inputs to outputs using labeled training data, where the correct answers are provided during training.",
    hint: "Think about what makes it 'supervised' - what extra information do you have?"
  },
  {
    question: "What problem does regularization solve?",
    answer: "Regularization prevents overfitting by adding a penalty term to the loss function that discourages complex models, helping the model generalize better to unseen data.",
    hint: "Consider what happens when a model is too complex."
  },
  {
    question: "How does a neural network learn from its mistakes?",
    answer: "Neural networks learn through backpropagation, which calculates gradients of the loss with respect to each weight and uses gradient descent to update weights in the direction that minimizes the error.",
    hint: "Think about how errors flow backwards through the network."
  },
  {
    question: "What is the purpose of the activation function in neural networks?",
    answer: "Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns and relationships that cannot be captured by linear transformations alone.",
    hint: "What would happen if all operations were linear?"
  },
  {
    question: "Why do we split data into training and test sets?",
    answer: "We split data to evaluate how well the model generalizes to unseen data. The training set is used to fit the model, while the test set provides an unbiased estimate of model performance.",
    hint: "Consider how you would know if your model actually learned useful patterns."
  }
];

// State
let embedder = null;
let vectorIndex = null;
let answerEmbeddings = []; // Pre-computed answer embeddings for quiz
let currentQuizIndex = 0;
let quizScore = 0;
let quizAttempts = 0;

// DOM Elements
const modelStatus = document.getElementById('model-status');
const modelStatusText = document.getElementById('model-status-text');
const faissStatus = document.getElementById('faiss-status');
const faissStatusText = document.getElementById('faiss-status-text');
const progressFill = document.getElementById('progress-fill');
const searchInput = document.getElementById('search-input');
const searchBtn = document.getElementById('search-btn');
const quizzesList = document.getElementById('quizzes-list');
const resultsList = document.getElementById('results-list');
const logDiv = document.getElementById('log');

// Quiz DOM Elements
const quizQuestion = document.getElementById('quiz-question');
const quizAnswer = document.getElementById('quiz-answer');
const submitAnswerBtn = document.getElementById('submit-answer-btn');
const showHintBtn = document.getElementById('show-hint-btn');
const nextQuestionBtn = document.getElementById('next-question-btn');
const quizFeedback = document.getElementById('quiz-feedback');
const quizProgress = document.getElementById('quiz-progress');
const quizScoreDisplay = document.getElementById('quiz-score');

// Logging utility
function log(message, type = 'info') {
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  logDiv.appendChild(entry);
  logDiv.scrollTop = logDiv.scrollHeight;
  console.log(message);
}

// Update status indicators
function updateStatus(element, textElement, status, text) {
  element.className = `status-icon ${status}`;
  textElement.textContent = text;
}

// Display quiz questions
function displayQuizzes(indexed = false) {
  quizzesList.innerHTML = mlQuizzes.map((quiz, i) => `
    <div class="quiz-item ${indexed ? 'indexed' : ''}">
      ${i + 1}. ${quiz}
    </div>
  `).join('');
}

// Display search results
function displayResults(results) {
  if (results.length === 0) {
    resultsList.innerHTML = '<p style="color: #666;">No results found.</p>';
    return;
  }

  resultsList.innerHTML = results.map(result => `
    <div class="result-item">
      <div class="result-score">Similarity: ${(result.score * 100).toFixed(1)}%</div>
      <div class="result-text">${result.text}</div>
    </div>
  `).join('');
}

// Quiz Functions
function displayQuizQuestion() {
  const quiz = quizData[currentQuizIndex];
  quizQuestion.textContent = quiz.question;
  quizAnswer.value = '';
  quizAnswer.disabled = false;
  submitAnswerBtn.disabled = false;
  showHintBtn.disabled = false;
  nextQuestionBtn.style.display = 'none';
  quizFeedback.innerHTML = '';
  quizFeedback.className = 'quiz-feedback';
  quizProgress.textContent = `Question ${currentQuizIndex + 1} of ${quizData.length}`;
  updateQuizScore();
}

function updateQuizScore() {
  const percentage = quizAttempts > 0 ? Math.round((quizScore / quizAttempts) * 100) : 0;
  quizScoreDisplay.textContent = `Score: ${quizScore}/${quizAttempts} (${percentage}%)`;
}

function getScoreLabel(similarity) {
  if (similarity >= 0.85) return { label: 'Excellent!', class: 'excellent', points: 1 };
  if (similarity >= 0.70) return { label: 'Good!', class: 'good', points: 0.8 };
  if (similarity >= 0.55) return { label: 'Partial', class: 'partial', points: 0.5 };
  return { label: 'Try Again', class: 'incorrect', points: 0 };
}

async function evaluateAnswer() {
  const userAnswer = quizAnswer.value.trim();
  if (!userAnswer) {
    quizFeedback.innerHTML = '<span class="feedback-incorrect">Please enter an answer.</span>';
    return;
  }

  submitAnswerBtn.disabled = true;
  quizAnswer.disabled = true;
  showHintBtn.disabled = true;
  log(`Evaluating answer for question ${currentQuizIndex + 1}...`);

  try {
    // Get embedding for user's answer
    const userEmbedding = await embed(userAnswer);
    
    // Calculate similarity with expected answer
    const expectedEmbedding = answerEmbeddings[currentQuizIndex];
    const similarity = cosineSimilarity(userEmbedding, expectedEmbedding);
    
    const scoreInfo = getScoreLabel(similarity);
    quizAttempts++;
    quizScore += scoreInfo.points;
    
    const expectedAnswer = quizData[currentQuizIndex].answer;
    
    quizFeedback.innerHTML = `
      <div class="feedback-score ${scoreInfo.class}">
        <span class="score-label">${scoreInfo.label}</span>
        <span class="similarity-score">${(similarity * 100).toFixed(1)}% match</span>
      </div>
      <div class="expected-answer">
        <strong>Expected answer:</strong> ${expectedAnswer}
      </div>
    `;
    quizFeedback.className = `quiz-feedback ${scoreInfo.class}`;
    
    log(`Answer evaluated: ${(similarity * 100).toFixed(1)}% similarity - ${scoreInfo.label}`, 
        scoreInfo.points > 0 ? 'success' : 'info');
    
    updateQuizScore();
    nextQuestionBtn.style.display = 'inline-block';
    
  } catch (error) {
    log(`Error evaluating answer: ${error.message}`, 'error');
    quizFeedback.innerHTML = '<span class="feedback-incorrect">Error evaluating answer. Please try again.</span>';
    submitAnswerBtn.disabled = false;
    quizAnswer.disabled = false;
  }
}

function showHint() {
  const hint = quizData[currentQuizIndex].hint;
  quizFeedback.innerHTML = `<div class="hint">💡 Hint: ${hint}</div>`;
  quizFeedback.className = 'quiz-feedback hint-shown';
  showHintBtn.disabled = true;
}

function nextQuestion() {
  currentQuizIndex++;
  if (currentQuizIndex >= quizData.length) {
    // Quiz complete
    const finalPercentage = Math.round((quizScore / quizAttempts) * 100);
    quizQuestion.textContent = '🎉 Quiz Complete!';
    quizAnswer.style.display = 'none';
    submitAnswerBtn.style.display = 'none';
    showHintBtn.style.display = 'none';
    nextQuestionBtn.textContent = 'Restart Quiz';
    nextQuestionBtn.onclick = restartQuiz;
    quizFeedback.innerHTML = `
      <div class="quiz-complete">
        <h3>Final Score: ${quizScore.toFixed(1)}/${quizAttempts}</h3>
        <p>Accuracy: ${finalPercentage}%</p>
        <p>${finalPercentage >= 80 ? '🌟 Excellent work!' : finalPercentage >= 60 ? '👍 Good job!' : '📚 Keep practicing!'}</p>
      </div>
    `;
    quizFeedback.className = 'quiz-feedback complete';
    log(`Quiz completed! Final score: ${quizScore.toFixed(1)}/${quizAttempts} (${finalPercentage}%)`, 'success');
  } else {
    displayQuizQuestion();
  }
}

function restartQuiz() {
  currentQuizIndex = 0;
  quizScore = 0;
  quizAttempts = 0;
  quizAnswer.style.display = 'block';
  submitAnswerBtn.style.display = 'inline-block';
  showHintBtn.style.display = 'inline-block';
  nextQuestionBtn.textContent = 'Next Question';
  nextQuestionBtn.onclick = nextQuestion;
  displayQuizQuestion();
  log('Quiz restarted!', 'info');
}

// Cosine similarity helper (for comparing embeddings)
function cosineSimilarity(a, b) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Pre-compute answer embeddings for quiz
async function initializeQuizEmbeddings() {
  log('Pre-computing quiz answer embeddings...', 'info');
  answerEmbeddings = [];
  
  for (let i = 0; i < quizData.length; i++) {
    const embedding = await embed(quizData[i].answer);
    answerEmbeddings.push(embedding);
    log(`Embedded answer ${i + 1}/${quizData.length}`);
  }
  
  log('Quiz answer embeddings ready!', 'success');
}

// Initialize embedding model
async function initializeModel() {
  log('Loading embedding model...', 'info');
  updateStatus(modelStatus, modelStatusText, 'loading', 'Embedding model: Loading...');

  try {
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      dtype: 'q8',
      progress_callback: (progress) => {
        if (progress.status === 'progress') {
          const percent = Math.round(progress.progress || 0);
          progressFill.style.width = `${percent}%`;
        }
      }
    });

    progressFill.style.width = '100%';
    updateStatus(modelStatus, modelStatusText, 'ready', 'Embedding model: Ready ✓');
    log('Embedding model loaded successfully!', 'success');
    return true;
  } catch (error) {
    updateStatus(modelStatus, modelStatusText, 'error', 'Embedding model: Failed');
    log(`Failed to load model: ${error.message}`, 'error');
    return false;
  }
}

// Generate embedding for text
async function embed(text) {
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

// Initialize FAISS-like vector index and add quiz embeddings
async function initializeVectorIndex() {
  log('Initializing vector index...', 'info');
  updateStatus(faissStatus, faissStatusText, 'loading', 'Vector index: Building...');

  try {
    // Create vector index (384 dimensions for all-MiniLM-L6-v2)
    vectorIndex = new VectorIndex(384);

    // Generate and add embeddings for all quizzes
    for (let i = 0; i < mlQuizzes.length; i++) {
      const quiz = mlQuizzes[i];
      log(`Embedding quiz ${i + 1}/${mlQuizzes.length}...`);
      
      const embedding = await embed(quiz);
      vectorIndex.add(embedding, { id: i, text: quiz });
      
      // Update progress
      const percent = Math.round(((i + 1) / mlQuizzes.length) * 100);
      progressFill.style.width = `${percent}%`;
    }

    // Save to IndexedDB for persistence
    await vectorIndex.save('ml-quizzes-index');

    updateStatus(faissStatus, faissStatusText, 'ready', `Vector index: ${mlQuizzes.length} vectors ✓`);
    log(`Indexed ${mlQuizzes.length} quiz questions!`, 'success');
    displayQuizzes(true);
    return true;
  } catch (error) {
    updateStatus(faissStatus, faissStatusText, 'error', 'Vector index: Failed');
    log(`Failed to build index: ${error.message}`, 'error');
    return false;
  }
}

// Search for similar questions
async function search(query) {
  log(`Searching for: "${query}"`);
  
  const queryEmbedding = await embed(query);
  const results = vectorIndex.search(queryEmbedding, 3); // Top 3 results
  
  log(`Found ${results.length} results`, 'success');
  displayResults(results);
}

// Event handlers
searchBtn.addEventListener('click', async () => {
  const query = searchInput.value.trim();
  if (query) {
    searchBtn.disabled = true;
    await search(query);
    searchBtn.disabled = false;
  }
});

searchInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    searchBtn.click();
  }
});

// Quiz event handlers
submitAnswerBtn.addEventListener('click', evaluateAnswer);
showHintBtn.addEventListener('click', showHint);
nextQuestionBtn.addEventListener('click', nextQuestion);

quizAnswer.addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitAnswerBtn.click();
  }
});

// Initialize app
async function init() {
  log('Starting ML Quiz Vector Search...', 'info');
  displayQuizzes(false);

  // Try to load existing index from IndexedDB
  vectorIndex = new VectorIndex(384);
  const loaded = await vectorIndex.load('ml-quizzes-index');

  if (loaded && vectorIndex.size() > 0) {
    log(`Loaded ${vectorIndex.size()} vectors from cache`, 'success');
    updateStatus(faissStatus, faissStatusText, 'ready', `Vector index: ${vectorIndex.size()} vectors (cached) ✓`);
    displayQuizzes(true);
  }

  // Load embedding model
  const modelReady = await initializeModel();
  if (!modelReady) return;

  // Build index if not loaded from cache
  if (!loaded || vectorIndex.size() === 0) {
    const indexReady = await initializeVectorIndex();
    if (!indexReady) return;
  }

  // Enable search
  searchInput.disabled = false;
  searchBtn.disabled = false;
  searchInput.focus();

  // Initialize quiz
  await initializeQuizEmbeddings();
  quizAnswer.disabled = false;
  submitAnswerBtn.disabled = false;
  showHintBtn.disabled = false;
  displayQuizQuestion();

  log('Ready! Enter a query to search for similar ML questions, or take the quiz below!', 'success');
}

// Start the app
init();
