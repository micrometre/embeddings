/**
 * Semantic Quiz - AI-Powered Answer Grading
 * Uses Transformers.js for browser-based embeddings
 * Grades answers using cosine similarity for semantic matching
 */

import { pipeline, env } from '@huggingface/transformers';

// Suppress informational warnings from the library
env.allowLocalModels = false;

// Quiz questions with expected answers for semantic matching
const quizData = [
  {
    question: "What is the capital city of France?",
    answer: "Paris is the capital city of France.",
    hint: "Paris is the capital city of France or The capital of France is Paris."
  },
  {
    question: "How many days are there in a year?",
    answer: "There are 365 days in a year.",
    hint: "There are 365 days in a year or In a year there are 365 days."
  },
  {
    question: "How many continents are there on Earth?",
    answer: "There are seven continents on Earth.",
    hint: "There are seven continents on Earth or Earth has seven continents."
  },
];

// State
let embedder = null;
let answerEmbeddings = [];
let currentQuizIndex = 0;
let quizScore = 0;
let quizAttempts = 0;

// DOM Elements
const modelStatus = document.getElementById('model-status');
const modelStatusText = document.getElementById('model-status-text');
const progressFill = document.getElementById('progress-fill');

// Quiz DOM Elements
const quizQuestion = document.getElementById('quiz-question');
const quizAnswer = document.getElementById('quiz-answer');
const submitAnswerBtn = document.getElementById('submit-answer-btn');
const showHintBtn = document.getElementById('show-hint-btn');
const nextQuestionBtn = document.getElementById('next-question-btn');
const quizFeedback = document.getElementById('quiz-feedback');
const quizProgress = document.getElementById('quiz-progress');
const quizScoreDisplay = document.getElementById('quiz-score');

// Update status indicators
function updateStatus(status, text) {
  modelStatus.className = `status-icon ${status}`;
  modelStatusText.textContent = text;
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

  try {
    const userEmbedding = await embed(userAnswer);
    const expectedEmbedding = answerEmbeddings[currentQuizIndex];
    const similarity = cosineSimilarity(userEmbedding, expectedEmbedding);
    console.log(similarity);
    
    const scoreInfo = getScoreLabel(similarity);
    quizAttempts++;
    quizScore += scoreInfo.points;
    
    const expectedAnswer = quizData[currentQuizIndex].answer;
    
    const tryAgainButton = scoreInfo.points === 0 
      ? `<button class="btn-retry" onclick="window.retryAnswer()">🔄 Try Again</button>` 
      : '';
    
    quizFeedback.innerHTML = `
      <div class="feedback-score ${scoreInfo.class}">
        <span class="score-label">${scoreInfo.label}</span>
        <span class="similarity-score">${(similarity * 100).toFixed(1)}% match</span>
        ${tryAgainButton}
      </div>
      <div class="expected-answer">
        <strong>Expected answer:</strong> ${expectedAnswer}
      </div>
    `;
    quizFeedback.className = `quiz-feedback ${scoreInfo.class}`;
    
    updateQuizScore();
    nextQuestionBtn.style.display = 'inline-block';
    
  } catch (error) {
    console.error('Error evaluating answer:', error);
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
    const finalPercentage = Math.round((quizScore / quizAttempts) * 100);
    quizQuestion.textContent = '🎉 Quiz Complete!';
    quizAnswer.style.display = 'none';
    submitAnswerBtn.style.display = 'none';
    showHintBtn.style.display = 'none';
    nextQuestionBtn.textContent = 'Restart Quiz';
    nextQuestionBtn.style.display = 'inline-block';
    quizFeedback.innerHTML = `
      <div class="quiz-complete">
        <h3>Final Score: ${quizScore.toFixed(1)}/${quizAttempts}</h3>
        <p>Accuracy: ${finalPercentage}%</p>
        <p>${finalPercentage >= 80 ? '🌟 Excellent work!' : finalPercentage >= 60 ? '👍 Good job!' : '📚 Keep practicing!'}</p>
      </div>
    `;
    quizFeedback.className = 'quiz-feedback complete';
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
  nextQuestionBtn.style.display = 'none';
  displayQuizQuestion();
}

function retryAnswer() {
  quizAttempts--;
  quizAnswer.disabled = false;
  submitAnswerBtn.disabled = false;
  showHintBtn.disabled = false;
  nextQuestionBtn.style.display = 'none';
  quizFeedback.innerHTML = '<div class="hint">💪 Try again! Think about the key concepts.</div>';
  quizFeedback.className = 'quiz-feedback hint-shown';
  quizAnswer.focus();
  updateQuizScore();
}

window.retryAnswer = retryAnswer;

async function embed(text) {
  const result = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(result.data);
}

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

async function init() {
  updateStatus('loading', 'Loading AI model...');
  
  try {
    embedder = await pipeline(
      'feature-extraction',
      'Xenova/all-MiniLM-L6-v2',
      { 
        dtype: 'q8',
        progress_callback: (progress) => {
          if (progress.status === 'progress') {
            const percent = Math.round(progress.progress);
            progressFill.style.width = `${percent}%`;
            updateStatus('loading', `Loading model... ${percent}%`);
          }
        }
      }
    );
    
    progressFill.style.width = '100%';
    updateStatus('ready', 'AI model ready!');
    
    console.log('Pre-computing answer embeddings...');
    for (const quiz of quizData) {
      const embedding = await embed(quiz.answer);
      answerEmbeddings.push(embedding);
    }
    console.log('Answer embeddings ready!');
    
    displayQuizQuestion();
    
  } catch (error) {
    console.error('Initialization error:', error);
    updateStatus('error', `Error: ${error.message}`);
  }
}

// Event Listeners
submitAnswerBtn.addEventListener('click', evaluateAnswer);
showHintBtn.addEventListener('click', showHint);
nextQuestionBtn.addEventListener('click', () => {
  if (currentQuizIndex >= quizData.length - 1 && nextQuestionBtn.textContent === 'Restart Quiz') {
    restartQuiz();
  } else {
    nextQuestion();
  }
});

quizAnswer.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !submitAnswerBtn.disabled) {
    e.preventDefault();
    evaluateAnswer();
  }
});

init();
