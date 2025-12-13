import { pipeline, env } from '@huggingface/transformers';

// Skip local model check (always use remote/cached)
env.allowLocalModels = false;

// ============================================
// MODEL CONFIGURATION
// ============================================
const MODELS = {
    summarize: { name: 'Xenova/distilbart-cnn-6-6', task: 'summarization' },
    sentiment: { name: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english', task: 'sentiment-analysis' },
    text2text: { name: 'Xenova/flan-t5-small', task: 'text2text-generation' },
    fillmask: { name: 'Xenova/bert-base-uncased', task: 'fill-mask' },
};

// Store loaded pipelines
let summarizer = null;
let classifier = null;
let generator = null;
let filler = null;

// ============================================
// DOM ELEMENTS
// ============================================

// Summarization
const summarizeInput = document.getElementById('summarizeInput');
const summarizeBtn = document.getElementById('summarizeBtn');
const loadSummarizeBtn = document.getElementById('loadSummarizeBtn');
const summarizeStatus = document.getElementById('summarizeStatus');
const summarizeResult = document.getElementById('summarizeResult');
const summarizeProgressContainer = document.getElementById('summarizeProgressContainer');
const summarizeProgressBar = document.getElementById('summarizeProgressBar');
const summarizeProgressInfo = document.getElementById('summarizeProgressInfo');
const summarizeModelName = document.getElementById('summarizeModelName');
const summarizeModelDetails = document.getElementById('summarizeModelDetails');

// Sentiment
const sentimentInput = document.getElementById('sentimentInput');
const sentimentBtn = document.getElementById('sentimentBtn');
const loadSentimentBtn = document.getElementById('loadSentimentBtn');
const sentimentStatus = document.getElementById('sentimentStatus');
const sentimentResult = document.getElementById('sentimentResult');
const sentimentProgressContainer = document.getElementById('sentimentProgressContainer');
const sentimentProgressBar = document.getElementById('sentimentProgressBar');
const sentimentProgressInfo = document.getElementById('sentimentProgressInfo');
const sentimentModelName = document.getElementById('sentimentModelName');
const sentimentModelDetails = document.getElementById('sentimentModelDetails');

// Text2Text
const text2textInput = document.getElementById('text2textInput');
const text2textBtn = document.getElementById('text2textBtn');
const loadText2textBtn = document.getElementById('loadText2textBtn');
const text2textStatus = document.getElementById('text2textStatus');
const text2textResult = document.getElementById('text2textResult');
const text2textProgressContainer = document.getElementById('text2textProgressContainer');
const text2textProgressBar = document.getElementById('text2textProgressBar');
const text2textProgressInfo = document.getElementById('text2textProgressInfo');
const text2textModelName = document.getElementById('text2textModelName');
const text2textModelDetails = document.getElementById('text2textModelDetails');

// Fill-Mask
const fillmaskInput = document.getElementById('fillmaskInput');
const fillmaskBtn = document.getElementById('fillmaskBtn');
const loadFillmaskBtn = document.getElementById('loadFillmaskBtn');
const fillmaskStatus = document.getElementById('fillmaskStatus');
const fillmaskResult = document.getElementById('fillmaskResult');
const fillmaskProgressContainer = document.getElementById('fillmaskProgressContainer');
const fillmaskProgressBar = document.getElementById('fillmaskProgressBar');
const fillmaskProgressInfo = document.getElementById('fillmaskProgressInfo');
const fillmaskModelName = document.getElementById('fillmaskModelName');
const fillmaskModelDetails = document.getElementById('fillmaskModelDetails');

// Quiz
const quizBtns = document.querySelectorAll('.quiz-btn');
const quizBtn = document.getElementById('quizBtn');
const quizStatus = document.getElementById('quizStatus');
const quizResult = document.getElementById('quizResult');
const quizModelDetails = document.getElementById('quizModelDetails');

let selectedQuestion = null;

// ============================================
// PROGRESS CALLBACK FACTORY
// ============================================
function createProgressCallback(statusEl, progressContainerEl, progressBarEl, progressInfoEl) {
    return (progress) => {
        if (progress.status === 'initiate') {
            progressContainerEl.style.display = 'block';
            statusEl.textContent = `⏳ Downloading: ${progress.file}`;
        } else if (progress.status === 'progress') {
            const percent = Math.round(progress.progress);
            progressBarEl.style.width = `${percent}%`;
            const sizeMB = (progress.loaded / 1024 / 1024).toFixed(1);
            const totalMB = (progress.total / 1024 / 1024).toFixed(1);
            progressInfoEl.textContent = `${progress.file}: ${sizeMB}MB / ${totalMB}MB (${percent}%)`;
        } else if (progress.status === 'done') {
            progressInfoEl.textContent = `✓ ${progress.file} complete`;
        } else if (progress.status === 'ready') {
            progressContainerEl.style.display = 'none';
            progressInfoEl.textContent = '';
        }
    };
}

// ============================================
// INITIALIZE UI (No auto-loading)
// ============================================
summarizeModelName.textContent = MODELS.summarize.name;
summarizeModelDetails.innerHTML = `<span>📋 ${MODELS.summarize.task}</span><span>⚡ fp32</span><span>⏸️ Not loaded</span>`;

sentimentModelName.textContent = MODELS.sentiment.name;
sentimentModelDetails.innerHTML = `<span>📋 ${MODELS.sentiment.task}</span><span>⚡ fp32</span><span>⏸️ Not loaded</span>`;

text2textModelName.textContent = MODELS.text2text.name;
text2textModelDetails.innerHTML = `<span>📋 ${MODELS.text2text.task}</span><span>⚡ fp32</span><span>⏸️ Not loaded</span>`;

fillmaskModelName.textContent = MODELS.fillmask.name;
fillmaskModelDetails.innerHTML = `<span>📋 ${MODELS.fillmask.task}</span><span>⚡ fp32</span><span>⏸️ Not loaded</span>`;

// ============================================
// LOAD MODEL HANDLERS
// ============================================

// Load Summarization Model
loadSummarizeBtn.addEventListener('click', async () => {
    loadSummarizeBtn.disabled = true;
    loadSummarizeBtn.textContent = '⏳ Loading...';
    summarizeStatus.textContent = '⏳ Loading model...';
    summarizeModelDetails.innerHTML = `<span>📋 ${MODELS.summarize.task}</span><span>⚡ fp32</span><span>⏳ Loading...</span>`;

    summarizer = await pipeline(MODELS.summarize.task, MODELS.summarize.name, {
        dtype: 'fp32',
        progress_callback: createProgressCallback(
            summarizeStatus, summarizeProgressContainer, summarizeProgressBar, summarizeProgressInfo
        ),
    });

    summarizeProgressContainer.style.display = 'none';
    summarizeProgressInfo.textContent = '';
    summarizeStatus.textContent = '✅ Ready!';
    summarizeModelDetails.innerHTML = `<span>📋 ${MODELS.summarize.task}</span><span>⚡ fp32</span><span>✅ Ready</span>`;
    loadSummarizeBtn.style.display = 'none';
    summarizeBtn.disabled = false;
});

// Load Sentiment Model
loadSentimentBtn.addEventListener('click', async () => {
    loadSentimentBtn.disabled = true;
    loadSentimentBtn.textContent = '⏳ Loading...';
    sentimentStatus.textContent = '⏳ Loading model...';
    sentimentModelDetails.innerHTML = `<span>📋 ${MODELS.sentiment.task}</span><span>⚡ fp32</span><span>⏳ Loading...</span>`;

    classifier = await pipeline(MODELS.sentiment.task, MODELS.sentiment.name, {
        dtype: 'fp32',
        progress_callback: createProgressCallback(
            sentimentStatus, sentimentProgressContainer, sentimentProgressBar, sentimentProgressInfo
        ),
    });

    sentimentProgressContainer.style.display = 'none';
    sentimentProgressInfo.textContent = '';
    sentimentStatus.textContent = '✅ Ready!';
    sentimentModelDetails.innerHTML = `<span>📋 ${MODELS.sentiment.task}</span><span>⚡ fp32</span><span>✅ Ready</span>`;
    loadSentimentBtn.style.display = 'none';
    sentimentBtn.disabled = false;
});

// Load Text2Text Model
loadText2textBtn.addEventListener('click', async () => {
    loadText2textBtn.disabled = true;
    loadText2textBtn.textContent = '⏳ Loading...';
    text2textStatus.textContent = '⏳ Loading model...';
    text2textModelDetails.innerHTML = `<span>📋 ${MODELS.text2text.task}</span><span>⚡ fp32</span><span>⏳ Loading...</span>`;

    generator = await pipeline(MODELS.text2text.task, MODELS.text2text.name, {
        dtype: 'fp32',
        progress_callback: createProgressCallback(
            text2textStatus, text2textProgressContainer, text2textProgressBar, text2textProgressInfo
        ),
    });

    text2textProgressContainer.style.display = 'none';
    text2textProgressInfo.textContent = '';
    text2textStatus.textContent = '✅ Ready!';
    text2textModelDetails.innerHTML = `<span>📋 ${MODELS.text2text.task}</span><span>⚡ fp32</span><span>✅ Ready</span>`;
    loadText2textBtn.style.display = 'none';
    text2textBtn.disabled = false;

    // Also enable quiz since it uses text2text model
    quizModelDetails.innerHTML = `<span>🧠 Q&A</span><span>✅ Ready</span>`;
    quizBtns.forEach(btn => btn.disabled = false);
});

// Load Fill-Mask Model
loadFillmaskBtn.addEventListener('click', async () => {
    loadFillmaskBtn.disabled = true;
    loadFillmaskBtn.textContent = '⏳ Loading...';
    fillmaskStatus.textContent = '⏳ Loading model...';
    fillmaskModelDetails.innerHTML = `<span>📋 ${MODELS.fillmask.task}</span><span>⚡ fp32</span><span>⏳ Loading...</span>`;

    filler = await pipeline(MODELS.fillmask.task, MODELS.fillmask.name, {
        dtype: 'fp32',
        progress_callback: createProgressCallback(
            fillmaskStatus, fillmaskProgressContainer, fillmaskProgressBar, fillmaskProgressInfo
        ),
    });

    fillmaskProgressContainer.style.display = 'none';
    fillmaskProgressInfo.textContent = '';
    fillmaskStatus.textContent = '✅ Ready!';
    fillmaskModelDetails.innerHTML = `<span>📋 ${MODELS.fillmask.task}</span><span>⚡ fp32</span><span>✅ Ready</span>`;
    loadFillmaskBtn.style.display = 'none';
    fillmaskBtn.disabled = false;
});

// ============================================
// UK HISTORY QUIZ
// ============================================

// Handle question selection
quizBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        if (!generator) {
            quizResult.innerHTML = '<span style="color: #dc3545;">Please load the Text2Text model first!</span>';
            return;
        }
        quizBtns.forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        selectedQuestion = btn.dataset.question;
        quizBtn.disabled = false;
        quizResult.innerHTML = '';
    });
});

// ============================================
// ACTION HANDLERS
// ============================================

// Text Summarization
summarizeBtn.addEventListener('click', async () => {
    const text = summarizeInput.value.trim();
    if (!text) return;
    if (!summarizer) return;

    summarizeBtn.disabled = true;
    summarizeStatus.textContent = '📝 Summarizing...';

    const output = await summarizer(text, {
        max_new_tokens: 100,
        min_length: 20,
    });

    summarizeResult.innerHTML = `<strong>${output[0].summary_text}</strong>`;
    summarizeStatus.textContent = '';
    summarizeBtn.disabled = false;
});

// Sentiment Analysis
sentimentBtn.addEventListener('click', async () => {
    const text = sentimentInput.value.trim();
    if (!text) return;
    if (!classifier) return;

    sentimentBtn.disabled = true;
    sentimentStatus.textContent = '🔍 Analyzing...';

    const output = await classifier(text);
    const { label, score } = output[0];

    sentimentResult.innerHTML = `
        <strong class="${label.toLowerCase()}">${label}</strong><br>
        Confidence: ${(score * 100).toFixed(1)}%
    `;
    sentimentStatus.textContent = '';
    sentimentBtn.disabled = false;
});

// Text2Text Generation
text2textBtn.addEventListener('click', async () => {
    const text = text2textInput.value.trim();
    if (!text) return;
    if (!generator) return;

    text2textBtn.disabled = true;
    text2textStatus.textContent = '✨ Generating...';

    const output = await generator(text, {
        max_new_tokens: 100,
    });

    text2textResult.innerHTML = `<strong>${output[0].generated_text}</strong>`;
    text2textStatus.textContent = '';
    text2textBtn.disabled = false;
});

// UK History Quiz
quizBtn.addEventListener('click', async () => {
    if (!selectedQuestion) return;
    if (!generator) {
        quizResult.innerHTML = '<span style="color: #dc3545;">Please load the Text2Text model first!</span>';
        return;
    }

    quizBtn.disabled = true;
    quizBtns.forEach(btn => btn.disabled = true);
    quizStatus.textContent = '🤔 Thinking...';

    const prompt = `Answer this UK history question: ${selectedQuestion}`;
    const output = await generator(prompt, {
        max_new_tokens: 50,
    });

    quizResult.innerHTML = `
        <strong>Q:</strong> ${selectedQuestion}<br><br>
        <strong>A:</strong> ${output[0].generated_text}
    `;
    quizStatus.textContent = '';
    quizBtn.disabled = false;
    quizBtns.forEach(btn => btn.disabled = false);
});

// Fill in the Blank
fillmaskBtn.addEventListener('click', async () => {
    const text = fillmaskInput.value.trim();
    if (!text || !text.includes('[MASK]')) {
        fillmaskResult.innerHTML = '<span style="color: #dc3545;">Please include [MASK] in your text!</span>';
        return;
    }
    if (!filler) return;

    fillmaskBtn.disabled = true;
    fillmaskStatus.textContent = '🔤 Filling in...';

    const output = await filler(text);
    
    // Show top 3 predictions
    const topResults = output.slice(0, 3).map((item, i) => 
        `${i + 1}. <strong>${item.token_str}</strong> <span style="color: #666;">(${(item.score * 100).toFixed(1)}%)</span>`
    ).join('<br>');

    fillmaskResult.innerHTML = topResults;
    fillmaskStatus.textContent = '';
    fillmaskBtn.disabled = false;
});