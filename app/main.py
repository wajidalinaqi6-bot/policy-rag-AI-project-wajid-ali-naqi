"""
main.py – Flask entry-point for the Policy RAG application.

FIX 1: load_dotenv() is now called at the very top, before any other
        import that reads os.getenv().  Without this call, all API-key
        env vars silently evaluate to '' and the RAG pipeline raises:
            ValueError: No LLM provider configured.

FIX 2: sys.path manipulation replaced with a proper package-relative
        import so the app can be started from any working directory:
            python app/main.py          (from project root)
            cd app && python main.py    (from app/)
        Both work because we add the *project root* (parent of app/) to
        sys.path rather than the directory of the current file.
"""

import os
import sys
import time

# ── FIX 1: load .env before anything that reads os.getenv() ───────────────
from dotenv import load_dotenv
load_dotenv()  # reads <project_root>/.env into os.environ

# ── FIX 2: stable sys.path so 'import config' always resolves ─────────────
#    __file__ is  .../project_final/app/main.py
#    We want      .../project_final/app  on the path (where config.py lives)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from vector_store import get_vector_store
from document_loader import load_documents
from text_splitter import chunk_documents
from rag_pipeline import create_rag_pipeline

import config

app = Flask(__name__)
CORS(app)

vector_store = None
rag_pipeline = None
indexed = False


# ── HTML chat UI ──────────────────────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Company Policy Q&amp;A</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: white; text-align: center; margin-bottom: 10px; font-size: 2.5em; }
        .subtitle { color: rgba(255,255,255,0.8); text-align: center; margin-bottom: 30px; }
        .chat-container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        #chat-messages {
            height: 500px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 10px;
            max-width: 85%;
        }
        .user-message { background: #667eea; color: white; margin-left: auto; }
        .bot-message { background: white; color: #333; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .citations {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            font-size: 0.85em;
            color: #666;
        }
        .citation-item {
            background: #f0f0f0;
            padding: 8px 12px;
            border-radius: 5px;
            margin-top: 5px;
        }
        .input-container {
            display: flex;
            padding: 20px;
            background: white;
            border-top: 1px solid #eee;
        }
        #user-input {
            flex: 1;
            padding: 12px 20px;
            border: 2px solid #ddd;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        #user-input:focus { border-color: #667eea; }
        #send-btn {
            margin-left: 10px;
            padding: 12px 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
        }
        #send-btn:hover { background: #5568d3; }
        #send-btn:disabled { background: #ccc; cursor: not-allowed; }
        .loading { display: inline-block; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
</head>
<body>
    <div class="container">
        <h1>Company Policy Q&amp;A</h1>
        <p class="subtitle">Ask questions about our company policies and procedures</p>
        <div class="chat-container">
            <div id="chat-messages"></div>
            <div class="input-container">
                <input type="text" id="user-input" placeholder="Type your question here..." autofocus>
                <button id="send-btn">Send</button>
            </div>
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chat-messages');
        const userInput    = document.getElementById('user-input');
        const sendBtn      = document.getElementById('send-btn');

        function addMessage(content, isUser, citations = []) {
            const div = document.createElement('div');
            div.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            let html = content;
            if (!isUser && citations.length) {
                html += '<div class="citations"><strong>Sources:</strong>';
                citations.forEach(c => {
                    html += `<div class="citation-item"><strong>${c.source}:</strong> ${c.content}</div>`;
                });
                html += '</div>';
            }
            div.innerHTML = html;
            chatMessages.appendChild(div);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        async function sendMessage() {
            const question = userInput.value.trim();
            if (!question) return;

            userInput.value = '';
            sendBtn.disabled = true;
            addMessage(question, true);
            addMessage('<span class="loading">Thinking…</span>', false);

            try {
                const res  = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                const data = await res.json();
                chatMessages.removeChild(chatMessages.lastChild);
                if (data.success) {
                    addMessage(data.answer, false, data.citations);
                } else {
                    addMessage('Sorry, an error occurred: ' + (data.error || 'Unknown error'), false);
                }
            } catch {
                chatMessages.removeChild(chatMessages.lastChild);
                addMessage('Sorry, an error occurred. Please try again.', false);
            }

            sendBtn.disabled = false;
            userInput.focus();
        }

        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', e => { if (e.key === 'Enter') sendMessage(); });
    </script>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the web chat interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/chat', methods=['POST'])
def chat():
    import traceback
    try:
        data = request.json
        question = data.get('question', '')
        print(f"📝 Question: {question}")
        
        result = rag_pipeline.answer(question)
        print(f"📤 Result: {result}")
        
        return jsonify(result)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return jsonify({'answer': f'Error: {str(e)}', 'success': False})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status':              'healthy',
        'indexed':             indexed,
        'vector_store_ready':  vector_store is not None
    })


@app.route('/reindex', methods=['POST'])
def reindex():
    """Re-index the document collection.

    Safe to call at any time — uses upsert under the hood so existing
    documents are overwritten rather than duplicated.
    """
    global indexed
    try:
        documents = load_documents()
        chunks    = chunk_documents(documents)
        vector_store.add_documents(chunks)
        indexed   = True
        return jsonify({
            'success': True,
            'message': f'Successfully indexed {len(chunks)} document chunks'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Startup ───────────────────────────────────────────────────────────────

def initialize_app():
    """Initialize vector store and RAG pipeline on startup."""
    global vector_store, rag_pipeline, indexed

    print("Initializing Policy Q&A Application…")

    vector_store = get_vector_store()

    try:
        docs = vector_store.get_all_documents()
        if docs:
            print(f"Loaded {len(docs)} documents from existing vector store")
            indexed = True
        else:
            raise ValueError("Empty vector store — building index")
    except Exception as e:
        print(f"{e}  →  indexing documents now…")
        documents = load_documents()
        chunks    = chunk_documents(documents)
        vector_store.add_documents(chunks)
        indexed   = True
        print(f"Indexed {len(chunks)} document chunks")

    rag_pipeline = create_rag_pipeline(vector_store)
    print("RAG pipeline initialized — ready.")


initialize_app()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
