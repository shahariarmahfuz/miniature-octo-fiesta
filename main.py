import os
from flask import Flask, request, jsonify
import google.generativeai as genai
import threading
import time
import requests
from collections import deque
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set.")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

chat_sessions = {}  # Dictionary to store chat sessions per user

@app.route('/ask', methods=['GET'])
def ask():
    query = request.args.get('q')
    user_id = request.args.get('id')

    if not query or not user_id:
        logger.warning("Missing query or id parameters.")
        return jsonify({"error": "Please provide both query and id parameters."}), 400

    if user_id not in chat_sessions:
        chat_sessions[user_id] = {
            "chat": model.start_chat(history=[]),
            "history": deque(maxlen=25)  # Stores the last 25 messages
        }

    chat_session = chat_sessions[user_id]["chat"]
    history = chat_sessions[user_id]["history"]

    # Add the user query to history
    history.append(f"User: {query}")
    try:
        response = chat_session.send_message(query)
        # Add the bot response to history
        history.append(f"Bot: {response.text}")
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": "Failed to process the query."}), 500

    return jsonify({"response": response.text})

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "alive"})

def keep_alive():
    url = "https://your-api-url.netlify.app/ping"  # Replace with your actual URL
    while True:
        time.sleep(600)  # Ping every 10 minutes
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logger.info("Ping successful")
            else:
                logger.warning(f"Ping failed with status code {response.status_code}")
        except Exception as e:
            logger.error(f"Ping failed with exception: {e}")

if __name__ == '__main__':
    # Start keep-alive thread
    threading.Thread(target=keep_alive, daemon=True).start()
    app.run(host='0.0.0.0', port=8080)
