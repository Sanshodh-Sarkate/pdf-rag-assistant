import re

# Sets of conversational phrases
GREETINGS = {
    "hi", "hello", "hey", "heya", "hola", "namaste", "greetings", "howdy",
    "good morning", "good afternoon", "good evening", "good day",
    "hi there", "hello there", "hey there", "hello assistant", "hi bot"
}

CHIT_CHAT = {
    "how are you", "how are you doing", "hows it going", "how is it going",
    "whats up", "what's up"
}

GRATITUDES = {
    "thank you", "thanks", "thanks a lot", "thank you so much", "thx",
    "many thanks", "thank you very much"
}

FAREWELLS = {
    "bye", "goodbye", "see you", "cya", "have a nice day", "have a good day",
    "good night", "take care"
}

BOT_INFO = {
    "who are you", "what are you", "what can you do", "help", "what is your name"
}

# handle the  greetings reponse 
def get_conversational_response(user_input: str) -> str | None:
    """
    Check if the user input is a pure greeting, gratitude, farewell, or bot question.
    Returns a friendly predefined response if matched, otherwise None (allowing standard RAG).
    
    If the message combines a greeting with a real question (e.g., 'Hello, what is the policy?'),
    this returns None so the full question can be answered from the document context.
    """
    if not user_input or not isinstance(user_input, str):
        return None

    # lowercase, remove punctuation, collapse whitespace
    cleaned = re.sub(r"[^\w\s]", "", user_input.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    # 1. Direct match with greetings
    if cleaned in GREETINGS:
        return "Hello! How can I assist you with your PDF documents today?"

    # 2. General chit-chat how are you
    if cleaned in CHIT_CHAT:
        return "I'm doing well, thank you! How can I assist you with your PDF documents today?"

    # 3. Gratitude
    if cleaned in GRATITUDES:
        return "You're very welcome! Let me know if you need anything else from the documents."

    # 4. Farewells
    if cleaned in FAREWELLS:
        return "Goodbye! Have a great day and feel free to reach out if you have more questions."

    # 5. Bot inquiries  Help
    if cleaned in BOT_INFO:
        return (
            "I am your PDF RAG Assistant. Ask me anything about the uploaded PDF documents, "
            "policies, or procedures, and I will find the answers directly from the text."
        )

    return None
