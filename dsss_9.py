from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the TinyLlama model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Telegram Bot Token
TOKEN = '7694753045:AAGz4dL0jkGCChqt-jFliOZKwOcR9ZZiKlg'  # Replace with your token

# Helper Function: Generate a response
def generate_response(prompt):
    try:
        # Add a structured prompt for clarity
        structured_prompt = (
            f"The user asked: '{prompt}'\n"
            "Provide a detailed, factual, and clear response without repeating the question."
        )
        inputs = tokenizer(structured_prompt, return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(
            **inputs,
            max_length=300,  # Set a maximum length for responses
            do_sample=True,  # Enable sampling for creative outputs
            temperature=0.7,  # Adjust for balanced randomness
            top_k=50,  # Restrict output to top 50 words
            top_p=0.9,  # Use nucleus sampling
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Fallback for nonsensical or too short responses
        if not response or len(response.split()) < 5:
            response = "I couldn't generate a proper response. Please try rephrasing your question."

    except Exception as e:
        response = "An error occurred while generating a response. Please try again later."
        print(f"Error: {e}")  # Log error for debugging

    return response

# Define the /start command handler
async def start(update: Update, context):
    await update.message.reply_text(
        "Hello! I am your AI assistant. Ask me anything, and I will try my best to provide a helpful answer!"
    )

# Define the message handler to process user input
async def handle_message(update: Update, context):
    user_input = update.message.text.strip()  # Preprocess user input
    response = generate_response(user_input)  # Generate response using helper function
    await update.message.reply_text(response)

# Set up the Telegram bot application
app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Run the bot
if __name__ == "__main__":
    print("Bot is running...")
    app.run_polling()
