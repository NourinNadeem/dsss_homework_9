from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

TOKEN = '7694753045:AAGz4dL0jkGCChqt-jFliOZKwOcR9ZZiKlg'  # Replace with your token

def generate_response(prompt):
    try:

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

        if not response or len(response.split()) < 5:
            response = "I couldn't generate a proper response. Please try rephrasing your question."

    except Exception as e:
        response = "An error occurred while generating a response. Please try again later."
        print(f"Error: {e}")  # Log error for debugging

    return response

async def start(update: Update, context):
    await update.message.reply_text(
        "Hello! I am your AI assistant. Ask me anything, and I will try my best to provide a helpful answer!"
    )

async def handle_message(update: Update, context):
    user_input = update.message.text.strip()
    response = generate_response(user_input)
    await update.message.reply_text(response)

app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

if __name__ == "__main__":
    print("Bot is running...")
    app.run_polling()
