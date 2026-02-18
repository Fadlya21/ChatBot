"""
🤖 AI Chatbot — Generative AI Learning Project

A full-featured AI chatbot with a web UI, built with:
  - Groq API (free) + Llama 3.3 (open-source LLM by Meta)
  - Gradio (web interface framework)

How to run:
  1. pip install -r requirements.txt
  2. Add your Groq API key to .env
  3. python chatbot.py
  4. Open http://localhost:7860
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
import gradio as gr

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

SYSTEM_PROMPT = (
    "You are a friendly and knowledgeable AI assistant. "
    "Provide clear, helpful, and well-structured responses. "
    "Use markdown formatting (bold, lists, code blocks) when appropriate."
)


def chat(message, history, model, temperature, max_tokens):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": message})

    updated_history = history + [{"role": "user", "content": message}]

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=int(max_tokens),
        )

        partial_response = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                partial_response += content
                yield updated_history + [{"role": "assistant", "content": partial_response}]

    except Exception as e:
        yield updated_history + [{"role": "assistant", "content": f"❌ Error: {str(e)}\n\n💡 Check your API key in .env"}]


def export_chat(history):
    if not history:
        return "No chat to export yet!"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_export_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Chat Export — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        for msg in history:
            icon = "👤 You" if msg["role"] == "user" else "🤖 Bot"
            f.write(f"{icon}: {msg['content']}\n\n")
            f.write("-" * 40 + "\n\n")

    return f"✅ Saved to {filename}"


with gr.Blocks(title="🤖 AI Chatbot") as demo:

    gr.Markdown(
        """
        # 🤖 AI Chatbot
        ### A Generative AI Learning Project
        Powered by **Llama 3** via **Groq** — Completely Free!
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(
                placeholder="Type your message here... (Enter to send)",
                container=False,
                autofocus=True,
            )

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            model_dropdown = gr.Dropdown(
                choices=MODELS,
                value="llama-3.3-70b-versatile",
                label="🤖 Model",
                info="Different models have different strengths",
            )
            temperature_slider = gr.Slider(
                minimum=0.0, maximum=2.0, value=0.7, step=0.1,
                label="🌡️ Temperature",
                info="0 = focused, 1 = balanced, 2 = creative",
            )
            max_tokens_slider = gr.Slider(
                minimum=64, maximum=4096, value=1024, step=64,
                label="📏 Max Tokens",
                info="Maximum response length",
            )

            gr.Markdown("### 🛠️ Actions")
            clear_btn = gr.ClearButton([msg, chatbot], value="🗑️ Clear Chat")
            export_btn = gr.Button("💾 Export Chat")
            export_status = gr.Textbox(label="Export Status", interactive=False)

    gr.Examples(
        examples=[
            "Explain machine learning like I'm 5 years old",
            "Write a Python function to reverse a string, with comments",
            "What are the top 5 tips for prompt engineering?",
            "Compare Python and JavaScript — which should I learn?",
        ],
        inputs=msg,
        label="💡 Try these examples",
    )

    msg.submit(
        fn=chat,
        inputs=[msg, chatbot, model_dropdown,
                temperature_slider, max_tokens_slider],
        outputs=chatbot,
    ).then(fn=lambda: "", outputs=msg)

    export_btn.click(fn=export_chat, inputs=chatbot, outputs=export_status)


if __name__ == "__main__":
    print("🤖 AI Chatbot starting...")
    print("Open http://localhost:7860 in your browser!\n")
    demo.launch(share=False)
