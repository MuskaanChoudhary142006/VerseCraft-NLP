"""
Paragraph-to-Poem NLP Web Application
Powered by Hugging Face Transformers & Gradio
"""

import time
import tempfile
import os
import gradio as gr
from poem_generator import PoemGenerator, STYLE_TEMPLATES, TONE_PROMPTS
from sample_paragraphs import SAMPLES

# Initialize core model engine
generator = PoemGenerator(model_name="google/flan-t5-small")

def convert_paragraph_to_poem(paragraph, style, tone, engine_mode, temperature):
    """Callback function to generate poem and compute metadata."""
    start_time = time.time()
    
    if not paragraph or not paragraph.strip():
        return (
            "⚠️ Please enter or paste a paragraph above to generate a poem.",
            "No input provided",
            None
        )

    result = generator.generate(
        paragraph=paragraph,
        style=style,
        tone=tone,
        engine_mode=engine_mode,
        temperature=temperature
    )
    
    elapsed = round(time.time() - start_time, 2)
    poem_text = result["poem"]
    engine_name = result["engine"]
    
    status_info = (
        f"**Engine Used:** `{engine_name}` | "
        f"**Style:** `{style}` | "
        f"**Tone:** `{tone}` | "
        f"**Time Taken:** `{elapsed}s`"
    )

    # Save to a temporary text file for one-click download
    tmp_path = None
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, "generated_poem.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"=== PARAGRAPH-TO-POEM NLP GENERATOR ===\n")
            f.write(f"Style: {style} | Tone: {tone}\n")
            f.write(f"Engine: {engine_name}\n\n")
            f.write("--- ORIGINAL PARAGRAPH ---\n")
            f.write(paragraph.strip() + "\n\n")
            f.write("--- GENERATED POEM ---\n")
            f.write(poem_text + "\n")
        tmp_path = file_path
    except Exception:
        tmp_path = None

    return poem_text, status_info, tmp_path


# Define Gradio Interface
custom_css = """
#poem-output textarea {
    font-family: 'Georgia', 'Palatino', serif;
    font-size: 1.15rem;
    line-height: 1.7;
    background-color: #fdfbf7;
    color: #2c2523;
}
"""

with gr.Blocks(title="Paragraph to Poem NLP Generator") as demo:
    gr.Markdown(
        """
        # 📜 Paragraph to Poem NLP Generator
        ### Transform ordinary prose into captivating, rhythmic poetry using Natural Language Processing & Transformers
        """
    )
    
    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("#### ✍️ Input Paragraph")
            input_paragraph = gr.Textbox(
                lines=8,
                placeholder="Paste or write your paragraph here (e.g. describing a landscape, an emotional memory, a rainy city afternoon, or a journey)...",
                label="Your Paragraph",
                value=SAMPLES[0]["text"]
            )
            
            with gr.Row():
                style_dropdown = gr.Dropdown(
                    choices=list(STYLE_TEMPLATES.keys()),
                    value="Rhyming Stanzas",
                    label="Poetic Style",
                    info="Choose the poetic meter and stanza architecture"
                )
                tone_dropdown = gr.Dropdown(
                    choices=list(TONE_PROMPTS.keys()),
                    value="Mystical & Nature",
                    label="Poetic Mood / Tone",
                    info="Set the emotional coloring of the poem"
                )

            with gr.Row():
                engine_dropdown = gr.Dropdown(
                    choices=[
                        "Auto (Smart Fallback)",
                        "Deep Learning (Flan-T5)",
                        "Fast Offline NLP"
                    ],
                    value="Auto (Smart Fallback)",
                    label="NLP Engine Mode",
                    info="Deep Learning runs HuggingFace Seq2Seq; Fast Offline NLP generates instantly"
                )
                temp_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    label="Creativity / Temperature",
                    info="Higher values produce more creative & varied imagery"
                )

            generate_btn = gr.Button("✨ Transform into Poem", variant="primary", size="lg")

            # Presets Section
            gr.Markdown("#### 💡 Quick Test Samples")
            with gr.Row():
                for sample in SAMPLES:
                    gr.Button(f"📌 {sample['title']}").click(
                        fn=lambda s=sample: (s["text"], s["style"], s["tone"]),
                        inputs=[],
                        outputs=[input_paragraph, style_dropdown, tone_dropdown]
                    )

        with gr.Column(scale=5):
            gr.Markdown("#### 🎭 Generated Poem")
            output_poem = gr.Textbox(
                lines=12,
                label="Poetic Transformation",
                elem_id="poem-output",
                interactive=False
            )
            status_markdown = gr.Markdown("Ready to generate. Click **Transform into Poem** to start.")
            download_file = gr.File(label="💾 Download Poem (.txt)")

    # Connect button event
    generate_btn.click(
        fn=convert_paragraph_to_poem,
        inputs=[input_paragraph, style_dropdown, tone_dropdown, engine_dropdown, temp_slider],
        outputs=[output_poem, status_markdown, download_file]
    )


if __name__ == "__main__":
    print("Launching Paragraph-to-Poem NLP Studio...")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, css=custom_css)
