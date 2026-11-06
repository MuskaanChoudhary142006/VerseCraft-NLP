"""
Interactive Command Line Interface for Paragraph-to-Poem NLP
============================================================
Run directly in terminal:
    python run_cli.py
"""

import sys
from poem_generator import PoemGenerator, STYLE_TEMPLATES, TONE_PROMPTS

def main():
    print("=" * 65)
    print("      📜 PARAGRAPH-TO-POEM NLP GENERATOR (CLI MODE)        ")
    print("=" * 65)
    print()

    generator = PoemGenerator(model_name="google/flan-t5-small")

    # Prompt user for input paragraph
    print("Enter or paste your paragraph (press Enter twice to finish):")
    lines = []
    while True:
        try:
            line = input()
            if line == "":
                if lines:
                    break
            else:
                lines.append(line)
        except EOFError:
            break

    paragraph = " ".join(lines).strip()
    if not paragraph:
        print("No text entered. Using default ocean sunset paragraph.\n")
        paragraph = (
            "The golden sun slowly dipped beneath the calm ocean horizon, casting long ripples "
            "of amber and violet across the waters. As the evening breeze whispered through the coastal "
            "palms, solitary seagulls glided back to their nests, and the tide gently embraced the silent sand."
        )

    # Style Selection
    styles = list(STYLE_TEMPLATES.keys())
    print("\nSelect Poetic Style:")
    for idx, s in enumerate(styles, 1):
        print(f"  [{idx}] {s}")
    choice = input(f"Enter choice (1-{len(styles)}, default=1): ").strip()
    selected_style = styles[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(styles) else styles[0]

    # Tone Selection
    tones = list(TONE_PROMPTS.keys())
    print("\nSelect Mood / Tone:")
    for idx, t in enumerate(tones, 1):
        print(f"  [{idx}] {t}")
    choice = input(f"Enter choice (1-{len(tones)}, default=1): ").strip()
    selected_tone = tones[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(tones) else tones[0]

    # Engine Selection
    engines = ["Auto (Smart Fallback)", "Deep Learning (Flan-T5)", "Fast Offline NLP"]
    print("\nSelect Engine Mode:")
    for idx, e in enumerate(engines, 1):
        print(f"  [{idx}] {e}")
    choice = input(f"Enter choice (1-{len(engines)}, default=1): ").strip()
    selected_engine = engines[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(engines) else engines[0]

    print("\n✨ Generating your poem...\n")
    result = generator.generate(
        paragraph=paragraph,
        style=selected_style,
        tone=selected_tone,
        engine_mode=selected_engine
    )

    print("-" * 65)
    print(f"Engine: {result['engine']} | Style: {result['style']} | Tone: {result['tone']}")
    print("-" * 65)
    print()
    print(result["poem"])
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
