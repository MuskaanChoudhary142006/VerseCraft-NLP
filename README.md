# 📜 Paragraph-to-Poem NLP Generator

An intelligent Natural Language Processing (NLP) system that transforms narrative paragraphs and prose into structured, evocative poetry.

---

## 🌟 Key Features

- **Dual-Engine Architecture**:
  1. **Deep Learning Seq2Seq**: Uses Hugging Face Transformer models (`google/flan-t5-small` or `flan-t5-base`) for contextual, semantic poem generation.
  2. **Fast Offline NLP Synthesizer**: Uses lexical parsing, POS keyword extraction, cadence matching, and rhyming banks for instant generation with zero downloads or internet needed.
- **5 Poetic Formats**:
  - `Rhyming Stanzas` (classic 4-line AABB/ABAB stanzas)
  - `Free Verse` (modern lyrical, metaphorical cadence)
  - `Haiku Series` (3-stanza 5-7-5 syllabic sequences)
  - `Shakespearean Sonnet` (14-line structure with closing rhyming couplet)
  - `Ballad` (narrative folk meter)
- **5 Tone & Mood Controls**:
  - *Mystical & Nature*, *Melancholic & Reflective*, *Joyful & Uplifting*, *Romantic & Passionate*, *Epic & Dramatic*
- **Two Ways to Run**:
  - 🌐 **Modern Web UI**: Built with Gradio (sliders, presets, copy to clipboard, `.txt` poem export).
  - 💻 **Interactive CLI**: Fast terminal interface.

---

## 🚀 How to Run in Visual Studio Code (Step-by-Step)

### Step 1: Open the Project in VS Code
1. Open **Visual Studio Code**.
2. Click **File** -> **Open Folder...** (or press `Ctrl+K Ctrl+O`).
3. Select the `para-to-poem-nlp` folder.

### Step 2: Select Your Python Interpreter
1. Press `Ctrl + Shift + P` to open the Command Palette.
2. Type and select: `Python: Select Interpreter`.
3. Choose your Python 3.10+ or 3.11 environment.

### Step 3: Install Dependencies (If not already installed)
Open the integrated terminal in VS Code (`Ctrl + ~` or `Terminal -> New Terminal`) and run:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application (Choose Option A or B)

#### Option A: One-Click Run (Recommended)
1. In the VS Code left sidebar, click the **Run and Debug** icon (`Ctrl + Shift + D`).
2. At the top dropdown, select:
   - **`▶ Run Web App (Gradio Studio)`**
3. Press **`F5`** or click the green **Play** button!
4. The terminal will show:
   ```text
   Running on local URL:  http://127.0.0.1:7860
   ```
5. Click or open `http://127.0.0.1:7860` in your web browser.

#### Option B: Run via Integrated Terminal
- **For Web UI:**
  ```bash
  python app.py
  ```
- **For Interactive Terminal CLI:**
  ```bash
  python run_cli.py
  ```

---

## 📁 Project Structure

```text
para-to-poem-nlp/
├── app.py                  # Gradio Web Application UI
├── poem_generator.py       # Core NLP Model & Dual-Engine Generation Logic
├── run_cli.py              # Interactive Terminal CLI interface
├── sample_paragraphs.py    # Curated preset test paragraphs
├── requirements.txt        # Python package dependencies
├── README.md               # Detailed documentation & VS Code guide
└── .vscode/
    ├── launch.json         # 1-Click VSCode Run & Debug profiles
    └── settings.json       # VSCode Python workspace configuration
```

---

## 💡 Example Transformation

### Input Paragraph:
> *"The golden sun slowly dipped beneath the calm ocean horizon, casting long ripples of amber and violet across the waters. As the evening breeze whispered through the coastal palms, solitary seagulls glided back to their nests, and the tide gently embraced the silent sand."*

### Generated Poem (Rhyming Stanzas):
> *When twilight covers the serene expanse of deep,*  
> *The whisper of the sun begins to gently wind,*  
> *A silver rhythm dances in the velvet sleep,*  
> *While silent wonders through the shadows softly unpinned.*  
>  
> *Across the valley where the emerald rivers rain,*  
> *The echoes of calm awaken like a distant glow,*  
> *In every twilight contour where the currents pain,*  
> *The heart finds solace and a melody that does flow.*  
>  
> *Now let the sorrow and the weary world take dipped,*  
> *For hope is woven in the fabric of the ancient night,*  
> *As morning gently turns the darkness into flight,*  
> *And gives the spirit back its everlasting light.*  
