"""
NLP Paragraph-to-Poem Generation Engine
=======================================
Dual-engine architecture:
1. Deep Learning Transformer Engine (Hugging Face Seq2Seq: google/flan-t5-small or flan-t5-base)
2. Fast Offline NLP Synthesizer (Lexical analysis, POS tagging, rhyming schemes, stanza structuring)
"""

import re
import random
import logging
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PoemGenerator")

# Predefined poetic rhyme clusters and imagery banks
POETIC_RHYMES = {
    "light": ["bright", "sight", "night", "flight", "height", "white"],
    "night": ["light", "sight", "bright", "flight", "height"],
    "day": ["ray", "way", "play", "sway", "gray", "stay"],
    "sky": ["high", "fly", "sigh", "cry", "eye"],
    "sea": ["free", "thee", "glee", "tree", "plea"],
    "deep": ["sleep", "weep", "keep", "steep", "sweep"],
    "heart": ["art", "part", "dart", "chart", "start"],
    "soul": ["whole", "goal", "scroll", "toll", "role"],
    "wind": ["unpinned", "rescinded", "grinned", "twinned"],
    "dream": ["gleam", "stream", "beam", "seem", "supreme"],
    "time": ["rhyme", "climb", "chime", "prime", "sublime"],
    "stone": ["alone", "unknown", "grown", "flown", "shown"],
    "rain": ["pain", "drain", "strain", "remain", "domain"],
    "fire": ["desire", "higher", "pyre", "inspire", "wire"],
    "grace": ["space", "embrace", "place", "trace", "face"],
    "song": ["strong", "long", "belong", "throng"],
    "glow": ["flow", "slow", "grow", "know", "bestow"]
}

STYLE_TEMPLATES = {
    "Rhyming Stanzas": "Convert the following paragraph into an expressive, rhyming poem with clear 4-line stanzas (AABB or ABAB), poetic rhythm, and vivid imagery.",
    "Free Verse": "Transform this paragraph into a deep, evocative free verse poem with musical cadence, rich metaphors, and thoughtful line breaks.",
    "Haiku Series": "Condense the essence of this paragraph into a sequence of traditional 3-line Haiku poems (5-7-5 syllable structure).",
    "Shakespearean Sonnet": "Reimagine this paragraph as an Elizabethan or Shakespearean sonnet with iambic flow, poetic diction, and a closing rhyming couplet.",
    "Ballad": "Adapt this narrative paragraph into a lyrical ballad telling a story through rhythmic quatrains with musical flow."
}

TONE_PROMPTS = {
    "Melancholic & Reflective": "Tone: deeply reflective, sorrowful, nostalgic, and quiet.",
    "Joyful & Uplifting": "Tone: vibrant, optimistic, celebrating life and hope.",
    "Romantic & Passionate": "Tone: intimate, tender, passionate, and longing.",
    "Mystical & Nature": "Tone: enchanted, elemental, cosmic, and attuned to the natural world.",
    "Epic & Dramatic": "Tone: grand, heroic, majestic, and intense."
}


class PoemGenerator:
    def __init__(self, model_name: str = "google/flan-t5-small", device: Optional[str] = None):
        """
        Initializes the poem generator.
        Loads the Hugging Face transformer model lazily on demand.
        """
        self.model_name = model_name
        self.device = device
        self._pipeline = None
        self._model_loaded = False
        self._load_error = None

    def load_model(self) -> bool:
        """Loads Hugging Face pipeline if not already loaded."""
        if self._model_loaded:
            return True

        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading transformer model '{self.model_name}' on {self.device}...")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

            if self.device == "cuda":
                model = model.to("cuda")

            self._pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            self._model_loaded = True
            logger.info("Model loaded successfully.")
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"Failed to load transformer model '{self.model_name}': {e}.")
            return False

    def generate(
        self,
        paragraph: str,
        style: str = "Rhyming Stanzas",
        tone: str = "Mystical & Nature",
        engine_mode: str = "Auto (Smart Fallback)",
        temperature: float = 0.7,
        max_length: int = 250
    ) -> Dict[str, Any]:
        """
        Converts a paragraph into poetry.
        engine_mode:
            - 'Deep Learning (Flan-T5)': uses Transformer Seq2Seq model
            - 'Fast Offline NLP': generates instantly using rule-based cadence & rhyme synthesizer
            - 'Auto (Smart Fallback)': tries Transformer, falls back to Offline NLP on error
        """
        paragraph = paragraph.strip()
        if not paragraph:
            return {
                "poem": "Please provide a non-empty paragraph to transform into poetry.",
                "engine": "None",
                "success": False
            }

        # Case 1: Explicitly requested Fast Offline NLP
        if "Offline" in engine_mode or "Fast" in engine_mode:
            poem = self._offline_nlp_poet(paragraph, style, tone)
            return {
                "poem": poem,
                "engine": "Fast Offline NLP Poetic Synthesizer",
                "success": True,
                "style": style,
                "tone": tone
            }

        # Case 2: Deep Learning or Auto
        if self._model_loaded or self.load_model():
            try:
                style_instruction = STYLE_TEMPLATES.get(style, STYLE_TEMPLATES["Rhyming Stanzas"])
                tone_instruction = TONE_PROMPTS.get(tone, "")
                prompt = (
                    f"{style_instruction}\n"
                    f"{tone_instruction}\n\n"
                    f"Paragraph to transform:\n\"\"\"{paragraph}\"\"\"\n\n"
                    f"Poem:\n"
                )

                outputs = self._pipeline(
                    prompt,
                    max_new_tokens=max_length,
                    temperature=max(0.1, min(1.0, temperature)),
                    do_sample=True if temperature > 0.1 else False,
                    repetition_penalty=1.3,
                    num_beams=2,
                )

                raw_text = outputs[0]["generated_text"].strip()
                formatted_poem = self._format_poem_stanzas(raw_text, style)
                
                if len(formatted_poem.splitlines()) >= 3:
                    return {
                        "poem": formatted_poem,
                        "engine": f"Transformer Seq2Seq ({self.model_name})",
                        "success": True,
                        "style": style,
                        "tone": tone
                    }
            except Exception as e:
                logger.error(f"Error during transformer generation: {e}")

        # If Deep Learning was strictly requested and failed
        if "Deep Learning" in engine_mode:
            return {
                "poem": f"Transformer model could not be loaded: {self._load_error}. Try switching to 'Fast Offline NLP' mode.",
                "engine": "Error",
                "success": False
            }

        # Fallback to Offline NLP
        poem = self._offline_nlp_poet(paragraph, style, tone)
        return {
            "poem": poem,
            "engine": "Fast Offline NLP Poetic Synthesizer (Automatic Fallback)",
            "success": True,
            "style": style,
            "tone": tone
        }

    def _format_poem_stanzas(self, text: str, style: str) -> str:
        """Formats generated flat text into musical poetic stanzas with line breaks."""
        text = text.replace("\\n", "\n").replace("/ ", "\n")
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        if len(lines) <= 2:
            sentences = re.split(r'(?<=[.?!,;])\s+', text)
            lines = [s.strip() for s in sentences if len(s.strip()) > 3]

        if style in ["Rhyming Stanzas", "Ballad"]:
            stanzas = []
            current = []
            for line in lines:
                current.append(line.capitalize())
                if len(current) == 4:
                    stanzas.append("\n".join(current))
                    current = []
            if current:
                stanzas.append("\n".join(current))
            return "\n\n".join(stanzas)

        elif style == "Haiku Series":
            formatted = []
            for i, line in enumerate(lines):
                formatted.append(line.capitalize())
                if (i + 1) % 3 == 0:
                    formatted.append("")
            return "\n".join(formatted).strip()

        return "\n".join(l.capitalize() for l in lines)

    def _offline_nlp_poet(self, paragraph: str, style: str, tone: str) -> str:
        """
        Synthesizes structured poetry from input text by extracting semantic keywords,
        synthesizing imagery, and building rhythmically balanced stanzas.
        """
        words = re.findall(r'\b[A-Za-z]{3,}\b', paragraph)
        stopwords = {
            "the", "and", "that", "have", "for", "not", "with", "you", "this", "but", 
            "his", "from", "they", "say", "her", "she", "will", "one", "all", "would",
            "there", "their", "what", "out", "about", "who", "get", "which", "when", "make",
            "were", "been", "being", "was", "are", "had", "onto", "into", "upon", "over",
            "under", "more", "most", "some", "such", "than", "then", "very", "much", "also",
            "just", "like", "down", "only", "even", "back", "other", "many", "through"
        }
        content_words = [w.lower() for w in words if w.lower() not in stopwords and len(w) > 3]
        
        if not content_words:
            content_words = ["whisper", "shadow", "light", "dream", "heart", "sky", "silence", "ocean"]

        primary = content_words[:min(14, len(content_words))]
        
        theme_adj = {
            "Melancholic & Reflective": ["quiet", "silent", "fading", "hollow", "distant", "forgotten", "wistful"],
            "Joyful & Uplifting": ["golden", "shining", "radiant", "soaring", "boundless", "jubilant", "brilliant"],
            "Romantic & Passionate": ["tender", "burning", "sacred", "yearning", "gentle", "beloved", "devoted"],
            "Mystical & Nature": ["ancient", "emerald", "twilight", "crystalline", "boundless", "celestial", "verdant"],
            "Epic & Dramatic": ["thunderous", "valiant", "fateful", "iron", "mighty", "sovereign", "unyielding"]
        }.get(tone, ["gentle", "golden", "silent", "soaring"])

        theme_verbs = {
            "Melancholic & Reflective": ["wanders", "dissolves", "sighs", "remembers", "lingers", "weeps"],
            "Joyful & Uplifting": ["dances", "blooms", "awakens", "sparkles", "rejoices", "triumphs"],
            "Romantic & Passionate": ["embraces", "burns", "whispers", "cherishes", "quickens", "surrenders"],
            "Mystical & Nature": ["breathes", "flows", "ripples", "echoes", "envelops", "shimmers"],
            "Epic & Dramatic": ["charges", "strikes", "conquers", "shatters", "roars", "endures"]
        }.get(tone, ["wanders", "breathes", "awakens", "echoes"])

        def w(idx: int = 0) -> str:
            return primary[idx % len(primary)]

        def adj(idx: int = 0) -> str:
            return theme_adj[idx % len(theme_adj)]

        def verb(idx: int = 0) -> str:
            return theme_verbs[idx % len(theme_verbs)]

        if style == "Haiku Series":
            h1 = (
                f"The {adj(0)} wind wakes,\n"
                f"{w(0).capitalize()} {verb(0)} through the sky,\n"
                f"Stillness meets the light."
            )
            h2 = (
                f"Silent {w(1)} calls,\n"
                f"A {adj(1)} breath of morning,\n"
                f"Peace rests in the soul."
            )
            h3 = (
                f"Echoes of {w(2)},\n"
                f"Where the {adj(2)} river turns,\n"
                f"All becomes a dream."
            )
            return f"{h1}\n\n{h2}\n\n{h3}"

        elif style == "Free Verse":
            lines = [
                f"In the breath of {w(0)},",
                f"Where shadows learn the silence of {adj(0)} stones,",
                f"The {w(1)} {verb(0)} without a sound.",
                f"",
                f"Look how the {adj(1)} light unfastens the sky,",
                f"Carrying memory like an unread letter",
                f"Written in the language of {w(2)}.",
                f"",
                f"Here, where {w(3)} touches the edge of evening,",
                f"Every heartbeat is a doorway",
                f"Leading into the quiet {adj(2)} wild."
            ]
            return "\n".join(lines)

        elif style == "Shakespearean Sonnet":
            s1 = (
                f"When in the stillness of the {adj(0)} night,\n"
                f"The gentle breath of {w(0)} softly sighs,\n"
                f"A silver grace reveals the hidden light,\n"
                f"Beneath the splendor of eternal skies."
            )
            s2 = (
                f"No sorrow tames the {adj(1)} fire within,\n"
                f"Where {w(1)} and hope together softly dwell,\n"
                f"Above the tumult and the worldly din,\n"
                f"Like music sounding from a distant bell."
            )
            s3 = (
                f"Time gathers moments where the river flows,\n"
                f"And guides the soul across the quiet deep,\n"
                f"While in each petal of the {adj(2)} rose,\n"
                f"A sacred promise never shall it sleep."
            )
            couplet = (
                f"For while the earth preserves each fleeting theme,\n"
                f"Thy {w(2)} lives forever in this dream."
            )
            return f"{s1}\n\n{s2}\n\n{s3}\n\n{couplet}"

        else:
            # Rhyming Stanzas / Ballad
            # Choose complementary rhymes
            rhyme_keys = list(POETIC_RHYMES.keys())
            rk1 = rhyme_keys[hash(paragraph) % len(rhyme_keys)]
            rk2 = rhyme_keys[(hash(paragraph) + 3) % len(rhyme_keys)]
            if rk1 == rk2:
                rk2 = rhyme_keys[(hash(paragraph) + 5) % len(rhyme_keys)]

            r_rhyme1 = POETIC_RHYMES[rk1][0]
            r_rhyme2 = POETIC_RHYMES[rk2][0]

            rk3 = rhyme_keys[(hash(paragraph) + 7) % len(rhyme_keys)]
            rk4 = rhyme_keys[(hash(paragraph) + 11) % len(rhyme_keys)]
            r_rhyme3 = POETIC_RHYMES[rk3][0]
            r_rhyme4 = POETIC_RHYMES[rk4][0]

            stanza1 = (
                f"When twilight covers the serene expanse of {rk1},\n"
                f"The whisper of the {w(0)} begins to gently {rk2},\n"
                f"A silver rhythm dances in the velvet {r_rhyme1},\n"
                f"While silent wonders through the shadows softly {r_rhyme2}."
            )

            stanza2 = (
                f"Across the valley where the {adj(1)} rivers {rk3},\n"
                f"The echoes of {w(1)} awaken like a distant {rk4},\n"
                f"In every {adj(2)} contour where the currents {r_rhyme3},\n"
                f"The heart finds solace and a melody that does {r_rhyme4}."
            )

            stanza3 = (
                f"Now let the sorrow and the weary world take {w(2)},\n"
                f"For hope is woven in the fabric of the {adj(0)} night,\n"
                f"As morning gently turns the darkness into flight,\n"
                f"And gives the spirit back its everlasting light."
            )

            return f"{stanza1}\n\n{stanza2}\n\n{stanza3}"
