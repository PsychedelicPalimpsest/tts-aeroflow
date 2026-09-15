"""
AeroFlow-v2 Phonemizer & G2P Tokenizer.
Provides:
- Exact 84-token ARPAbet vocabulary (including special tokens, punctuation, and vowels/consonants).
- Comprehensive built-in CMUdict lexicon hash table covering frequent English vocabulary,
  pronouns, auxiliary verbs, contractions, numbers, and test phrases.
- Deterministic Letter-to-Sound (LTS) rule-based fallback tokenizer for out-of-vocabulary words.
- Bidirectional token mapping (phonemes <-> integer IDs in [0, 83]).
"""

import re
from typing import Dict, List, Optional
from aeroflow.frontend.text_norm import TextNormalizer

# ==============================================================================
# 84-TOKEN ARPABET VOCABULARY DEFINITION
# ==============================================================================

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<sp>"]

PUNCTUATION_TOKENS = [",", ".", "!", "?", "-", ":", ";", '"', "'"]

CONSONANT_TOKENS = [
    "B", "CH", "D", "DH", "DX", "F", "G", "HH", "JH", "K", "L", "M",
    "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"
]  # 25 consonants (including alveolar flap DX)

# 15 ARPAbet Vowels with 3 stress levels (0: unstressed, 1: primary, 2: secondary)
BASE_VOWELS = [
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH",
    "IY", "OW", "OY", "UH", "UW"
]

VOWEL_TOKENS_0 = [f"{v}0" for v in BASE_VOWELS]  # 15
VOWEL_TOKENS_1 = [f"{v}1" for v in BASE_VOWELS]  # 15
VOWEL_TOKENS_2 = [f"{v}2" for v in BASE_VOWELS]  # 15

ALL_TOKENS: List[str] = (
    SPECIAL_TOKENS + PUNCTUATION_TOKENS + CONSONANT_TOKENS +
    VOWEL_TOKENS_0 + VOWEL_TOKENS_1 + VOWEL_TOKENS_2
)

assert len(ALL_TOKENS) == 84, f"Expected 84 tokens, got {len(ALL_TOKENS)}"

PHONEME_TO_ID: Dict[str, int] = {p: i for i, p in enumerate(ALL_TOKENS)}
ID_TO_PHONEME: Dict[int, str] = {i: p for i, p in enumerate(ALL_TOKENS)}

PAD_ID = PHONEME_TO_ID["<pad>"]
BOS_ID = PHONEME_TO_ID["<bos>"]
EOS_ID = PHONEME_TO_ID["<eos>"]
UNK_ID = PHONEME_TO_ID["<unk>"]
SPACE_ID = PHONEME_TO_ID["<sp>"]

# ==============================================================================
# IN-MEMORY CMUDICT LEXICON HASH TABLE (CORE VOCABULARY & EXTENDED CORPUS)
# ==============================================================================

_CMUDICT: Dict[str, List[str]] = {
    # Peter Piper tongue twister
    "peter": ["P", "IY1", "T", "ER0"],
    "piper": ["P", "AY1", "P", "ER0"],
    "picked": ["P", "IH1", "K", "T"],
    "a": ["AH0"],
    "peck": ["P", "EH1", "K"],
    "of": ["AH1", "V"],
    "pickled": ["P", "IH1", "K", "AH0", "L", "D"],
    "peppers": ["P", "EH1", "P", "ER0", "Z"],
    "where": ["W", "EH1", "R"],
    "is": ["IH1", "Z"],
    "the": ["DH", "AH0"],
    "that": ["DH", "AE1", "T"],

    # Common English words & Audio ML domain
    "aeroflow": ["EH1", "R", "OW0", "F", "L", "OW2"],
    "speech": ["S", "P", "IY1", "CH"],
    "synthesis": ["S", "IH1", "N", "TH", "AH0", "S", "AH0", "S"],
    "engine": ["EH1", "N", "JH", "AH0", "N"],
    "deep": ["D", "IY1", "P"],
    "resonant": ["R", "EH1", "Z", "AH0", "N", "AH0", "N", "T"],
    "authoritative": ["AH0", "TH", "AO1", "R", "AH0", "T", "EY2", "T", "IH0", "V"],
    "american": ["AH0", "M", "EH1", "R", "AH0", "K", "AH0", "N"],
    "male": ["M", "EY1", "L"],
    "voice": ["V", "OY1", "S"],
    "john": ["JH", "AA1", "N"],
    "van": ["V", "AE1", "N"],
    "stan": ["S", "T", "AE1", "N"],
    "intel": ["IH1", "N", "T", "EH2", "L"],
    "core": ["K", "AO1", "R"],
    "audio": ["AO1", "D", "IY0", "OW2"],
    "sound": ["S", "AW1", "N", "D"],
    "high": ["HH", "AY1"],
    "fidelity": ["F", "AH0", "D", "EH1", "L", "AH0", "T", "IY0"],
    "flow": ["F", "L", "OW1"],
    "matching": ["M", "AE1", "CH", "IH0", "NG"],
    "heun": ["HH", "OY1", "N"],
    "solver": ["S", "AA1", "L", "V", "ER0"],
    "fast": ["F", "AE1", "S", "T"],
    "real": ["R", "IY1", "L"],
    "time": ["T", "AY1", "M"],
    "factor": ["F", "AE1", "K", "T", "ER0"],
    "test": ["T", "EH1", "S", "T"],
    "run": ["R", "AH1", "N"],
    "training": ["T", "R", "EY1", "N", "IH0", "NG"],
    "dry": ["D", "R", "AY1"],
    "step": ["S", "T", "EH1", "P"],
    "loss": ["L", "AO1", "S"],
    "model": ["M", "AA1", "D", "AH0", "L"],
    "clean": ["K", "L", "IY1", "N"],
    "studio": ["S", "T", "UW1", "D", "IY0", "OW2"],
    "pristine": ["P", "R", "IH1", "S", "T", "IY0", "N"],
    "transients": ["T", "R", "AE1", "N", "ZH", "AH0", "N", "T", "S"],
    "formants": ["F", "AO1", "R", "M", "AH0", "N", "T", "S"],
    "quick": ["K", "W", "IH1", "K"],
    "brown": ["B", "R", "AW1", "N"],
    "fox": ["F", "AA1", "K", "S"],
    "jumps": ["JH", "AH1", "M", "P", "S"],
    "over": ["OW1", "V", "ER0"],
    "lazy": ["L", "EY1", "Z", "IY0"],
    "dog": ["D", "AO1", "G"],

    # Essential Auxiliary & High-Frequency Verbs
    "have": ["HH", "AE1", "V"],
    "has": ["HH", "AE1", "Z"],
    "had": ["HH", "AE1", "D"],
    "having": ["HH", "AE1", "V", "IH0", "NG"],
    "do": ["D", "UW1"],
    "does": ["D", "AH1", "Z"],
    "did": ["D", "IH1", "D"],
    "doing": ["D", "UW1", "IH0", "NG"],
    "done": ["D", "AH1", "N"],
    "be": ["B", "IY1"],
    "am": ["AE1", "M"],
    "are": ["AA1", "R"],
    "was": ["W", "AA1", "Z"],
    "were": ["W", "ER1"],
    "been": ["B", "IH1", "N"],
    "being": ["B", "IY1", "IH0", "NG"],
    "go": ["G", "OW1"],
    "goes": ["G", "OW1", "Z"],
    "went": ["W", "EH1", "N", "T"],
    "gone": ["G", "AO1", "N"],
    "going": ["G", "OW1", "IH0", "NG"],
    "say": ["S", "EY1"],
    "says": ["S", "EH1", "Z"],
    "said": ["S", "EH1", "D"],
    "saying": ["S", "EY1", "IH0", "NG"],
    "get": ["G", "EH1", "T"],
    "gets": ["G", "EH1", "T", "S"],
    "got": ["G", "AA1", "T"],
    "make": ["M", "EY1", "K"],
    "makes": ["M", "EY1", "K", "S"],
    "made": ["M", "EY1", "D"],
    "know": ["N", "OW1"],
    "knows": ["N", "OW1", "Z"],
    "knew": ["N", "UW1"],
    "known": ["N", "OW1", "N"],
    "think": ["TH", "IH1", "NG", "K"],
    "thinks": ["TH", "IH1", "NG", "K", "S"],
    "thought": ["TH", "AO1", "T"],
    "take": ["T", "EY1", "K"],
    "takes": ["T", "EY1", "K", "S"],
    "took": ["T", "UH1", "K"],
    "see": ["S", "IY1"],
    "sees": ["S", "IY1", "Z"],
    "saw": ["S", "AO1"],
    "seen": ["S", "IY1", "N"],
    "come": ["K", "AH1", "M"],
    "comes": ["K", "AH1", "M", "Z"],
    "came": ["K", "EY1", "M"],
    "want": ["W", "AA1", "N", "T"],
    "wants": ["W", "AA1", "N", "T", "S"],
    "look": ["L", "UH1", "K"],
    "looks": ["L", "UH1", "K", "S"],
    "looked": ["L", "UH1", "K", "T"],
    "use": ["Y", "UW1", "Z"],
    "uses": ["Y", "UW1", "Z", "AH0", "Z"],
    "used": ["Y", "UW1", "Z", "D"],
    "find": ["F", "AY1", "N", "D"],
    "found": ["F", "AW1", "N", "D"],
    "give": ["G", "IH1", "V"],
    "gave": ["G", "EY1", "V"],
    "tell": ["T", "EH1", "L"],
    "told": ["T", "OW1", "L", "D"],
    "ask": ["AE1", "S", "K"],
    "asked": ["AE1", "S", "K", "T"],
    "work": ["W", "ER1", "K"],
    "works": ["W", "ER1", "K", "S"],
    "worked": ["W", "ER1", "K", "T"],
    "feel": ["F", "IY1", "L"],
    "felt": ["F", "EH1", "L", "T"],
    "try": ["T", "R", "AY1"],
    "leave": ["L", "IY1", "V"],
    "left": ["L", "EH1", "F", "T"],
    "call": ["K", "AO1", "L"],
    "called": ["K", "AO1", "L", "D"],
    "talk": ["T", "AO1", "K"],
    "talked": ["T", "AO1", "K", "T"],
    "walk": ["W", "AO1", "K"],
    "live": ["L", "IH1", "V"],
    "lived": ["L", "IH1", "V", "D"],
    "turn": ["T", "ER1", "N"],
    "start": ["S", "T", "AA1", "R", "T"],
    "show": ["SH", "OW1"],
    "hear": ["HH", "IY1", "R"],
    "heard": ["HH", "ER1", "D"],
    "play": ["P", "L", "EY1"],
    "move": ["M", "UW1", "V"],
    "like": ["L", "AY1", "K"],
    "liked": ["L", "AY1", "K", "T"],
    "bring": ["B", "R", "IH1", "NG"],
    "brought": ["B", "R", "AO1", "T"],
    "write": ["R", "AY1", "T"],
    "wrote": ["R", "OW1", "T"],
    "read": ["R", "IY1", "D"],
    "open": ["OW1", "P", "AH0", "N"],
    "close": ["K", "L", "OW1", "Z"],

    # Modal verbs
    "can": ["K", "AE1", "N"],
    "could": ["K", "UH1", "D"],
    "will": ["W", "IH1", "L"],
    "would": ["W", "UH1", "D"],
    "shall": ["SH", "AE1", "L"],
    "should": ["SH", "UH1", "D"],
    "may": ["M", "EY1"],
    "might": ["M", "AY1", "T"],
    "must": ["M", "AH1", "S", "T"],

    # Common Contractions
    "don't": ["D", "OW1", "N", "T"],
    "doesn't": ["D", "AH1", "Z", "AH0", "N", "T"],
    "didn't": ["D", "IH1", "D", "AH0", "N", "T"],
    "can't": ["K", "AE1", "N", "T"],
    "cannot": ["K", "AE1", "N", "AA0", "T"],
    "couldn't": ["K", "UH1", "D", "AH0", "N", "T"],
    "won't": ["W", "OW1", "N", "T"],
    "wouldn't": ["W", "UH1", "D", "AH0", "N", "T"],
    "shouldn't": ["SH", "UH1", "D", "AH0", "N", "T"],
    "isn't": ["IH1", "Z", "AH0", "N", "T"],
    "aren't": ["AA1", "R", "N", "T"],
    "wasn't": ["W", "AA1", "Z", "AH0", "N", "T"],
    "weren't": ["W", "ER1", "N", "T"],
    "haven't": ["HH", "AE1", "V", "AH0", "N", "T"],
    "hasn't": ["HH", "AE1", "Z", "AH0", "N", "T"],
    "hadn't": ["HH", "AE1", "D", "AH0", "N", "T"],
    "it's": ["IH1", "T", "S"],
    "that's": ["DH", "AE1", "T", "S"],
    "what's": ["W", "AH1", "T", "S"],
    "there's": ["DH", "EH1", "R", "Z"],
    "here's": ["HH", "IY1", "R", "Z"],
    "he's": ["HH", "IY1", "Z"],
    "she's": ["SH", "IY1", "Z"],
    "i'm": ["AY1", "M"],
    "you're": ["Y", "UH1", "R"],
    "we're": ["W", "IY1", "R"],
    "they're": ["DH", "EH1", "R"],
    "i've": ["AY1", "V"],
    "you've": ["Y", "UW1", "V"],
    "we've": ["W", "IY1", "V"],
    "they've": ["DH", "EY1", "V"],
    "i'll": ["AY1", "L"],
    "you'll": ["Y", "UW1", "L"],
    "he'll": ["HH", "IY1", "L"],
    "she'll": ["SH", "IY1", "L"],
    "we'll": ["W", "IY1", "L"],
    "they'll": ["DH", "EY1", "L"],
    "i'd": ["AY1", "D"],
    "you'd": ["Y", "UW1", "D"],
    "he'd": ["HH", "IY1", "D"],
    "she'd": ["SH", "IY1", "D"],
    "we'd": ["W", "IY1", "D"],
    "they'd": ["DH", "EY1", "D"],
    "let's": ["L", "EH1", "T", "S"],

    # Pronouns & function words
    "i": ["AY1"],
    "you": ["Y", "UW1"],
    "he": ["HH", "IY1"],
    "she": ["SH", "IY1"],
    "it": ["IH1", "T"],
    "we": ["W", "IY1"],
    "they": ["DH", "EY1"],
    "me": ["M", "IY1"],
    "him": ["HH", "IH1", "M"],
    "her": ["HH", "ER1"],
    "us": ["AH1", "S"],
    "them": ["DH", "EH1", "M"],
    "my": ["M", "AY1"],
    "your": ["Y", "AO1", "R"],
    "his": ["HH", "IH1", "Z"],
    "its": ["IH1", "T", "S"],
    "our": ["AW1", "ER0"],
    "their": ["DH", "EH1", "R"],
    "this": ["DH", "IH1", "S"],
    "these": ["DH", "IY1", "Z"],
    "those": ["DH", "OW1", "Z"],
    "what": ["W", "AH1", "T"],
    "which": ["W", "IH1", "CH"],
    "who": ["HH", "UW1"],
    "whom": ["HH", "UW1", "M"],
    "whose": ["HH", "UW1", "Z"],
    "why": ["W", "AY1"],
    "how": ["HH", "AW1"],
    "when": ["W", "EH1", "N"],
    "and": ["AE1", "N", "D"],
    "or": ["AO1", "R"],
    "but": ["B", "AH1", "T"],
    "if": ["IH1", "F"],
    "because": ["B", "IH0", "K", "AO1", "Z"],
    "as": ["AE1", "Z"],
    "until": ["AH0", "N", "T", "IH1", "L"],
    "while": ["W", "AY1", "L"],
    "in": ["IH1", "N"],
    "on": ["AA1", "N"],
    "at": ["AE1", "T"],
    "by": ["B", "AY1"],
    "for": ["F", "AO1", "R"],
    "with": ["W", "IH1", "DH"],
    "about": ["AH0", "B", "AW1", "T"],
    "against": ["AH0", "G", "EH1", "N", "S", "T"],
    "between": ["B", "IH0", "T", "W", "IY1", "N"],
    "into": ["IH1", "N", "T", "UW0"],
    "through": ["TH", "R", "UW1"],
    "during": ["D", "UH1", "R", "IH0", "NG"],
    "before": ["B", "IH0", "F", "AO1", "R"],
    "after": ["AE1", "F", "T", "ER0"],
    "above": ["AH0", "B", "AH1", "V"],
    "below": ["B", "IH0", "L", "OW1"],
    "to": ["T", "UW1"],
    "from": ["F", "R", "AH1", "M"],
    "up": ["AH1", "P"],
    "down": ["D", "AW1", "N"],
    "out": ["AW1", "T"],
    "off": ["AO1", "F"],
    "again": ["AH0", "G", "EH1", "N"],
    "further": ["F", "ER1", "DH", "ER0"],
    "then": ["DH", "EH1", "N"],
    "once": ["W", "AH1", "N", "S"],
    "here": ["HH", "IY1", "R"],
    "there": ["DH", "EH1", "R"],
    "all": ["AO1", "L"],
    "any": ["EH1", "N", "IY0"],
    "both": ["B", "OW1", "TH"],
    "each": ["IY1", "CH"],
    "few": ["F", "Y", "UW1"],
    "more": ["M", "AO1", "R"],
    "most": ["M", "OW1", "S", "T"],
    "other": ["AH1", "DH", "ER0"],
    "some": ["S", "AH1", "M"],
    "such": ["S", "AH1", "CH"],
    "no": ["N", "OW1"],
    "nor": ["N", "AO1", "R"],
    "not": ["N", "AA1", "T"],
    "only": ["OW1", "N", "L", "IY0"],
    "own": ["OW1", "N"],
    "same": ["S", "EY1", "M"],
    "so": ["S", "OW1"],
    "than": ["DH", "AE1", "N"],
    "too": ["T", "UW1"],
    "very": ["V", "EH1", "R", "IY0"],
    "just": ["JH", "AH1", "S", "T"],
    "now": ["N", "AW1"],

    # Common Nouns & Adjectives
    "man": ["M", "AE1", "N"],
    "men": ["M", "EH1", "N"],
    "woman": ["W", "UH1", "M", "AH0", "N"],
    "women": ["W", "IH1", "M", "AH0", "N"],
    "child": ["CH", "AY1", "L", "D"],
    "children": ["CH", "IH1", "L", "D", "R", "AH0", "N"],
    "person": ["P", "ER1", "S", "AH0", "N"],
    "people": ["P", "IY1", "P", "AH0", "L"],
    "year": ["Y", "IH1", "R"],
    "years": ["Y", "IH1", "R", "Z"],
    "day": ["D", "EY1"],
    "days": ["D", "EY1", "Z"],
    "way": ["W", "EY1"],
    "thing": ["TH", "IH1", "NG"],
    "things": ["TH", "IH1", "NG", "Z"],
    "life": ["L", "AY1", "F"],
    "world": ["W", "ER1", "L", "D"],
    "hand": ["HH", "AE1", "N", "D"],
    "part": ["P", "AA1", "R", "T"],
    "place": ["P", "L", "EY1", "S"],
    "week": ["W", "IY1", "K"],
    "case": ["K", "EY1", "S"],
    "group": ["G", "R", "UW1", "P"],
    "problem": ["P", "R", "AA1", "B", "L", "AH0", "M"],
    "fact": ["F", "AE1", "K", "T"],
    "good": ["G", "UH1", "D"],
    "new": ["N", "UW1"],
    "last": ["L", "AE1", "S", "T"],
    "long": ["L", "AO1", "NG"],
    "great": ["G", "R", "EY1", "T"],
    "little": ["L", "IH1", "T", "AH0", "L"],
    "old": ["OW1", "L", "D"],
    "right": ["R", "AY1", "T"],
    "big": ["B", "IH1", "G"],
    "small": ["S", "M", "AO1", "L"],
    "large": ["L", "AA1", "R", "JH"],
    "next": ["N", "EH1", "K", "S", "T"],
    "early": ["ER1", "L", "IY0"],
    "young": ["Y", "AH1", "NG"],
    "bad": ["B", "AE1", "D"],
    "street": ["S", "T", "R", "IY1", "T"],
    "saint": ["S", "EY1", "N", "T"],
    "doctor": ["D", "AA1", "K", "T", "ER0"],
    "mister": ["M", "IH1", "S", "T", "ER0"],

    # Numbers
    "zero": ["Z", "IY1", "R", "OW0"],
    "one": ["W", "AH1", "N"],
    "two": ["T", "UW1"],
    "three": ["TH", "R", "IY1"],
    "four": ["F", "AO1", "R"],
    "five": ["F", "AY1", "V"],
    "six": ["S", "IH1", "K", "S"],
    "seven": ["S", "EH1", "V", "AH0", "N"],
    "eight": ["EY1", "T"],
    "nine": ["N", "AY1", "N"],
    "ten": ["T", "EH1", "N"],
    "eleven": ["IH0", "L", "EH1", "V", "AH0", "N"],
    "twelve": ["T", "W", "EH1", "L", "V"],
    "thirteen": ["TH", "ER0", "T", "IY1", "N"],
    "fourteen": ["F", "AO0", "R", "T", "IY1", "N"],
    "fifteen": ["F", "IH0", "F", "T", "IY1", "N"],
    "sixteen": ["S", "IH0", "K", "S", "T", "IY1", "N"],
    "seventeen": ["S", "EH0", "V", "AH0", "N", "T", "IY1", "N"],
    "eighteen": ["EY0", "T", "IY1", "N"],
    "nineteen": ["N", "AY0", "N", "T", "IY1", "N"],
    "twenty": ["T", "W", "EH1", "N", "T", "IY0"],
    "thirty": ["TH", "ER1", "T", "IY0"],
    "forty": ["F", "AO1", "R", "T", "IY0"],
    "fifty": ["F", "IH1", "F", "T", "IY0"],
    "sixty": ["S", "IH1", "K", "S", "T", "IY0"],
    "seventy": ["S", "EH1", "V", "AH0", "N", "T", "IY0"],
    "eighty": ["EY1", "T", "IY0"],
    "ninety": ["N", "AY1", "N", "T", "IY0"],
    "hundred": ["HH", "AH1", "N", "D", "R", "AH0", "D"],
    "thousand": ["TH", "AW1", "Z", "AH0", "N", "D"],
    "million": ["M", "IH1", "L", "Y", "AH0", "N"],
    "billion": ["B", "IH1", "L", "Y", "AH0", "N"],
    "first": ["F", "ER1", "S", "T"],
    "second": ["S", "EH1", "K", "AH0", "N", "D"],
    "third": ["TH", "ER1", "D"],
    "point": ["P", "OY1", "N", "T"],
    "dollar": ["D", "AA1", "L", "ER0"],
    "dollars": ["D", "AA1", "L", "ER0", "Z"],
    "cent": ["S", "EH1", "N", "T"],
    "cents": ["S", "EH1", "N", "T", "S"],
    "percent": ["P", "ER0", "S", "EH1", "N", "T"],
}

# Single character fallback mappings
_CHAR_TO_PHONE: Dict[str, List[str]] = {
    "a": ["AE1"],
    "b": ["B"],
    "c": ["K"],
    "d": ["D"],
    "e": ["EH1"],
    "f": ["F"],
    "g": ["G"],
    "h": ["HH"],
    "i": ["IH1"],
    "j": ["JH"],
    "k": ["K"],
    "l": ["L"],
    "m": ["M"],
    "n": ["N"],
    "o": ["AA1"],
    "p": ["P"],
    "q": ["K"],
    "r": ["R"],
    "s": ["S"],
    "t": ["T"],
    "u": ["AH1"],
    "v": ["V"],
    "w": ["W"],
    "x": ["K", "S"],
    "y": ["Y"],
    "z": ["Z"],
}


class Phonemizer:
    """Phonemizer converting normalized text into ARPAbet tokens and IDs."""

    def __init__(self):
        self.normalizer = TextNormalizer()
        self.lexicon = _CMUDICT
        self.vocab = ALL_TOKENS
        self.phoneme_to_id = PHONEME_TO_ID
        self.id_to_phoneme = ID_TO_PHONEME

    def _fallback_lts(self, word: str) -> List[str]:
        """Deterministic rule-based letter-to-sound for unknown words."""
        word = word.lower()
        phones: List[str] = []
        i = 0
        w_len = len(word)

        # Check for trailing silent-e rule: vowel + consonant + e (e.g., make, fine, note, cute)
        has_silent_e = (
            w_len >= 3 and
            word[-1] == "e" and
            word[-2] not in "aeiouy" and
            word[-3] in "aeiou" and
            (w_len == 3 or word[-4] not in "aeiou")
        )

        while i < w_len:
            # Skip silent e at end if triggered
            if has_silent_e and i == w_len - 1:
                break

            # Handle 3-letter combinations
            if i + 3 <= w_len:
                tri = word[i:i+3]
                if tri == "igh":
                    phones.append("AY1")
                    i += 3
                    continue
                elif tri == "tch":
                    phones.append("CH")
                    i += 3
                    continue
                elif tri == "dge":
                    phones.append("JH")
                    i += 3
                    continue

            # Handle 2-letter digraphs
            if i + 2 <= w_len:
                pair = word[i:i+2]
                if pair == "th":
                    phones.append("TH")
                    i += 2
                    continue
                elif pair == "sh":
                    phones.append("SH")
                    i += 2
                    continue
                elif pair == "ch":
                    phones.append("CH")
                    i += 2
                    continue
                elif pair == "ph":
                    phones.append("F")
                    i += 2
                    continue
                elif pair == "wh":
                    phones.append("W")
                    i += 2
                    continue
                elif pair == "ck":
                    phones.append("K")
                    i += 2
                    continue
                elif pair == "ng":
                    phones.append("NG")
                    i += 2
                    continue
                elif pair == "qu":
                    phones.extend(["K", "W"])
                    i += 2
                    continue
                elif pair == "ee":
                    phones.append("IY1")
                    i += 2
                    continue
                elif pair == "oo":
                    phones.append("UW1")
                    i += 2
                    continue
                elif pair == "ea":
                    phones.append("IY1")
                    i += 2
                    continue
                elif pair == "ai" or pair == "ay":
                    phones.append("EY1")
                    i += 2
                    continue
                elif pair == "oa" or pair == "oe":
                    phones.append("OW1")
                    i += 2
                    continue
                elif pair == "oi" or pair == "oy":
                    phones.append("OY1")
                    i += 2
                    continue
                elif pair == "ou" or pair == "ow":
                    phones.append("AW1")
                    i += 2
                    continue
                elif pair == "au" or pair == "aw":
                    phones.append("AO1")
                    i += 2
                    continue
                elif pair == "ew" or pair == "ue" or pair == "ui":
                    phones.append("UW1")
                    i += 2
                    continue
                elif pair == "kn" and i == 0:
                    phones.append("N")
                    i += 2
                    continue
                elif pair == "wr" and i == 0:
                    phones.append("R")
                    i += 2
                    continue

            # Check silent-e long vowel
            if has_silent_e and i == w_len - 3:
                vowel = word[i]
                long_map = {
                    "a": "EY1",
                    "e": "IY1",
                    "i": "AY1",
                    "o": "OW1",
                    "u": "UW1"
                }
                if vowel in long_map:
                    phones.append(long_map[vowel])
                    i += 1
                    continue

            ch = word[i]
            if ch in _CHAR_TO_PHONE:
                phones.extend(_CHAR_TO_PHONE[ch])
            i += 1

        return phones if phones else ["AH0"]

    def phonemize_word(self, word: str) -> List[str]:
        """Look up a clean word in lexicon or compute LTS fallback."""
        word = word.lower().strip()
        if not word:
            return []
        if word in self.lexicon:
            return list(self.lexicon[word])
        # Try without trailing punctuation/apostrophe
        clean_word = word.replace("'", "")
        if clean_word in self.lexicon:
            return list(self.lexicon[clean_word])
        return self._fallback_lts(word)

    def phonemize(self, text: str, add_bos_eos: bool = True) -> List[str]:
        """Convert arbitrary English text into a list of phonemes and punctuation."""
        norm_text = self.normalizer.normalize(text)
        if not norm_text:
            return ["<bos>", "<eos>"] if add_bos_eos else []

        # Tokenize by words and punctuation
        tokens = re.findall(r"[\w'-]+|[.,!?;:\"']", norm_text)
        result: List[str] = []

        if add_bos_eos:
            result.append("<bos>")

        for i, token in enumerate(tokens):
            if token in PUNCTUATION_TOKENS:
                result.append(token)
            else:
                phones = self.phonemize_word(token)
                result.extend(phones)
                # Append space token if not followed by punctuation or end
                if i < len(tokens) - 1 and tokens[i + 1] not in PUNCTUATION_TOKENS:
                    result.append("<sp>")

        if add_bos_eos:
            result.append("<eos>")

        return result

    def text_to_sequence(self, text: str, add_bos_eos: bool = True) -> List[int]:
        """Directly map raw text into a list of token IDs."""
        phonemes = self.phonemize(text, add_bos_eos=add_bos_eos)
        return [self.phoneme_to_id.get(p, UNK_ID) for p in phonemes]

    def sequence_to_phonemes(self, sequence: List[int]) -> List[str]:
        """Convert token IDs back into phoneme representation."""
        return [self.id_to_phoneme.get(idx, "<unk>") for idx in sequence]
