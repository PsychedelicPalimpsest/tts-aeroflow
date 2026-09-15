"""
AeroFlow-v2 Text Normalization Frontend.
Provides deterministic, rule-based text cleaning and spoken-form expansion:
- Numbers (integers, decimals, ordinals, currency, percentages)
- Common English abbreviations and titles
- Mathematical and ASCII symbols
- Punctuation cleaning and whitespace standardization
"""

import re
from typing import Dict, List, Optional

_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen"
]

_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
]

_ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth", 20: "twentieth", 30: "thirtieth", 40: "fortieth",
    50: "fiftieth", 60: "sixtieth", 70: "seventieth", 80: "eightieth",
    90: "ninetieth"
}

_ABBREVIATIONS: Dict[str, str] = {
    r"\bmr\.": "mister",
    r"\bmrs\.": "missus",
    r"\bms\.": "miss",
    r"\bdr\.": "doctor",
    r"\bprof\.": "professor",
    r"\bgen\.": "general",
    r"\bcol\.": "colonel",
    r"\bcapt\.": "captain",
    r"\blt\.": "lieutenant",
    r"\bsgt\.": "sergeant",
    r"\brp\.": "representative",
    r"\bsen\.": "senator",
    r"\bgov\.": "governor",
    r"\bdept\.": "department",
    r"\best\.": "established",
    r"\bcorp\.": "corporation",
    r"\binc\.": "incorporated",
    r"\bltd\.": "limited",
    r"\bco\.": "company",
    r"\bvs\.": "versus",
    r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is",
    r"\betc\.": "etcetera",
    r"\bapprox\.": "approximately",
    r"\bave\.": "avenue",
    r"\brd\.": "road",
    r"\bblvd\.": "boulevard",
}

_SYMBOLS: Dict[str, str] = {
    "%": " percent ",
    "&": " and ",
    "+": " plus ",
    "=": " equals ",
    "@": " at ",
    "#": " number ",
    "/": " slash ",
    "\\": " backslash ",
    "*": " times ",
    "<": " less than ",
    ">": " greater than ",
}


def _integer_to_words(n: int) -> str:
    """Convert an integer in [0, 999,999,999,999] into words."""
    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + _integer_to_words(-n)

    parts: List[str] = []

    billions = n // 1_000_000_000
    n %= 1_000_000_000
    if billions > 0:
        parts.append(f"{_integer_to_words(billions)} billion")

    millions = n // 1_000_000
    n %= 1_000_000
    if millions > 0:
        parts.append(f"{_integer_to_words(millions)} million")

    thousands = n // 1000
    n %= 1000
    if thousands > 0:
        parts.append(f"{_integer_to_words(thousands)} thousand")

    hundreds = n // 100
    n %= 100
    if hundreds > 0:
        parts.append(f"{_ONES[hundreds]} hundred")

    if n > 0:
        if n < 20:
            parts.append(_ONES[n])
        else:
            tens = _TENS[n // 10]
            ones = _ONES[n % 10]
            parts.append(f"{tens}-{ones}" if ones else tens)

    return " ".join(parts)


def _ordinal_to_words(n: int) -> str:
    """Convert an integer into its ordinal word form (e.g. 1 -> first, 23 -> twenty-third)."""
    if n in _ORDINALS:
        return _ORDINALS[n]
    if n < 20:
        return _ONES[n] + "th"
    if n < 100:
        tens = _TENS[n // 10]
        rem = n % 10
        if rem == 0:
            return tens[:-1] + "ieth" if tens.endswith("y") else tens + "th"
        return f"{tens}-{_ordinal_to_words(rem)}"
    base = _integer_to_words(n - (n % 10))
    rem = n % 10
    if rem == 0:
        return base + "th"
    return f"{base} {_ordinal_to_words(rem)}"


def _expand_currency(match: re.Match) -> str:
    sym = match.group(1)
    dollars = int(match.group(2).replace(",", ""))
    cents = match.group(3)
    
    currency_names = {
        "$": ("dollar", "dollars", "cent", "cents"),
        "€": ("euro", "euros", "cent", "cents"),
        "£": ("pound", "pounds", "penny", "pence"),
    }
    unit_s, unit_p, sub_s, sub_p = currency_names.get(sym, ("dollar", "dollars", "cent", "cents"))
    
    result = []
    if dollars == 1:
        result.append(f"one {unit_s}")
    elif dollars > 1 or not cents:
        result.append(f"{_integer_to_words(dollars)} {unit_p}")
        
    if cents:
        cents_int = int(cents)
        if cents_int == 1:
            result.append(f"one {sub_s}")
        elif cents_int > 0:
            result.append(f"{_integer_to_words(cents_int)} {sub_p}")
            
    return " ".join(result) if result else f"zero {unit_p}"


def _expand_decimals(match: re.Match) -> str:
    int_part = match.group(1).replace(",", "")
    dec_part = match.group(2)
    int_words = _integer_to_words(int(int_part))
    dec_words = " ".join(_ONES[int(digit)] if digit != "0" else "zero" for digit in dec_part)
    return f"{int_words} point {dec_words}"


def _expand_ordinals(match: re.Match) -> str:
    number = int(match.group(1).replace(",", ""))
    return _ordinal_to_words(number)


def _expand_integers(match: re.Match) -> str:
    val = int(match.group(0).replace(",", ""))
    # Check for likely 4-digit years (e.g. 1984 -> nineteen eighty-four)
    if 1000 <= val <= 2099 and val % 100 != 0 and "," not in match.group(0):
        century = val // 100
        decade = val % 100
        if 11 <= century <= 19 and decade < 10:
            return f"{_ONES[century]} o-{_ONES[decade]}"
        elif 11 <= century <= 19:
            return f"{_ONES[century]} {_integer_to_words(decade)}"
        elif century == 20 and decade < 10:
            return f"two thousand {_ONES[decade]}"
    return _integer_to_words(val)


class TextNormalizer:
    """Deterministic English text normalizer for AeroFlow-v2."""

    def __init__(self):
        self.abbr_patterns = [(re.compile(p, re.IGNORECASE), repl) for p, repl in _ABBREVIATIONS.items()]
        self.currency_regex = re.compile(r"([$€£])(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{2}))?")
        self.decimal_regex = re.compile(r"(\d+)\.(\d+)")
        self.ordinal_regex = re.compile(r"(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
        self.integer_regex = re.compile(r"\b(?:\d{1,3}(?:,\d{3})+|\d+)\b")

    def normalize(self, text: str) -> str:
        """Applies full normalization pipeline to raw input string."""
        if not text:
            return ""

        # 1. Normalize unicode quotes, dashes, ellipses
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("—", ", ").replace("–", ", ")
        text = text.replace("…", "...")

        # Context-dependent Saint vs Street expansion
        text = re.sub(r"\b[Ss]t\.(?=\s+[A-Z])", "saint", text)
        text = re.sub(r"\b[Ss]t\.", "street", text)

        # 2. Expand abbreviations
        for pattern, replacement in self.abbr_patterns:
            text = pattern.sub(replacement, text)

        # 3. Currency ($12.50 -> twelve dollars fifty cents)
        text = self.currency_regex.sub(_expand_currency, text)

        # 4. Decimals (3.14 -> three point one four)
        text = self.decimal_regex.sub(_expand_decimals, text)

        # 5. Ordinals (21st -> twenty-first)
        text = self.ordinal_regex.sub(_expand_ordinals, text)

        # 6. Isolated Integers (1984 -> nineteen eighty-four)
        text = self.integer_regex.sub(_expand_integers, text)

        # 7. Expand ASCII symbols
        for sym, word in _SYMBOLS.items():
            text = text.replace(sym, word)

        # 8. Clean up whitespace and punctuation
        text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text
