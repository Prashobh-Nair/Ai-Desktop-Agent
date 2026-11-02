"""
Text Preprocessor and Normalizer for Kokoro-82M TTS.
Handles:
1. Markdown stripping & LLM output sanitization.
2. Number, currency, date, time, and percentage expansion.
3. Acronym and abbreviation normalization.
4. Smart sentence splitting and punctuation handling for long LLM responses.
5. Input validation and edge-case handling.
"""

import re
from typing import List, Tuple, Optional


class TextProcessor:
    """
    Normalizes raw LLM output text for high-fidelity speech synthesis.
    Converts numbers, dates, currency, abbreviations, and markdown into natural spoken English.
    """

    # Numbers 0-19
    ONES = [
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"
    ]
    
    # Tens 20-90
    TENS = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
    ]

    # Ordinals 1st - 19th
    ORDINALS = {
        1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
        6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
        11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
        15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
        19: "nineteenth", 20: "twentieth", 30: "thirtieth"
    }

    # Common abbreviations and technical acronyms spoken in desktop assistants
    ABBREVIATIONS = {
        r"\bAI\b": "A I",
        r"\bLLM\b": "L L M",
        r"\bSTT\b": "S T T",
        r"\bTTS\b": "T T S",
        r"\bAPI\b": "A P I",
        r"\bAPIs\b": "A P I's",
        r"\bCPU\b": "C P U",
        r"\bGPU\b": "G P U",
        r"\bRAM\b": "R A M",
        r"\bROM\b": "R O M",
        r"\bOS\b": "O S",
        r"\bUI\b": "U I",
        r"\bUX\b": "U X",
        r"\bCLI\b": "C L I",
        r"\bURL\b": "U R L",
        r"\bHTML\b": "H T M L",
        r"\bCSS\b": "C S S",
        r"\bSQL\b": "sequel",
        r"\bJSON\b": "jason",
        r"\bWAV\b": "wave",
        r"\bMP3\b": "M P 3",
        r"\bPDF\b": "P D F",
        r"\bUSB\b": "U S B",
        r"\bWiFi\b": "wi-fi",
        r"\bPC\b": "P C",
        r"\bID\b": "I D",
        r"\bDr\.\b": "Doctor",
        r"\bMr\.\b": "Mister",
        r"\bMrs\.\b": "Missus",
        r"\bMs\.\b": "Miss",
        r"\bProf\.\b": "Professor",
        r"\betc\.\b": "et cetera",
        r"\be\.g\.\b": "for example",
        r"\bi\.e\.\b": "that is",
        r"\bvs\.\b": "versus",
        r"\bdept\.\b": "department",
        r"\bapt\.\b": "apartment",
        r"\bave\.\b": "avenue",
        r"\bst\.\b": "street",
        r"\bmin\.\b": "minutes",
        r"\bsec\.\b": "seconds",
        r"\bapprox\.\b": "approximately",
        r"\bw/\b": "with",
        r"\bw/o\b": "without",
    }

    # Units of measurement
    UNITS = {
        r"(\d+)\s*GB\b": r"\1 gigabytes",
        r"(\d+)\s*MB\b": r"\1 megabytes",
        r"(\d+)\s*KB\b": r"\1 kilobytes",
        r"(\d+)\s*TB\b": r"\1 terabytes",
        r"(\d+)\s*GHz\b": r"\1 gigahertz",
        r"(\d+)\s*MHz\b": r"\1 megahertz",
        r"(\d+)\s*Hz\b": r"\1 hertz",
        r"(\d+)\s*ms\b": r"\1 milliseconds",
        r"(\d+)\s*km/h\b": r"\1 kilometers per hour",
        r"(\d+)\s*mph\b": r"\1 miles per hour",
        r"(\d+)\s*kg\b": r"\1 kilograms",
        r"(\d+)\s*cm\b": r"\1 centimeters",
        r"(\d+)\s*mm\b": r"\1 millimeters",
        r"(\d+)\s*°C\b": r"\1 degrees Celsius",
        r"(\d+)\s*°F\b": r"\1 degrees Fahrenheit",
    }

    def __init__(self, max_chunk_chars: int = 250, max_chunk_words: int = 40):
        self.max_chunk_chars = max_chunk_chars
        self.max_chunk_words = max_chunk_words

    @classmethod
    def int_to_words(cls, n: int) -> str:
        """Converts an integer (up to 999,999,999) to English words."""
        if n < 0:
            return "minus " + cls.int_to_words(-n)
        if n == 0:
            return "zero"

        def _convert_below_thousand(num: int) -> str:
            words = []
            if num >= 100:
                words.append(cls.ONES[num // 100] + " hundred")
                num %= 100
                if num > 0:
                    words.append("and")
            if num >= 20:
                t = cls.TENS[num // 10]
                o = cls.ONES[num % 10]
                words.append(f"{t}-{o}" if o else t)
            elif num > 0:
                words.append(cls.ONES[num])
            return " ".join(words)

        parts = []
        if n >= 1_000_000:
            millions = n // 1_000_000
            parts.append(_convert_below_thousand(millions) + " million")
            n %= 1_000_000

        if n >= 1_000:
            thousands = n // 1_000
            parts.append(_convert_below_thousand(thousands) + " thousand")
            n %= 1_000

        if n > 0:
            parts.append(_convert_below_thousand(n))

        return " ".join(parts).strip()

    @classmethod
    def _expand_ordinal(cls, match: re.Match) -> str:
        num_str = match.group(1)
        try:
            num = int(num_str)
            if num in cls.ORDINALS:
                return cls.ORDINALS[num]
            if num % 100 in cls.ORDINALS:
                tens_prefix = cls.int_to_words((num // 100) * 100)
                return f"{tens_prefix} and {cls.ORDINALS[num % 100]}"
            cardinal = cls.int_to_words(num)
            if cardinal.endswith("y"):
                return cardinal[:-1] + "ieth"
            return cardinal + "th"
        except Exception:
            return match.group(0)

    @classmethod
    def _expand_currency(cls, match: re.Match) -> str:
        symbol = match.group(1)
        amount_str = match.group(2).replace(",", "")
        
        currencies = {
            "$": ("dollar", "dollars", "cent", "cents"),
            "£": ("pound", "pounds", "penny", "pence"),
            "€": ("euro", "euros", "cent", "cents"),
            "₹": ("rupee", "rupees", "paise", "paise")
        }
        
        curr = currencies.get(symbol, ("dollar", "dollars", "cent", "cents"))
        
        try:
            if "." in amount_str:
                major_part, minor_part = amount_str.split(".", 1)
                major = int(major_part) if major_part else 0
                minor = int(minor_part[:2]) if minor_part else 0
                
                major_words = cls.int_to_words(major)
                major_unit = curr[0] if major == 1 else curr[1]
                
                if minor > 0:
                    minor_words = cls.int_to_words(minor)
                    minor_unit = curr[2] if minor == 1 else curr[3]
                    return f"{major_words} {major_unit} and {minor_words} {minor_unit}"
                return f"{major_words} {major_unit}"
            else:
                val = int(amount_str)
                words = cls.int_to_words(val)
                unit = curr[0] if val == 1 else curr[1]
                return f"{words} {unit}"
        except Exception:
            return match.group(0)

    @classmethod
    def _expand_percentage(cls, match: re.Match) -> str:
        val_str = match.group(1)
        try:
            if "." in val_str:
                whole, dec = val_str.split(".", 1)
                dec_words = " ".join(cls.ONES[int(d)] if d.isdigit() else d for d in dec)
                return f"{cls.int_to_words(int(whole))} point {dec_words} percent"
            return f"{cls.int_to_words(int(val_str))} percent"
        except Exception:
            return f"{val_str} percent"

    @classmethod
    def _expand_decimal(cls, match: re.Match) -> str:
        whole = match.group(1)
        dec = match.group(2)
        try:
            whole_words = cls.int_to_words(int(whole))
            dec_words = " ".join(cls.ONES[int(d)] if d.isdigit() else d for d in dec)
            return f"{whole_words} point {dec_words}"
        except Exception:
            return match.group(0)

    @classmethod
    def _expand_cardinal(cls, match: re.Match) -> str:
        num_str = match.group(0).replace(",", "")
        try:
            val = int(num_str)
            # Four digit years between 1900 and 2099
            if 1900 <= val <= 2099:
                century = val // 100
                remainder = val % 100
                if remainder == 0:
                    return f"{cls.int_to_words(century)} hundred"
                elif remainder < 10:
                    return f"{cls.int_to_words(century)} oh {cls.int_to_words(remainder)}"
                else:
                    return f"{cls.int_to_words(century)} {cls.int_to_words(remainder)}"
            return cls.int_to_words(val)
        except Exception:
            return match.group(0)

    @classmethod
    def _expand_time(cls, match: re.Match) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))
        period = match.group(3) if match.group(3) else ""
        
        h_words = cls.int_to_words(hour)
        if minute == 0:
            m_words = "o'clock" if not period else ""
        elif minute < 10:
            m_words = f"oh {cls.int_to_words(minute)}"
        else:
            m_words = cls.int_to_words(minute)
            
        period_str = " ".join(period.upper()) if period else ""
        parts = [p for p in (h_words, m_words, period_str) if p]
        return " ".join(parts)

    def clean_markdown(self, text: str) -> str:
        """
        Removes Markdown formatting commonly produced by LLMs (bold, italics, code, headers, links).
        """
        if not text:
            return ""
            
        # Code blocks: ```python ... ``` -> replace with spoken notice
        text = re.sub(r"```[\w]*\n(.*?)```", r"Here is the code: \1", text, flags=re.DOTALL)
        
        # Inline code: `variable` -> variable
        text = re.sub(r"`([^`]+)`", r"\1", text)
        
        # Markdown links: [Title](url) -> Title
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        
        # Markdown headers: # Header -> Header
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        
        # Bold and italics: **text** or *text* or __text__ or _text_
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        
        # Blockquotes: > quote -> quote
        text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
        
        # Unordered bullet lists: - item or * item -> item
        text = re.sub(r"^[\*\-\+]\s+", "", text, flags=re.MULTILINE)
        
        # Ordered lists: 1. item -> item
        text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
        
        # Horizontal rules
        text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        
        return text

    def normalize(self, text: str) -> str:
        """
        Full text normalization pipeline:
        1. Strips markdown.
        2. Normalizes symbols, currencies, units, times, dates, and numbers.
        3. Expands common tech abbreviations.
        4. Cleans extraneous whitespace and punctuation.
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Markdown removal
        text = self.clean_markdown(text)

        # Step 2: Time patterns (e.g., 10:30 AM, 3:45pm, 12:00)
        text = re.sub(r"\b(\d{1,2}):(\d{2})\s*(am|pm|AM|PM)?\b", self._expand_time, text)

        # Step 3: Currency patterns (e.g., $10, $5.99, £20, €15.50, ₹1000)
        text = re.sub(r"([$£€₹])\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)", self._expand_currency, text)

        # Step 4: Unit patterns (e.g., 16 GB, 2.4 GHz, 25°C)
        for pattern, replacement in self.UNITS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Step 5: Percentage patterns (e.g., 99.5%, 15%)
        text = re.sub(r"(\d+(?:\.\d+)?)\s*%", self._expand_percentage, text)

        # Step 6: Ordinal numbers (e.g., 1st, 2nd, 3rd, 21st)
        text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", self._expand_ordinal, text, flags=re.IGNORECASE)

        # Step 7: Decimals (e.g., 3.14, 0.05)
        text = re.sub(r"\b(\d+)\.(\d+)\b", self._expand_decimal, text)

        # Step 8: Standard numbers (e.g., 42, 1,000, 2026)
        text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b", self._expand_cardinal, text)

        # Step 9: Abbreviations and acronyms
        for pattern, replacement in self.ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text)

        # Step 10: Symbols
        text = text.replace("&", " and ")
        text = text.replace("+", " plus ")
        text = text.replace("=", " equals ")
        text = text.replace("@", " at ")
        text = text.replace("#", " number ")
        text = text.replace("/", " slash ")
        text = text.replace("\\", " backslash ")
        text = text.replace("~", " approximately ")

        # Step 11: Clean repeated punctuation and excessive whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\.{2,}", "...", text)
        text = re.sub(r"([?!,;])\1+", r"\1", text)
        text = text.strip()

        return text

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Splits text into natural sentences using robust boundary detection.
        Respects abbreviations and avoids accidental splits on decimals or titles.
        """
        if not text:
            return []

        # Split on sentence terminals (. ! ?) followed by whitespace or end of string
        # Using lookbehind to ensure we keep the punctuation with the sentence
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        
        sentences = []
        for s in raw_sentences:
            s = s.strip()
            if s:
                sentences.append(s)
                
        return sentences

    def chunk_text(self, text: str) -> List[str]:
        """
        Breaks long text (such as long LLM responses) into manageable chunks
        suitable for the Kokoro neural TTS model.
        
        Each chunk is guaranteed to respect sentence boundaries or clause boundaries,
        keeping chunks within optimal character and word limits to minimize latency.
        """
        normalized_text = self.normalize(text)
        if not normalized_text:
            return []

        sentences = self.split_into_sentences(normalized_text)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_char_len = 0
        current_word_len = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())
            sentence_chars = len(sentence)

            # If a single sentence is itself too long, split it at commas / semicolons / clauses
            if sentence_chars > self.max_chunk_chars or sentence_words > self.max_chunk_words:
                sub_clauses = re.split(r"(?<=[,;:])\s+", sentence)
                for clause in sub_clauses:
                    clause = clause.strip()
                    if not clause:
                        continue
                    clause_words = len(clause.split())
                    clause_chars = len(clause)
                    
                    if current_char_len + clause_chars > self.max_chunk_chars or current_word_len + clause_words > self.max_chunk_words:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk).strip())
                            current_chunk = []
                            current_char_len = 0
                            current_word_len = 0
                    current_chunk.append(clause)
                    current_char_len += clause_chars + 1
                    current_word_len += clause_words
            else:
                # Standard sentence packing
                if current_char_len + sentence_chars > self.max_chunk_chars or current_word_len + sentence_words > self.max_chunk_words:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk).strip())
                        current_chunk = []
                        current_char_len = 0
                        current_word_len = 0
                        
                current_chunk.append(sentence)
                current_char_len += sentence_chars + 1
                current_word_len += sentence_words

        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())

        return [c for c in chunks if c]

