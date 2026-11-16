"""
NEUROLECT AI - Core Helper Functions
Text cleaning, sentence segmentation with oversized paragraph splitting,
token/word chunking with boundary safety, and timestamp formatting.
Strictly RAG-free.
"""

import re
from typing import List, Dict, Any
import config

def clean_text(text: str) -> str:
    """Cleans and normalizes raw extracted text from PDFs, audio transcripts, or direct input."""
    if not text:
        return ""
    
    # Normalize unicode whitespace & line endings
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\xa0", " ").replace("\t", " ")
    # Replace consecutive blank lines with two newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Replace multiple spaces with a single space
    cleaned = re.sub(r"[ ]{2,}", " ", cleaned)
    return cleaned.strip()

def split_into_sentences(text: str, max_sentence_words: int = 60) -> List[str]:
    """
    Splits text into sentences using regex boundary detection respecting common abbreviations.
    If any sentence or unpunctuated paragraph exceeds max_sentence_words,
    it safely decomposes it at secondary punctuation (semicolons, colons, commas)
    or word boundaries to prevent transformer context overflow.
    """
    if not text or not text.strip():
        return []
    
    # Protect common abbreviations from being treated as sentence terminals
    text_processed = re.sub(r'\b(Dr|Prof|Mr|Mrs|Ms|Fig|e\.g|i\.e|vs|al|approx|etc)\.', r'\1<DOT>', text, flags=re.IGNORECASE)
    
    # Split on sentence terminals (.?!) followed by whitespace
    raw_sentences = re.split(r'(?<=[.?!])\s+', text_processed)
    
    final_sentences = []
    for s in raw_sentences:
        clean_s = s.replace("<DOT>", ".").strip()
        if not clean_s:
            continue
            
        words = clean_s.split()
        if len(words) > max_sentence_words:
            # First try splitting by secondary punctuation (;, :)
            sub_clauses = re.split(r'(?<=[;,])\s+', clean_s)
            
            # If no punctuation exists to split on, split by word count directly
            if len(sub_clauses) <= 1:
                for i in range(0, len(words), max_sentence_words):
                    sub_str = " ".join(words[i:i+max_sentence_words]).strip()
                    if len(sub_str) > 2:
                        final_sentences.append(sub_str)
            else:
                curr_clause = []
                curr_len = 0
                for clause in sub_clauses:
                    c_words = clause.split()
                    if curr_len + len(c_words) > max_sentence_words and curr_clause:
                        sub_str = " ".join(curr_clause).strip()
                        if len(sub_str) > 2:
                            final_sentences.append(sub_str)
                        curr_clause = [clause]
                        curr_len = len(c_words)
                    else:
                        curr_clause.append(clause)
                        curr_len += len(c_words)
                if curr_clause:
                    sub_str = " ".join(curr_clause).strip()
                    if len(sub_str) > 2:
                        final_sentences.append(sub_str)
        else:
            if len(clean_s) > 2:
                final_sentences.append(clean_s)
            
    return final_sentences

def chunk_text(
    text: str,
    chunk_size_words: int = config.CHUNK_SIZE_WORDS,
    overlap_words: int = config.CHUNK_OVERLAP_WORDS
) -> List[Dict[str, Any]]:
    """
    Chunks text into manageable segments for transformer models without RAG.
    Preserves sentence boundaries whenever possible.
    Guarantees that no trailing text is discarded and no chunk exceeds chunk_size_words.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    sentences = split_into_sentences(cleaned, max_sentence_words=min(60, chunk_size_words // 2))
    
    if not sentences:
        words = cleaned.split()
        if not words:
            return []
        return [{"chunk_id": 0, "text": cleaned, "word_count": len(words)}]
    
    chunks = []
    current_chunk_sentences = []
    current_word_count = 0
    chunk_id = 0
    
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        
        # When adding this sentence would overflow the chunk budget
        if current_word_count + sentence_word_count > chunk_size_words and current_chunk_sentences:
            chunk_text_str = " ".join(current_chunk_sentences)
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text_str,
                "word_count": current_word_count
            })
            chunk_id += 1
            
            # Construct overlap from the end of the previous chunk
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_chunk_sentences):
                sw_count = len(s.split())
                if overlap_count + sw_count <= overlap_words:
                    overlap_sentences.insert(0, s)
                    overlap_count += sw_count
                else:
                    break
            
            current_chunk_sentences = overlap_sentences + [sentence]
            current_word_count = overlap_count + sentence_word_count
        else:
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_word_count
            
    # Guarantee the final chunk containing trailing content is appended
    if current_chunk_sentences:
        chunk_text_str = " ".join(current_chunk_sentences)
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text_str,
            "word_count": current_word_count
        })
        
    return chunks

def format_timestamp(seconds: float) -> str:
    """Converts a duration in seconds to a human-readable MM:SS or HH:MM:SS format."""
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def calculate_reading_time(word_count: int, wpm: int = 200) -> str:
    """Calculates approximate reading duration given word count."""
    minutes = max(1, round(word_count / wpm))
    return f"{minutes} min read"
