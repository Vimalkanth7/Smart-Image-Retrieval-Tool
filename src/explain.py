from typing import List

def explain(query: str, caption: str, keywords: List[str], objects: List[dict]):
    bits = []
    if keywords:
        intersects = [k for k in keywords if k in query.lower()]
        if intersects:
            bits.append(f"matches keywords: {', '.join(intersects)}")
    if caption:
        bits.append(f'caption mentions: "{caption}"')
    return "; ".join(bits) if bits else "semantic similarity in the embedding space"
