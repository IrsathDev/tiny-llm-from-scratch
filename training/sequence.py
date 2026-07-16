"""
TinyLLM Sequence Builder

Phase 11.05.01

Responsibilities
----------------
✓ Format instruction/response
✓ Encode using any tokenizer
✓ Build token sequences

Author: Mohamed Irsath
Project: TinyLLM Framework
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from training.dataset import Sample


# ==========================================================
# Encoded Sample
# ==========================================================

@dataclass
class EncodedSample:
    text: str
    token_ids: List[int]
    length: int


# ==========================================================
# Sequence Builder
# ==========================================================

class SequenceBuilder:

    def __init__(
        self,
        tokenizer,
        template="default",
    ):

        self.tokenizer = tokenizer
        self.template = template

    # ------------------------------------------------------

    def combine(self, sample: Sample) -> str:
        """
        Convert a dataset sample into a single prompt.
        """

        if self.template == "default":

            return (
                f"User: {sample.instruction}\n\n"
                f"Assistant: {sample.response}"
            )

        elif self.template == "chatml":

            return (
                "<|user|>\n"
                f"{sample.instruction}\n"
                "<|assistant|>\n"
                f"{sample.response}"
            )

        else:

            raise ValueError(
                f"Unknown template: {self.template}"
            )

    # ------------------------------------------------------

    def tokenize(self, text: str) -> List[int]:
        """
        Encode text using the tokenizer.
        """

        return self.tokenizer.encode(text)

    # ------------------------------------------------------

    def encode_sample(
        self,
        sample: Sample,
    ) -> EncodedSample:
        """
        Encode one dataset sample.
        """

        text = self.combine(sample)

        token_ids = self.tokenize(text)

        return EncodedSample(
            text=text,
            token_ids=token_ids,
            length=len(token_ids),
        )

    # ------------------------------------------------------

    def statistics(
        self,
        encoded_samples: List[EncodedSample],
    ):

        lengths = [
            x.length
            for x in encoded_samples
        ]

        if not lengths:

            return {}

        return {

            "samples": len(lengths),

            "min_length": min(lengths),

            "max_length": max(lengths),

            "avg_length":

                sum(lengths) / len(lengths),

        }

    # ------------------------------------------------------

    def preview(
        self,
        encoded: EncodedSample,
        max_tokens=40,
    ):

        print("=" * 60)

        print("Prompt")

        print("=" * 60)

        print(encoded.text)

        print()

        print("=" * 60)

        print("Token IDs")

        print("=" * 60)

        print(
            encoded.token_ids[:max_tokens]
        )

        print()

        print(
            "Sequence Length:",
            encoded.length
        )

        print("=" * 60)