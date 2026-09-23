"""
Profile the Marathi TTS training data before launching real fine-tuning.

This script intentionally profiles the existing BodhanTrainingDataset rather
than duplicating preprocessing/tokenization logic.

It reports:
- prompt token lengths
- serialized speech-token lengths
- total sequence lengths
- truncation behavior under candidate max_seq_length values

No model training is performed.
"""

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from marathi_tts.config import TrainingConfig
from marathi_tts.training import BodhanTrainingDataset


logger = logging.getLogger("profile_training_data")


def percentile(values, p):
    """Compute a percentile without requiring NumPy."""

    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return float(values[0])

    rank = (len(values) - 1) * (p / 100.0)

    lower = int(rank)
    upper = min(lower + 1, len(values))

    fraction = rank - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * fraction
    )


def summarize(values):
    """Return descriptive statistics for a numeric list."""

    if not values:
        return {
            "count": 0,
            "min": 0,
            "mean": 0,
            "median": 0,
            "p90": 0,
            "p95": 0,
            "max": 0,
        }

    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def format_stats(name, stats):
    """Format one statistics block."""

    print(f"\n{name}")
    print("-" * 60)

    print(f"  Count:       {stats['count']}")
    print(f"  Min:         {stats['min']:.2f}")
    print(f"  Mean:        {stats['mean']:.2f}")
    print(f"  Median:      {stats['median']:.2f}")
    print(f"  P90:         {stats['p90']:.2f}")
    print(f"  P95:         {stats['p95']:.2f}")
    print(f"  Max:         {stats['max']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile Bodhan Marathi TTS training sequence lengths."
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=25,
        help="Number of dataset examples to profile.",
    )

    parser.add_argument(
        "--max-text-length",
        type=int,
        default=512,
        help="Maximum prompt token length.",
    )

    parser.add_argument(
        "--max-audio-tokens",
        type=int,
        default=2048,
        help="Maximum serialized speech tokens before sequence limiting.",
    )

    parser.add_argument(
        "--candidate-lengths",
        type=int,
        nargs="+",
        default=[512, 768, 1024, 1536, 2048, 2600],
        help="Candidate sequence lengths to evaluate.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/data_profile.json",
        help="Path for JSON profile output.",
    )

    args = parser.parse_args()

    if args.max_samples <= 0:
        raise ValueError("--max-samples must be greater than 0.")

    if args.max_text_length <= 0:
        raise ValueError("--max-text-length must be greater than 0.")

    if args.max_audio_tokens <= 0:
        raise ValueError("--max-audio-tokens must be greater than 0.")

    if not args.candidate_lengths:
        raise ValueError("At least one candidate sequence length is required.")

    if any(length <= 0 for length in args.candidate_lengths):
        raise ValueError(
            "All candidate sequence lengths must be greater than 0."
        )

    print()
    print("=" * 80)
    print("BODHAN MARATHI TTS — TRAINING DATA PROFILE")
    print("=" * 80)
    print()

    print("Configuration")
    print("-" * 60)
    print(f"  Dataset:             SPRINGLab/IndicTTS_Marathi")
    print(f"  Samples requested:   {args.max_samples}")
    print(f"  Max text tokens:     {args.max_text_length}")
    print(f"  Max audio tokens:    {args.max_audio_tokens}")
    print(
        f"  Candidate lengths:   "
        f"{', '.join(map(str, args.candidate_lengths))}"
    )

    print()
    print("Creating dataset...")
    print()

    config = TrainingConfig(
        max_samples=args.max_samples,
        max_text_length=args.max_text_length,
        max_audio_tokens=args.max_audio_tokens,
        # Use the largest requested candidate here so that the profiler can
        # observe the natural sequence lengths without an unnecessarily small
        # training limit.
        max_seq_length=max(args.candidate_lengths),
    )

    start_time = time.time()

    dataset = BodhanTrainingDataset(
        config=config,
    )

    load_time = time.time() - start_time

    print(
        f"Dataset ready: {len(dataset)} examples "
        f"({load_time:.1f}s)"
    )

    prompt_lengths = []
    speech_lengths = []
    total_lengths = []

    examples = []

    print()
    print("Processing examples...")
    print()

    for idx in range(len(dataset)):
        example_start = time.time()

        try:
            sample = dataset[idx]

            text_length = int(sample["text_length"])
            speech_length = int(sample["speech_length"])
            total_length = int(sample["total_length"])

            prompt_lengths.append(text_length)
            speech_lengths.append(speech_length)
            total_lengths.append(total_length)

            examples.append(
                {
                    "index": idx,
                    "text_length": text_length,
                    "speech_length": speech_length,
                    "total_length": total_length,
                }
            )

            elapsed = time.time() - example_start

            print(
                f"  [{idx + 1:>3}/{len(dataset)}] "
                f"text={text_length:>4} "
                f"speech={speech_length:>4} "
                f"total={total_length:>4} "
                f"({elapsed:.1f}s)"
            )

        except Exception as exc:
            print(
                f"  [{idx + 1:>3}/{len(dataset)}] ERROR: {exc}"
            )

    if not total_lengths:
        raise RuntimeError(
            "No examples could be processed successfully."
        )

    prompt_stats = summarize(prompt_lengths)
    speech_stats = summarize(speech_lengths)
    total_stats = summarize(total_lengths)

    print()
    print("=" * 80)
    print("SEQUENCE LENGTH STATISTICS")
    print("=" * 80)

    format_stats(
        "Prompt / text tokens",
        prompt_stats,
    )

    format_stats(
        "Serialized speech tokens",
        speech_stats,
    )

    format_stats(
        "Total training sequence",
        total_stats,
    )

    print()
    print("=" * 80)
    print("CANDIDATE MAX SEQUENCE LENGTH ANALYSIS")
    print("=" * 80)

    candidate_results = []

    for max_length in args.candidate_lengths:

        truncated_count = 0
        retained_count = 0
        retained_speech_tokens = []
        original_speech_tokens = []

        for example in examples:
            original_total = example["total_length"]
            original_speech = example["speech_length"]
            text_length = example["text_length"]

            # The actual number of speech tokens that can fit into this
            # candidate sequence length is:
            #
            # max_length
            # - prompt
            # - start_of_speech
            # - end_of_speech
            #
            available_speech = max(
                0,
                max_length - text_length - 2,
            )

            effective_speech = min(
                original_speech,
                args.max_audio_tokens,
                available_speech,
            )

            effective_total = (
                text_length
                + 1
                + effective_speech
                + 1
            )

            original_speech_tokens.append(original_speech)
            retained_speech_tokens.append(effective_speech)

            if effective_total < original_total:
                truncated_count += 1
            else:
                retained_count += 1

        sample_count = len(examples)

        truncation_percentage = (
            100.0 * truncated_count / sample_count
            if sample_count
            else 0.0
        )

        average_original_speech = statistics.mean(
            original_speech_tokens
        )

        average_retained_speech = statistics.mean(
            retained_speech_tokens
        )

        result = {
            "max_seq_length": max_length,
            "samples": sample_count,
            "fully_retained": retained_count,
            "truncated": truncated_count,
            "truncation_percentage": truncation_percentage,
            "average_original_speech_tokens": average_original_speech,
            "average_retained_speech_tokens": average_retained_speech,
        }

        candidate_results.append(result)

        print()
        print(f"max_seq_length = {max_length}")
        print("-" * 60)
        print(
            f"  Fully retained:          "
            f"{retained_count}/{sample_count}"
        )
        print(
            f"  Truncated:               "
            f"{truncated_count}/{sample_count}"
        )
        print(
            f"  Truncation percentage:   "
            f"{truncation_percentage:.2f}%"
        )
        print(
            f"  Avg original speech:     "
            f"{average_original_speech:.2f}"
        )
        print(
            f"  Avg retained speech:     "
            f"{average_retained_speech:.2f}"
        )

    # Find the smallest candidate that retains all observed examples.
    no_truncation_candidates = [
        result
        for result in candidate_results
        if result["truncated"] == 0
    ]

    recommended_candidate = None

    if no_truncation_candidates:
        recommended_candidate = min(
            no_truncation_candidates,
            key=lambda x: x["max_seq_length"],
        )

    print()
    print("=" * 80)
    print("PROFILE CONCLUSION")
    print("=" * 80)

    if recommended_candidate:
        print()
        print(
            "Smallest tested sequence length with zero truncation "
            "on this sample:"
        )
        print()
        print(
            f"  {recommended_candidate['max_seq_length']} tokens"
        )
        print()
        print(
            "This is an empirical observation for the profiled samples, "
            "not a claim about the entire dataset."
        )
    else:
        print()
        print(
            "None of the tested sequence lengths retained every profiled "
            "sample."
        )
        print(
            "Use the truncation percentages above to choose the next "
            "candidate."
        )

    elapsed_total = time.time() - start_time

    profile = {
        "dataset": config.dataset_name,
        "split": config.dataset_split,
        "requested_samples": args.max_samples,
        "processed_samples": len(examples),
        "max_text_length": args.max_text_length,
        "max_audio_tokens": args.max_audio_tokens,
        "candidate_lengths": args.candidate_lengths,
        "load_and_profile_seconds": elapsed_total,
        "statistics": {
            "prompt_tokens": prompt_stats,
            "speech_tokens": speech_stats,
            "total_sequence_tokens": total_stats,
        },
        "candidate_analysis": candidate_results,
        "examples": examples,
        "recommended_candidate_without_observed_truncation": (
            recommended_candidate["max_seq_length"]
            if recommended_candidate
            else None
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            profile,
            f,
            indent=2,
        )

    print()
    print(f"Profile saved to: {output_path}")
    print()
    print("=" * 80)
    print("DATA PROFILE COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()