#!/usr/bin/env python
"""Real LLM evaluation script for refund grievance categorization.

Usage:
    # Set real Groq key first
    export GROQ_API_KEY=your_key_here
    # Or create .env with GROQ_API_KEY=your_key

    python eval_llm.py                    # Run full evaluation
    python eval_llm.py --dataset custom.json  # Custom test cases
    python eval_llm.py --compare          # Compare stub vs real LLM
    python eval_llm.py --delay 2          # Add 2s delay between requests (rate limit)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure we can import app modules
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.db import close_db, init_db
from app.llm.groq_client import get_llm_client, LLMUnderstanding, understand_grievance
from app.rag.ingest import ingest_policies
from app.rag.retrieval import retrieve


@dataclass
class TestCase:
    grievance_text: str
    expected_category: str | None = None
    expected_amount: float | None = None
    description: str = ""


@dataclass
class EvalResult:
    grievance_text: str
    expected_category: str | None
    predicted_category: str
    expected_amount: float | None
    predicted_amount: float | None
    confidence: float
    category_correct: bool | None
    amount_correct: bool | None
    reasoning: str


# Realistic test cases covering vague/ambiguous grievances
DEFAULT_TEST_CASES = [
    TestCase(
        grievance_text="The screen on my new phone has a hairline crack right out of the box. Never even turned it on.",
        expected_category="defective_product",
        expected_amount=None,
        description="Clear defect, no amount mentioned",
    ),
    TestCase(
        grievance_text="I ordered the red sweater but received a blue one. This is completely wrong.",
        expected_category="not_as_described",
        expected_amount=None,
        description="Wrong color - not as described",
    ),
    TestCase(
        grievance_text="My package was supposed to arrive Friday but it's now Wednesday and still nothing. Tracking shows it's stuck in transit.",
        expected_category="late_delivery",
        expected_amount=None,
        description="Late delivery, no amount",
    ),
    TestCase(
        grievance_text="I was charged $89.99 twice for the same order #ORD-456. My bank shows two identical charges.",
        expected_category="billing_error",
        expected_amount=89.99,
        description="Duplicate charge with specific amount",
    ),
    TestCase(
        grievance_text="I changed my mind about the coffee maker. It's been 10 days, box unopened.",
        expected_category="change_of_mind",
        expected_amount=None,
        description="Change of mind within 14 days",
    ),
    TestCase(
        grievance_text="The laptop says 16GB RAM but system only shows 8GB. Also the charger is missing from the box.",
        expected_category="not_as_described",
        expected_amount=None,
        description="Multiple issues: wrong specs + missing accessory",
    ),
    TestCase(
        grievance_text="Delivery was 3 weeks late. I needed this for a trip and it arrived after I left. Want full refund of $299.",
        expected_category="late_delivery",
        expected_amount=299.0,
        description="Very late delivery with amount",
    ),
    TestCase(
        grievance_text="There's a weird charge of $15.50 on my statement from your store but I didn't buy anything that day.",
        expected_category="billing_error",
        expected_amount=15.50,
        description="Mysterious charge",
    ),
    TestCase(
        grievance_text="The shoes are too tight. I wore them once inside the house. Can I return them? Order was $120.",
        expected_category="change_of_mind",
        expected_amount=120.0,
        description="Change of mind with amount mentioned",
    ),
    TestCase(
        grievance_text="Product is garbage. Fell apart after two uses. Cheap plastic, not the metal described. Paid $45.",
        expected_category="defective_product",
        expected_amount=45.0,
        description="Quality defect + material mismatch + amount",
    ),
    TestCase(
        grievance_text="I never received my order but tracking says delivered. Neighbor says they didn't get it either.",
        expected_category="late_delivery",
        expected_amount=None,
        description="Missing delivery (could be late_delivery or lost)",
    ),
    TestCase(
        grievance_text="The color is off - looks nothing like the website photo. More orange than red. Also shipping took forever.",
        expected_category="not_as_described",
        expected_amount=None,
        description="Color mismatch + late shipping",
    ),
    TestCase(
        grievance_text="My card was charged in EUR instead of USD, and the conversion rate they used was terrible. Lost about $12.",
        expected_category="billing_error",
        expected_amount=12.0,
        description="Currency conversion error",
    ),
    TestCase(
        grievance_text="I don't want this anymore. It's been 3 weeks though. Original price $200.",
        expected_category="change_of_mind",
        expected_amount=200.0,
        description="Change of mind outside 14-day window",
    ),
    TestCase(
        grievance_text="The blender motor burned out on first use. Smoke came out. This is a safety hazard.",
        expected_category="defective_product",
        expected_amount=None,
        description="Safety-critical defect",
    ),
]


def run_evaluation(test_cases: list[TestCase], use_stub: bool = False, delay: float = 0.0) -> list[EvalResult]:
    """Run evaluation on test cases."""
    client = get_llm_client(use_stub=use_stub)

    results = []
    for i, tc in enumerate(test_cases):
        # Retrieve policy context
        chunks = retrieve(tc.grievance_text, k=5)

        # Get LLM understanding
        try:
            understanding: LLMUnderstanding = client.understand(tc.grievance_text, chunks)
        except Exception as e:
            results.append(EvalResult(
                grievance_text=tc.grievance_text,
                expected_category=tc.expected_category,
                predicted_category="ERROR",
                expected_amount=tc.expected_amount,
                predicted_amount=None,
                confidence=0.0,
                category_correct=None,
                amount_correct=None,
                reasoning=f"LLM error: {e}",
            ))
            continue

        category_correct = None
        if tc.expected_category:
            category_correct = (understanding.category == tc.expected_category)

        amount_correct = None
        if tc.expected_amount is not None and understanding.requested_amount is not None:
            # Allow small floating point tolerance
            amount_correct = abs(understanding.requested_amount - tc.expected_amount) < 0.01

        results.append(EvalResult(
            grievance_text=tc.grievance_text,
            expected_category=tc.expected_category,
            predicted_category=understanding.category,
            expected_amount=tc.expected_amount,
            predicted_amount=understanding.requested_amount,
            confidence=understanding.confidence,
            category_correct=category_correct,
            amount_correct=amount_correct,
            reasoning=understanding.reasoning,
        ))

        # Rate limiting delay between requests (skip on last iteration)
        if delay > 0 and i < len(test_cases) - 1:
            print(f"  ⏳ Waiting {delay}s for rate limit...")
            time.sleep(delay)

    return results


def print_results(results: list[EvalResult], label: str = "Evaluation"):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"{label} Results")
    print(f"{'='*60}")

    cat_correct = [r for r in results if r.category_correct is True]
    cat_incorrect = [r for r in results if r.category_correct is False]
    cat_unknown = [r for r in results if r.category_correct is None]

    amt_correct = [r for r in results if r.amount_correct is True]
    amt_incorrect = [r for r in results if r.amount_correct is False]
    amt_unknown = [r for r in results if r.amount_correct is None]

    print(f"\nCategory Accuracy: {len(cat_correct)}/{len(cat_correct)+len(cat_incorrect)} "
          f"({len(cat_correct)/max(1,len(cat_correct)+len(cat_incorrect))*100:.1f}%)")
    print(f"Amount Accuracy:   {len(amt_correct)}/{len(amt_correct)+len(amt_incorrect)} "
          f"({len(amt_correct)/max(1,len(amt_correct)+len(amt_incorrect))*100:.1f}%)")
    print(f"Avg Confidence:    {sum(r.confidence for r in results)/len(results):.2f}")

    if cat_incorrect:
        print(f"\n❌ Category Mismatches ({len(cat_incorrect)}):")
        for r in cat_incorrect:
            print(f"  - '{r.grievance_text[:60]}...'")
            print(f"    Expected: {r.expected_category}, Got: {r.predicted_category} (conf={r.confidence:.2f})")
            print(f"    Reasoning: {r.reasoning}")

    if amt_incorrect:
        print(f"\n❌ Amount Mismatches ({len(amt_incorrect)}):")
        for r in amt_incorrect:
            print(f"  - '{r.grievance_text[:60]}...'")
            print(f"    Expected: ${r.expected_amount}, Got: ${r.predicted_amount}")

    print(f"\n✅ Correct Categories ({len(cat_correct)}):")
    for r in cat_correct:
        print(f"  - '{r.grievance_text[:60]}...' → {r.predicted_category} (conf={r.confidence:.2f})")


def load_custom_dataset(path: str) -> list[TestCase]:
    """Load test cases from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [TestCase(**item) for item in data]


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM grievance understanding")
    parser.add_argument("--dataset", help="Path to JSON test dataset")
    parser.add_argument("--compare", action="store_true", help="Compare stub vs real LLM")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay in seconds between API requests (rate limiting)")
    args = parser.parse_args()

    # Initialize DB and ingest policies
    print("Initializing database...")
    init_db()
    print("Ingesting policies...")
    n = ingest_policies()
    print(f"Ingested {n} policy chunks")

    # Load test cases
    if args.dataset:
        test_cases = load_custom_dataset(args.dataset)
        print(f"Loaded {len(test_cases)} test cases from {args.dataset}")
    else:
        test_cases = DEFAULT_TEST_CASES
        print(f"Using {len(test_cases)} default test cases")

    if args.compare:
        print("\n🔄 Running STUB evaluation...")
        stub_results = run_evaluation(test_cases, use_stub=True)
        print_results(stub_results, "STUB")

        print("\n🔄 Running REAL LLM evaluation...")
        real_results = run_evaluation(test_cases, use_stub=False, delay=args.delay)
        print_results(real_results, "REAL LLM")

        if args.output:
            output = {
                "stub": [asdict(r) for r in stub_results],
                "real": [asdict(r) for r in real_results],
            }
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        # Check if real LLM is available
        use_stub = not (settings.groq_api_key or os.environ.get("GROQ_API_KEY"))
        mode = "STUB" if use_stub else "REAL LLM"
        print(f"\n🔄 Running {mode} evaluation...")
        results = run_evaluation(test_cases, use_stub=use_stub, delay=args.delay)
        print_results(results, mode)

        if args.output:
            with open(args.output, "w") as f:
                json.dump([asdict(r) for r in results], f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()