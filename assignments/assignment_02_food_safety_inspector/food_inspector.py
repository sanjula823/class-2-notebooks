"""
Assignment 2: AI Food Safety Inspector
Zero-Shot Prompting with Structured Outputs

Your mission: Analyze restaurant reviews and complaints to detect health violations
using only clear instructions — no training examples needed!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ViolationCategory(Enum):
    TEMPERATURE_CONTROL = "Food Temperature Control"
    PERSONAL_HYGIENE = "Personal Hygiene"
    PEST_CONTROL = "Pest Control"
    CROSS_CONTAMINATION = "Cross Contamination"
    FACILITY_MAINTENANCE = "Facility Maintenance"
    UNKNOWN = "Unknown"


class SeverityLevel(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class InspectionPriority(Enum):
    URGENT = "URGENT"
    HIGH = "HIGH"
    ROUTINE = "ROUTINE"
    LOW = "LOW"


@dataclass
class Violation:
    """Structured violation data"""

    category: str
    description: str
    severity: str
    evidence: str
    confidence: float


@dataclass
class InspectionReport:
    """Complete inspection analysis"""

    restaurant_name: str
    overall_risk_score: int
    violations: List[Violation]
    inspection_priority: str
    recommended_actions: List[str]
    follow_up_required: bool


class FoodSafetyInspector:
    """
    AI-powered food safety analyzer using zero-shot structured prompting.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """Initialize with LLM for consistent violation detection."""
        # TODO: Initialize an LLM for consistent JSON-style outputs
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        
        self.analysis_chain = None
        self.risk_chain = None
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Create zero-shot prompts for violation detection and risk assessment.

        Create TWO chains:
        1. analysis_chain: Detects violations and extracts evidence
        2. risk_chain: Calculates risk scores based on violations

        Requirements:
        - Must output valid JSON
        - Include all violation categories
        - Extract specific evidence quotes
        - Generate confidence scores
        """

        # TODO: Create violation detection prompt (as a raw template string)
        analysis_template_str = (
    "You are a food safety inspector AI.\n"
    "Analyze the text and identify possible food safety violations.\n\n"
    "Categories:\n"
    "- Food Temperature Control\n"
    "- Personal Hygiene\n"
    "- Pest Control\n"
    "- Cross Contamination\n"
    "- Facility Maintenance\n"
    "- Unknown\n\n"
    "Return VALID JSON in this exact format:\n"
    "{\n"
    '  "violations": [\n'
    "    {\n"
    '      "category": "...",\n'
    '      "description": "...",\n'
    '      "severity": "Critical|High|Medium|Low",\n'
    '      "evidence": "exact quote",\n'
    '      "confidence": 0.0\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "If no violations, return an empty list.\n\n"
    "Text: {review_text}"
)


        # TODO: Create risk assessment prompt (as a raw template string)
        risk_template_str = (
    "You are assessing food safety risk.\n\n"
    "Given violations JSON, calculate:\n"
    "- risk_score (0–100)\n"
    "- priority (URGENT, HIGH, ROUTINE, LOW)\n\n"
    "Rules:\n"
    "- Critical adds 40\n"
    "- High adds 25\n"
    "- Medium adds 15\n"
    "- Low adds 5\n\n"
    "Return ONLY JSON:\n"
    "{ \"risk_score\": number, \"priority\": \"...\" }\n\n"
    "Violations: {violations}"
)


        # TODO: Build PromptTemplate objects from the strings above
        analysis_template = PromptTemplate.from_template(analysis_template_str)
        risk_template = PromptTemplate.from_template(risk_template_str)
        # TODO: Set up the chains
        self.analysis_chain = analysis_template | self.llm | StrOutputParser()
        self.risk_chain = risk_template | self.llm | StrOutputParser()
        self.analysis_chain = None
        self.risk_chain = None

    def detect_violations(self, text: str) -> List[Violation]:
        try:
           raw = self.analysis_chain.invoke({"review_text": text})
           data = json.loads(raw)

           violations: List[Violation] = []
           for v in data.get("violations", []):
               violations.append(
                   Violation(
                       category=v.get("category", "Unknown"),
                       description=v.get("description", ""),
                       severity=v.get("severity", "Low"),
                       evidence=v.get("evidence", ""),
                       confidence=float(v.get("confidence", 0.5)),
                   )
              )
           return violations

        except Exception as e:
            print(f"Error detecting violations: {e}")
            return []


    def calculate_risk_score(self, violations: List[Violation]) -> Tuple[int, str]:
        score = 0
        for v in violations:
            if v.severity == "Critical":
                score += 40
            elif v.severity == "High":
                score += 25
            elif v.severity == "Medium":
                score += 15
            elif v.severity == "Low":
                score += 5

        score = min(score, 100)

        if score >= 70:
            priority = InspectionPriority.URGENT.value
        elif score >= 40:
            priority = InspectionPriority.HIGH.value
        elif score >= 15:
            priority = InspectionPriority.ROUTINE.value
        else:
            priority = InspectionPriority.LOW.value

        return score, priority


    def analyze_review(
        self, text: str, restaurant_name: str = "Unknown"
    ) -> InspectionReport:
        """
        TODO #4: Complete analysis pipeline for a single review.

        Args:
            text: Review text to analyze
            restaurant_name: Name of the restaurant

        Returns:
            Complete InspectionReport with all findings
        """

        # TODO: Implement full analysis pipeline
        # 1. Detect violations
        # 2. Calculate risk score
        # 3. Generate recommendations
        # 4. Create InspectionReport

        violations = self.detect_violations(text)
        risk_score, priority = self.calculate_risk_score(violations)

        actions = []
        if priority in ("URGENT", "HIGH"):
            actions.append("Schedule immediate inspection")
        if violations:
            actions.append("Review food safety procedures")

        return InspectionReport(
    restaurant_name=restaurant_name,
    overall_risk_score=risk_score,
    violations=violations,
    inspection_priority=priority,
    recommended_actions=actions,
    follow_up_required=priority in ("URGENT", "HIGH"),
)
 

    def batch_analyze(self, reviews: List[Dict[str, str]]) -> InspectionReport:
        all_violations: List[Violation] = []

    # Analyze each review individually
        for review in reviews:
            text = review.get("text", "")
            violations = self.detect_violations(text)
            all_violations.extend(violations)

    # Remove duplicate violations (category + evidence)
        unique = {}
        for v in all_violations:
            key = (v.category, v.evidence)
            if key not in unique or v.confidence > unique[key].confidence:
                unique[key] = v

        final_violations = list(unique.values())

    # Calculate overall risk
        risk_score, priority = self.calculate_risk_score(final_violations)

        recommended_actions = [
        "Conduct a full health inspection",
        "Address repeated customer complaints",
        "Ensure staff retraining on food safety",
    ]

        return InspectionReport(
         restaurant_name="Aggregated Restaurant",
        overall_risk_score=risk_score,
        violations=final_violations,
        inspection_priority=priority,
        recommended_actions=recommended_actions,
        follow_up_required=priority in ["URGENT", "HIGH"],
    )


    def filter_false_positives(self, violations: List[Violation]) -> List[Violation]:
        filtered: List[Violation] = []

        sarcasm_markers = ["just kidding", "lol", "😂", "jk"]

        for v in violations:
            text = v.evidence.lower()

        # Drop low-confidence violations
            if v.confidence < 0.4:
               continue

        # Drop sarcastic / joking statements
            if any(marker in text for marker in sarcasm_markers):
             continue

            filtered.append(v)

        return filtered


def test_inspector():
    """Test the food safety inspector with various scenarios."""

    inspector = FoodSafetyInspector()

    # Test cases with varying violation types
    test_reviews = [
        {
            "restaurant": "Bob's Burgers",
            "text": "Great food but saw a mouse run across the dining room! Also, the chef wasn't wearing gloves while handling raw chicken.",
        },
        {
            "restaurant": "Pizza Palace",
            "text": "Just left and the bathroom had no soap, and I'm pretty sure that meat sitting on the counter wasn't refrigerated 😷",
        },
        {
            "restaurant": "Sushi Express",
            "text": "Love this place! Though it's weird they keep the raw fish next to the vegetables #sushitime #questionable",
        },
        {
            "restaurant": "Taco Town",
            "text": "Best tacos in town! Super clean kitchen, staff always wears hairnets, everything looks fresh!",
        },
        {
            "restaurant": "Burger Barn",
            "text": "The cockroach in my salad added extra protein! Just kidding, but seriously the place needs cleaning.",
        },
    ]

    print("🍽️ FOOD SAFETY INSPECTION SYSTEM 🍽️\n")
    print("=" * 70)

    for review_data in test_reviews:
        print(f"\n🏪 Restaurant: {review_data['restaurant']}")
        print(f"📝 Review: \"{review_data['text'][:100]}...\"")

        # Analyze the review
        report = inspector.analyze_review(
            review_data["text"], review_data["restaurant"]
        )

        # Display results
        print(f"\n📊 Inspection Report:")
        print(f"  Risk Score: {report.overall_risk_score}/100")
        print(f"  Priority: {report.inspection_priority}")
        print(f"  Violations Found: {len(report.violations)}")

        if report.violations:
            print("\n  Detected Violations:")
            for v in report.violations:
                print(f"    • [{v.severity}] {v.category}: {v.description}")
                print(f'      Evidence: "{v.evidence[:50]}..."')
                print(f"      Confidence: {v.confidence:.0%}")

        if report.recommended_actions:
            print("\n  Recommended Actions:")
            for action in report.recommended_actions:
                print(f"    ✓ {action}")

        print(f"\n  Follow-up Required: {'Yes' if report.follow_up_required else 'No'}")
        print("-" * 70)

    # Test batch analysis
    print("\n🔬 BATCH ANALYSIS TEST:")
    print("=" * 70)

    # Multiple reviews for same restaurant
    batch_reviews = [
        {"text": "Saw bugs in the kitchen!", "source": "Yelp"},
        {"text": "Food was cold and undercooked", "source": "Google"},
        {"text": "Staff not wearing hairnets", "source": "Twitter"},
    ]

    # TODO: Uncomment when batch_analyze is implemented
    # batch_report = inspector.batch_analyze(batch_reviews)
    # print(f"Aggregate Risk Score: {batch_report.overall_risk_score}/100")
    # print(f"Total Violations: {len(batch_report.violations)}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    test_inspector()
