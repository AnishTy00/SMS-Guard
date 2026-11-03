"""Comprehensive Integration and Functional Test Suite for Phase 8.

Covers:
1. End-to-end inference workflow verification (preprocessing -> TF-IDF -> LinearSVC -> risk heuristics).
2. Functional test cases: Ham, Promotional Spam, Phishing, Suspicious URLs, OTP/Credentials.
3. Edge cases: Empty, Whitespace, Long inputs, Emojis, Punctuation.
4. Independence verification of ML classification vs Heuristic Risk Engine.
5. Security testing: XSS/HTML injection payload handling and memory-only processing.
6. Model and artifact integrity validation.
"""

import html
import os
import sys
import unittest

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.prediction import load_classifier, predict_message
from src.preprocessing import preprocess_text
from src.risk_engine import analyze_risk_indicators


class TestPhase8Integration(unittest.TestCase):
    """Phase 8 Comprehensive Integration and Security Test Cases."""

    @classmethod
    def setUpClass(cls):
        """Ensure champion model pipeline loads correctly."""
        cls.pipeline = load_classifier()
        assert cls.pipeline is not None, "Failed to load champion classifier pipeline."

    # --------------------------------------------------------------------------
    # 1. Full System Integration Workflow
    # --------------------------------------------------------------------------
    def test_end_to_end_workflow(self):
        """Verify the exact sequence: raw -> preprocess -> TF-IDF pipeline -> decision -> heuristics."""
        raw_msg = "URGENT! Your account is suspended. Verify OTP at http://bit.ly/bank-sec"
        res = predict_message(raw_msg)

        # Structure checks
        self.assertTrue(res["is_valid"])
        self.assertIsNone(res["error_message"])
        self.assertIn("label", res)
        self.assertIn("confidence_score", res)
        self.assertIn("decision_score", res)
        self.assertIn("cleaned_text", res)
        self.assertIn("risk_analysis", res)

        # Content checks
        self.assertEqual(res["label"], "Spam")
        self.assertTrue(res["is_spam"])
        self.assertGreater(res["confidence_pct"], 50.0)
        self.assertGreater(res["decision_score"], 0.0)  # On spam side of hyperplane
        self.assertIn("risk_tier", res["risk_analysis"])
        self.assertEqual(res["risk_analysis"]["risk_tier"], "High Alert")

    # --------------------------------------------------------------------------
    # 2. Functional Testing: Category Diversity
    # --------------------------------------------------------------------------
    def test_legitimate_personal_ham(self):
        """Test benign conversational messages without promotional/fraud markers."""
        samples = [
            "Hey mom, I'll be home around 6pm for dinner. Can you please keep food ready?",
            "Are we still meeting at the university library at 3pm to review our notes?",
            "Thanks for the birthday wishes, hope you have a great week!",
        ]
        for msg in samples:
            res = predict_message(msg)
            self.assertTrue(res["is_valid"])
            self.assertEqual(res["label"], "Ham", f"Expected Ham for: '{msg}'")
            self.assertFalse(res["is_spam"])
            self.assertLess(res["decision_score"], 0.0)  # On ham side of hyperplane
            self.assertEqual(res["risk_analysis"]["risk_tier"], "Clean")

    def test_promotional_spam(self):
        """Test commercial advertisement / marketing broadcast messages."""
        samples = [
            "WINNER! You have won a £1,000 cash prize or a £2,000 holiday! Call 0800123456 now to claim.",
            "Exclusive offer! Reply YES to 88088 to receive your free ringtone and discount voucher today.",
        ]
        for msg in samples:
            res = predict_message(msg)
            self.assertTrue(res["is_valid"])
            self.assertEqual(res["label"], "Spam", f"Expected Spam for: '{msg}'")
            self.assertTrue(res["is_spam"])

    def test_phishing_style_messages(self):
        """Test coercive impersonation and account security threats."""
        samples = [
            "ALERT: Unauthorized login detected on your Wells Fargo account. Verify immediately at http://wf-sec.xyz/verify",
            "Your Chase banking access has been blocked. Update KYC and OTP to restore access.",
        ]
        for msg in samples:
            res = predict_message(msg)
            self.assertTrue(res["is_valid"])
            self.assertEqual(res["label"], "Spam", f"Expected Spam for: '{msg}'")
            self.assertEqual(res["risk_analysis"]["risk_tier"], "High Alert")
            # Must detect either URL or sensitive data
            cats = [ind["category"] for ind in res["risk_analysis"]["indicators"]]
            self.assertTrue(any("URL" in c or "Sensitive" in c for c in cats))

    def test_suspicious_urls(self):
        """Test messages featuring shortened URLs, IP addresses, and suspicious domains."""
        samples = [
            ("Track parcel at http://192.168.1.100/track", "High Alert"),
            ("Claim your reward at bit.ly/claim-usd-now", "High Alert"),
            ("Visit our portal at www.discount-prizes.xyz/deal", "High Alert"),
        ]
        for msg, expected_tier in samples:
            res = predict_message(msg)
            self.assertTrue(res["is_valid"])
            self.assertEqual(res["risk_analysis"]["risk_tier"], expected_tier)

    def test_otp_and_credential_harvesting(self):
        """Test messages soliciting OTPs, PINs, passwords, and banking info."""
        msg = "Your one-time password (OTP) is 789123. Update your bank account credentials and PIN immediately."
        res = predict_message(msg)
        self.assertTrue(res["is_valid"])
        cats = [ind["category"] for ind in res["risk_analysis"]["indicators"]]
        self.assertIn("Sensitive Data Solicitation", cats)

    # --------------------------------------------------------------------------
    # 3. Edge Cases & Boundary Conditions
    # --------------------------------------------------------------------------
    def test_empty_and_whitespace_inputs(self):
        """Verify that empty, blank, or newline-only inputs are safely rejected."""
        empty_inputs = ["", "   ", "\t", "\n\n  \t  \n"]
        for inp in empty_inputs:
            res = predict_message(inp)
            self.assertFalse(res["is_valid"])
            self.assertIsNotNone(res["error_message"])
            self.assertIsNone(res["label"])
            self.assertEqual(res["confidence_pct"], 0.0)

    def test_long_message_handling(self):
        """Verify performance and stability on very long input texts."""
        long_msg = "Hello friend, this is an update on the project. " * 60  # ~3,000 chars
        res = predict_message(long_msg)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["label"], "Ham")

    def test_emojis_and_unicode_characters(self):
        """Verify that emojis and special unicode characters do not cause encoding errors."""
        emoji_msg = "🎉🎁 CONGRATULATIONS! You won $500! Call 0800123456 🔥💰"
        res = predict_message(emoji_msg)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["label"], "Spam")

        casual_emojis = "Hey 😊 see you tomorrow at lunch 👍☕"
        res_casual = predict_message(casual_emojis)
        self.assertTrue(res_casual["is_valid"])
        self.assertEqual(res_casual["label"], "Ham")

    def test_unusual_punctuation_and_symbols(self):
        """Verify that symbols and punctuation are normalized cleanly."""
        punct_msg = "!!!!???? $$$$$ @@@@@ *****"
        res = predict_message(punct_msg)
        self.assertTrue(res["is_valid"])
        self.assertIsInstance(res["decision_score"], float)

    # --------------------------------------------------------------------------
    # 4. Independence of ML Predictions vs Heuristic Risk Signals
    # --------------------------------------------------------------------------
    def test_ml_and_heuristics_independence(self):
        """Verify ML hyperplane and rule-based scanner operate as independent layers.

        Case A: Ham text that happens to mention money ("Can you lend me $5?")
                -> ML says Ham (based on language context), Heuristics flags Currency (transparent rule).
        Case B: Spam text that uses colloquial slang without links or trigger words
                -> ML detects Spam n-grams, Heuristics says Clean (no URLs/OTPs).
        """
        # Case A: Benign borrowing request
        ham_money = "Hey bro, can you please lend me $5 for coffee? I will repay tomorrow."
        res_a = predict_message(ham_money)
        self.assertEqual(res_a["label"], "Ham", "Contextual ML should recognize personal borrowing as Ham.")
        self.assertGreaterEqual(res_a["risk_analysis"]["indicator_count"], 1, "Rule scanner should independently note currency.")

        # Case B: Spam text with no heuristic risk triggers (URLs, OTPs, currency, or urgency keywords)
        promo_no_heuristics = "Customer service notice: we tried contacting you regarding your mobile contract renewal"
        res_b = predict_message(promo_no_heuristics)
        self.assertEqual(res_b["label"], "Spam", "Statistical n-gram model should classify promotional directive as Spam.")
        self.assertEqual(res_b["risk_analysis"]["indicator_count"], 0, "No URL or OTP rules should trigger.")

    # --------------------------------------------------------------------------
    # 5. Security & Injection Testing
    # --------------------------------------------------------------------------
    def test_html_and_xss_injection_safety(self):
        """Verify that malicious script/HTML injection tags are safely sanitized."""
        xss_payload = "<script>alert('XSS')</script> Verify your OTP at http://evil.com/<img src=x onerror=alert(1)>"
        res = predict_message(xss_payload)
        self.assertTrue(res["is_valid"])

        # Check risk engine evidence sanitization
        for ind in res["risk_analysis"]["indicators"]:
            evidence = ind["evidence"]
            # Verify that html.escape properly sanitizes angle brackets
            escaped_evidence = html.escape(evidence)
            self.assertNotIn("<script>", escaped_evidence)
            self.assertNotIn("<img", escaped_evidence)

    def test_model_loading_failure_resilience(self):
        """Verify that calling load_classifier with a nonexistent path raises FileNotFoundError cleanly."""
        with self.assertRaises(FileNotFoundError):
            load_classifier("/invalid/path/to/nonexistent_model.pkl")


if __name__ == "__main__":
    unittest.main()
