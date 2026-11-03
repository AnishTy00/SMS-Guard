"""Rule-Based Risk Signal Detector (Transparent Heuristics).

Inspects raw SMS text for common scam, phishing, and fraud patterns:
1. Suspicious web links, shortened URLs, and IP addresses.
2. High-urgency and coercive language.
3. Monetary amounts, currency symbols, and lottery/prize claims.
4. Solicitations for sensitive credentials (OTPs, PINs, passwords, KYC, bank info).
5. Suspicious phone numbers and premium shortcodes.

NOTE: These heuristic indicators are supplementary signals and do not
independently establish fraud or malicious intent.
"""

import re
from typing import Any, Dict, List

# Regex patterns for heuristic risk detection
URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.(?:com|org|net|xyz|top|biz|info|cc|online|site|live|vip|club|link|tk|ml|ga|cf|gq)\b/\S*|\bbit\.ly/\S+|\btinyurl\.com/\S+|\bgoo\.gl/\S+|\bt\.co/\S+|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)",
    re.IGNORECASE,
)

CURRENCY_PATTERN = re.compile(
    r"([\$£€¥₹]|\b(?:usd|eur|gbp|inr|rs\.?|cash|prize|reward|bonus|jackpot|lottery|payout|claim|won|winner)\b)",
    re.IGNORECASE,
)

URGENCY_PATTERN = re.compile(
    r"\b(urgent|urgently|immediate|immediately|act now|action required|expire|expiring|expires|last chance|final notice|hurry|limited time|valid today|don't delay|instant|now!)\b",
    re.IGNORECASE,
)

SENSITIVE_DATA_PATTERN = re.compile(
    r"\b(otp|pin|password|passcode|secret code|bank account|account number|verify account|update kyc|kyc update|credit card|debit card|cvv|security code|ssn|identity|credentials|login to)\b",
    re.IGNORECASE,
)

PHONE_SHORTCODE_PATTERN = re.compile(
    r"(\b(?:call|text|txt|ring)\s+(?:now\s+)?(\b\d{4,12}\b)|\b(?:0800|0845|0906|0870|070)\d{6,8}\b|\b\d{5}\b)",
    re.IGNORECASE,
)


def analyze_risk_indicators(text: str) -> Dict[str, Any]:
    """Scan raw message text for heuristic risk indicators.

    Returns:
        Structured dictionary containing detected indicators, count, risk tier,
        and transparency disclaimer.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "risk_tier": "None",
            "indicator_count": 0,
            "indicators": [],
            "disclaimer": "No message provided for heuristic scanning.",
        }

    indicators: List[Dict[str, str]] = []

    # 1. URL / Link Detection
    found_urls = URL_PATTERN.findall(text)
    if found_urls:
        indicators.append({
            "category": "Suspicious URL / Link",
            "severity": "High",
            "evidence": ", ".join(found_urls[:3]),
            "description": "Message contains external web links or shortened URLs often utilized in phishing.",
        })

    # 2. Sensitive Information / OTP Requests
    found_sensitive = SENSITIVE_DATA_PATTERN.findall(text)
    if found_sensitive:
        indicators.append({
            "category": "Sensitive Data Solicitation",
            "severity": "High",
            "evidence": ", ".join(list(set(found_sensitive))[:3]),
            "description": "Message references OTPs, PINs, passwords, bank accounts, or KYC verification.",
        })

    # 3. Urgency / Coercive Language
    found_urgency = URGENCY_PATTERN.findall(text)
    if found_urgency:
        indicators.append({
            "category": "Urgency & Pressure Language",
            "severity": "Moderate",
            "evidence": ", ".join(list(set(found_urgency))[:3]),
            "description": "Urgent phrasing prompting rapid action before verification.",
        })

    # 4. Financial / Currency Cues
    found_currency = CURRENCY_PATTERN.findall(text)
    if found_currency:
        indicators.append({
            "category": "Financial & Prize Claims",
            "severity": "Moderate",
            "evidence": ", ".join(list(set(found_currency))[:4]),
            "description": "References monetary rewards, currency signs, or lottery/prize claims.",
        })

    # 5. Shortcodes / Premium Callback Numbers
    found_phones = PHONE_SHORTCODE_PATTERN.findall(text)
    if found_phones:
        phone_matches = [m[0] if isinstance(m, tuple) else m for m in found_phones]
        indicators.append({
            "category": "Callback Shortcode / Premium Number",
            "severity": "Moderate",
            "evidence": ", ".join(phone_matches[:3]),
            "description": "Prompts a callback or SMS to a premium or shortcode number.",
        })

    count = len(indicators)
    high_count = sum(1 for ind in indicators if ind["severity"] == "High")

    if count == 0:
        risk_tier = "Clean"
    elif high_count >= 1 or count >= 3:
        risk_tier = "High Alert"
    elif count >= 1:
        risk_tier = "Moderate Caution"
    else:
        risk_tier = "Low"

    return {
        "risk_tier": risk_tier,
        "indicator_count": count,
        "indicators": indicators,
        "disclaimer": (
            "Rule-based indicators highlight heuristic risk signals and do not "
            "independently establish fraud. They serve as supporting context alongside ML classification."
        ),
    }


if __name__ == "__main__":
    sample = "URGENT! Your Bank account is locked. Verify your OTP now at http://bit.ly/bank-sec or call 0800123456"
    res = analyze_risk_indicators(sample)
    print("Risk Tier:", res["risk_tier"])
    print("Indicators Found:", res["indicator_count"])
    for ind in res["indicators"]:
        print(f" - [{ind['severity']}] {ind['category']}: {ind['evidence']}")
