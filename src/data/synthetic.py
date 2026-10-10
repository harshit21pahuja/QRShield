"""
QRShield — Synthetic Smoke-Test Dataset

Generates a small, deterministic synthetic dataset for testing the
entire pipeline without the real CIC-Trap4Phish data.

The synthetic URLs are designed to exercise feature extraction edge cases:
  - Short vs long URLs
  - HTTP vs HTTPS
  - IP-based hostnames
  - URLs with many special characters
  - Suspicious-looking tokens
  - Clean legitimate-looking URLs

This dataset is NOT for research evaluation. It is ONLY for:
  - verifying the pipeline works end-to-end
  - running unit/integration tests
  - CI/CD smoke tests

SECURITY: All URLs are synthetic strings. They are NEVER visited.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _benign_urls() -> List[str]:
    """Generate a list of synthetic benign-looking URLs."""
    return [
        "https://www.google.com/search?q=weather",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://github.com/torvalds/linux",
        "https://stackoverflow.com/questions/12345",
        "https://www.bbc.co.uk/news/technology",
        "https://docs.python.org/3/library/json.html",
        "https://www.reddit.com/r/programming/",
        "https://medium.com/@user/article-title-here",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://mail.google.com/mail/u/0/#inbox",
        "https://www.amazon.com/dp/B08N5WRWNW",
        "https://www.nytimes.com/2024/01/15/world.html",
        "https://calendar.google.com/calendar/r",
        "https://www.linkedin.com/in/jane-doe-12345/",
        "https://www.apple.com/iphone/",
        "https://www.microsoft.com/en-us/windows",
        "https://pytorch.org/docs/stable/index.html",
        "https://numpy.org/doc/stable/reference/",
        "https://pandas.pydata.org/docs/getting_started.html",
        "https://scikit-learn.org/stable/modules/ensemble.html",
        "https://www.mozilla.org/en-US/firefox/new/",
        "https://www.dropbox.com/home",
        "https://www.spotify.com/us/premium/",
        "https://www.netflix.com/browse",
        "https://zoom.us/j/1234567890",
        "https://www.coursera.org/learn/machine-learning",
        "https://www.khanacademy.org/math/algebra",
        "https://www.nationalgeographic.com/animals/",
        "https://www.weather.com/weather/today/",
        "https://www.imdb.com/title/tt0111161/",
        # Different domains to test grouping
        "https://news.ycombinator.com/item?id=12345",
        "https://www.theguardian.com/world/2024",
        "https://arxiv.org/abs/2301.00001",
        "https://www.reuters.com/technology",
        "https://www.nasa.gov/missions",
    ]


def _malicious_urls() -> List[str]:
    """Generate a list of synthetic malicious-looking URLs."""
    return [
        "http://192.168.1.100/login/verify-account.php",
        "http://10.0.0.1:8080/admin/cmd=resetpassword",
        "http://secure-banking-login.xyz/account/update",
        "http://paypal-verify.suspicious-domain.tk/signin",
        "http://microsoft-support.malicious.ru/helpdesk/unlock",
        "http://apple-id-confirm.phishing.cn/security/alert",
        "http://google-account-verify.bad.info/recover",
        "http://facebook-login-secure.evil.cc/credential",
        "http://netflix-billing.fraud.top/payment/invoice",
        "http://amazon-order-confirm.scam.xyz/wallet",
        "http://192.168.0.1/webscr?cmd=login&suspend=true",
        "http://172.16.0.1:443/admin/password/authenticate",
        "http://login-verify-update.co.tk/secure/expire",
        "http://banking-alert-urgent.pw/account/restore",
        "http://support-helpdesk-admin.gq/credential/unlock",
        "http://1.2.3.4/phishing/page.html?user=admin&pass=123",
        "http://secure-update-verify.ml/signin/confirm",
        "http://paypal-billing-payment.ga/security/invoice",
        "http://microsoft-admin-support.cf/alert/urgent",
        "http://apple-account-suspend.tk/recover/password",
        "http://10.20.30.40:9090/login.php?action=verify&token=abc",
        "http://google-security-alert.bad.ws/authenticate",
        "http://facebook-unlock-account.evil.biz/confirm",
        "http://netflix-password-expire.scam.pro/billing",
        "http://amazon-wallet-update.phish.net/payment",
        "http://login-banking-secure.co.tk/admin/suspend",
        "http://verify-credential-update.malicious.org/helpdesk",
        "http://192.0.2.1/signin?redirect=http://evil.com",
        "http://confirm-payment-invoice.fraud.io/urgent",
        "http://192.168.100.200:4443/secure-login/webscr/cmd=verify",
        # More with different malicious domains
        "http://banking-phish-login.evil.xyz/secure/update",
        "http://ebay-account-alert.hack.top/verify/password",
        "http://support-unlock-urgent.bad.club/credential",
        "http://paypal-suspend-expire.scam.site/admin",
        "http://google-billing-invoice.phish.live/payment",
    ]


def generate_smoke_dataset(seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic synthetic dataset for smoke testing.

    The dataset contains:
      - 35 benign URLs with label=0
      - 35 malicious URLs with label=1
      - Diverse domains to test grouped splitting
      - Edge cases for feature extraction

    Args:
        seed: Not used for generation (deterministic), but kept for API consistency.

    Returns:
        DataFrame with columns: ['url', 'label']
    """
    benign = _benign_urls()
    malicious = _malicious_urls()

    urls = benign + malicious
    labels = [0] * len(benign) + [1] * len(malicious)

    df = pd.DataFrame({"url": urls, "label": labels})

    logger.info(
        "Generated smoke dataset: %d samples (%d benign, %d malicious)",
        len(df), len(benign), len(malicious),
    )

    return df
