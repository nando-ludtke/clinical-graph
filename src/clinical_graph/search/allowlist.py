"""Verified clinical website allowlist partitioned into dominant and regional tiers."""

from urllib.parse import urlparse

DOMINANT_DOMAINS: frozenset[str] = frozenset(
    {
        "who.int",
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "merckmanuals.com",
        "msdmanuals.com",
        "nih.gov",
        "medlineplus.gov",
        "cdc.gov",
        "fda.gov",
        "dailymed.nlm.nih.gov",
        "open.fda.gov",
        "mayoclinic.org",
        "my.clevelandclinic.org",
        "hopkinsmedicine.org",
        "nhs.uk",
        "nice.org.uk",
        "ema.europa.eu",
        "ecdc.europa.eu",
        "canada.ca",
        "healthdirect.gov.au",
    }
)

REGIONAL_DOMAINS: frozenset[str] = frozenset(
    {
        # Global specialized
        "clinicaltrials.gov",
        "orpha.net",
        # UK regulatory (excluding geo-blocked bnf.nice.org.uk / cks.nice.org.uk)
        "gov.uk",
        "products.mhra.gov.uk",
        # EU / Europe (PDF-heavy national bodies)
        "register.awmf.org",
        "awmf.org",
        "rki.de",
        "has-sante.fr",
        # Canada / Australia
        "cda-amc.ca",
        "tga.gov.au",
        "safetyandquality.gov.au",
        "australianprescriber.tg.org.au",
        # India
        "icmr.gov.in",
        "ncdirindia.org",
        "mohfw.gov.in",
        "nhsrcindia.org",
        "cdsco.gov.in",
        "aiims.edu",
        "newbornwhocc.org",
        "nhp.gov.in",
        "janaushadhi.gov.in",
        # China
        "nhc.gov.cn",
        "nmpa.gov.cn",
        "chinacdc.cn",
        "cma.org.cn",
        "guidelines-registry.org",
        "cpa.org.cn",
    }
)

ALLOWED_DOMAINS: frozenset[str] = DOMINANT_DOMAINS | REGIONAL_DOMAINS

_GEO_BLOCKED_EXACT: frozenset[str] = frozenset(
    {
        "bnf.nice.org.uk",
        "cks.nice.org.uk",
    }
)


def extract_hostname(url: str) -> str:
    """Return normalized lowercase hostname without leading ``www.`` or port."""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"}:
        return ""
    host = (parsed.hostname or "").lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def _matches_domain_set(host: str, domains: frozenset[str]) -> bool:
    if not host or host in _GEO_BLOCKED_EXACT:
        return False
    if host in domains:
        return True
    return any(host.endswith(f".{domain}") for domain in domains)


def is_url_allowed(url: str) -> bool:
    """Return ``True`` if ``url`` is HTTP(S) and belongs to the verified allowlist."""
    host = extract_hostname(url)
    return _matches_domain_set(host, ALLOWED_DOMAINS)


def classify_domain_tier(url: str) -> str:
    """Classify a URL as ``dominant``, ``regional``, or ``disallowed``."""
    host = extract_hostname(url)
    if _matches_domain_set(host, DOMINANT_DOMAINS):
        return "dominant"
    if _matches_domain_set(host, REGIONAL_DOMAINS):
        return "regional"
    return "disallowed"
