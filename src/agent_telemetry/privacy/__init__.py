"""The four stages that separate metrics from content.

1. ``allowlist``     decides which fields may be read at all
2. ``pseudonymize``  replaces identifiers with keyed hashes
3. ``kanon``         folds rare categories together
4. ``scan``          checks the result independently and refuses to continue on a finding
"""

from .allowlist import filter_mapping, unknown_fields
from .kanon import DEFAULT_K, apply_k_anonymity, verify_k_anonymity
from .pseudonymize import Pseudonymizer, load_or_create_salt
from .scan import Finding, PrivacyError, enforce, scan_frame, scan_rows

__all__ = [
    "DEFAULT_K",
    "Finding",
    "PrivacyError",
    "Pseudonymizer",
    "apply_k_anonymity",
    "enforce",
    "filter_mapping",
    "load_or_create_salt",
    "scan_frame",
    "scan_rows",
    "unknown_fields",
    "verify_k_anonymity",
]
