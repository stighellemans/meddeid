"""Run on the driver with an approved Databricks service credential and synthetic input."""
from urllib.parse import urlsplit

from meddeid_core.client import GatewayClient
from azure.keyvault.secrets import SecretClient
from meddeid_core.rendering import PseudonymizationScope, apply_spans


def check_integration(*, credential, gateway_url, audience, model_name, bundle_sha256,
                      vault_url, secret_name, secret_version, language="nl-BE", verify=True):
    """Return a synthetic rendered note after checking identity, coverage and source slices."""
    parsed = urlsplit(gateway_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Set a private HTTPS gateway URL")
    if verify is False or not secret_version or not audience:
        raise ValueError("Configure certificate verification, audience and a pinned secret version")
    scope = audience.rstrip("/") + "/.default"
    note = {"document_id": "example-version-1", "text": "Patiënt Alex Peters kwam op 03/03/2026.",
            "metadata": {"lang": language, "document_creation_date": "2026-03-03",
                         "patient": {"given_name": "Alex", "family_name": "Peters"}}}
    client = GatewayClient(
        gateway_url,
        bearer_token=lambda: credential.get_token(scope).token,
        ca_file=None if verify is True else verify,
    )
    [record] = client.deidentify_batch([note])
    provenance = record["provenance"]
    model = provenance["model"]
    if (model["name"], model["bundle_sha256"], provenance["language_profile"]["profile_id"]) != (
        model_name, bundle_sha256, language
    ):
        raise ValueError("Unexpected model, bundle or language")
    spans = record["spans"]
    if not all("text" in span for span in spans) or not any(
        span["label"] == "Name:Patient" for span in spans
    ):
        raise ValueError("Unexpected synthetic detection")
    with SecretClient(vault_url, credential) as vault:
        key = vault.get_secret(secret_name, secret_version).value
    return apply_spans(
        note["text"], spans, profile_id=language, document_creation_date="2026-03-03",
        pseudonymization_scope=PseudonymizationScope(patient_id="example-patient", secret=key),
        name_replacement=True, date_replacement=True, age_replacement=True,
    )
