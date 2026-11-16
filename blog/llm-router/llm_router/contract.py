import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def message_key(message):
    return hashlib.sha256(' '.join(message.split()).casefold().encode()).hexdigest()


class Contract:
    def __init__(self, version='extraction-v1.1.0'):
        if version != 'extraction-v1.1.0':
            raise ValueError('Only the explicitly selected frozen contract is supported')
        self.version = version
        folder = ROOT/'frozen'/version
        manifest = json.loads((folder/'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Frozen contract hash drift: '+name)
        self.settings = json.loads((folder/'inference-settings.json').read_text())
        import os
        profile=os.environ.get('ROUTER_INFERENCE_PROFILE')
        if profile:
            if Path(profile).name!=profile: raise ValueError('Use a project profile filename')
            override=json.loads((ROOT/'inference_profiles'/profile).read_text())
            self.settings.update(override['settings'])
            self.settings['profile_version']=override['version']
            self.profile_hash=fingerprint(override)
        else:self.profile_hash=None
        self.schema = json.loads((folder/'schema.json').read_text())
        self.prompt = (folder/'prompt.txt').read_text()
        self.prompt_hash = hashlib.sha256(self.prompt.encode()).hexdigest()
        self.hash = fingerprint(dict(manifest=manifest,profile_hash=self.profile_hash))
        spec = importlib.util.spec_from_file_location('frozen_runtime_validator', folder/'runtime_validator.py')
        self.validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.validator)

    def check(self, raw, message=None):
        try:
            parsed, errors = self.validator.validate_response(raw)
            result=dict(schema_valid=True, runtime_check_pass=not errors, parsed=parsed.model_dump(), runtime_errors=errors)
            if message is not None:
                from .validation_v2 import validate_order_membership
                result=validate_order_membership(result,message)
            return result
        except ValueError as exc:
            return dict(schema_valid=False, runtime_check_pass=False, parsed=None, runtime_errors=[str(exc)])


def assert_runtime_only(value):
    """Prevent reference/annotation metadata from entering runtime stores."""
    forbidden = {'reference', 'reference_correct', 'rationale', 'label_explanation', 'field_correctness', 'schema_scope_case', 'offline_case_type', 'coverage_tags'}
    if isinstance(value, dict):
        if forbidden & value.keys():
            raise ValueError('Offline reference/annotation data is forbidden in runtime records')
        for item in value.values():
            assert_runtime_only(item)
    elif isinstance(value, list):
        for item in value:
            assert_runtime_only(item)
