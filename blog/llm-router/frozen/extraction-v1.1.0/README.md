# Frozen extraction contract v1.1.0

Prompt, schema, annotation-guide and inference-settings IDs are respectively `support-extraction-prompt-v1.1.0`, `support-extraction-schema-v1.1.0`, `support-annotation-guide-v1.1.0`, and `support-inference-settings-v1.1.0`.

The prompt is revised and **untested**. It contains general rules only, without held-out messages, IDs or reference answers. No model performance improvement is claimed. The output schema shape and runtime validator logic are unchanged; model digests and generation options are unchanged. Version IDs and prompt bindings are updated. Paid inference remains disabled.

The alignment review maps every substantive annotation convention to the old prompt coverage and new policy section. Experimental scoring/storage conventions remain offline responsibilities, never labels supplied to the model. `runtime_check_pass` is not semantic correctness; live `reference_correct` remains null and is unavailable to routing. No routing is implemented.

The manifest hashes every contract file. Preserve this directory; any subsequent change requires a new version. Hashes detect drift, not malicious manifest rewrites or filesystem edits. Previous contracts and pilot evidence remain preserved.
