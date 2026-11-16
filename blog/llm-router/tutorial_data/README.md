# Reader bounded example

These are byte-for-byte copies of the 40 seed and four demo input/reference files from synthetic-draft-v1.2.0. The original dataset is unchanged. Labels are synthetic Codex-authored drafts with AI-assisted review, pending independent human review; they are not certified ground truth.

`inputs/` contains only ID/message. `offline_labels/` is read only by the explicit offline checker. No evaluation inputs, labels, outcomes or scores are required. Copying these files is not new inference or a new experiment.

Fresh runs create their own state under `artifacts/<run-id>/`. No private journal is distributed or required. Four demo messages do not guarantee five-neighbor support, a model switch or improved routing.
