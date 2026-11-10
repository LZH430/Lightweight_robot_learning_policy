# Adaptive demonstration supervision

The existing adaptive_distill.json and adaptive_gated.json now use dataset actions,
not teacher_action. Names are kept to preserve the familiar commands. This is
self-paced imitation learning, not model-to-model knowledge distillation.
The adaptive runner rejects teacher distillation settings. The legacy train.py/export backend remains available for old experiments.

For sample i, compute valid-entry MSE e_i between the student and demonstration.
Define w_i=exp(-stop_gradient(e_i)/data_temperature).
L_data=sum_i(w_i * sum_valid((prediction-target)^2))/total_valid_count.
Total loss = ordinary masked action MSE + data_weight*L_data + future_weight*L_future.
The future term is present only in future-enabled models/stages.
Defaults: data_weight=0.1, data_temperature=0.05 (normalized squared-action units).
Weights are detached: the student cannot reduce the weight through its gradient.
Ordinary action MSE remains, so hard examples retain their original supervision.
The added term emphasizes examples the model already fits; it does not measure
demonstration correctness. At these defaults the action gradient multiplier lies
between 1 and 1.1 for each sample. This is a modest reweighting, not extra teacher
knowledge, and does not guarantee better performance. Validation metrics and
best.ckpt selection remain ordinary unweighted MSE for fair comparison.

No cache regeneration is required if current caches already contain matching
visual/text/state/action/action_mask (and future for the gated model).
Teacher actions are neither required nor read. Base encoders are unchanged.

```bash
python -m student_policy.train_adaptive --config configs/adaptive_distill.json
python -m student_policy.train_adaptive --config configs/adaptive_gated.json
```

Outputs: runs/adaptive_distill_data and runs/adaptive_gated_data.
Use fresh runs; --resume rejects changed loss settings. Plain student.json and
gated_future.json remain non-adaptive controls using the same caches.
Logs add adaptive_data_mse and data_weight_mean during training. data_weight_mean
is NOT teacher confidence, data quality, or success rate. Dataset validation and
Isaac Sim trials are still necessary to assess any improvement.
