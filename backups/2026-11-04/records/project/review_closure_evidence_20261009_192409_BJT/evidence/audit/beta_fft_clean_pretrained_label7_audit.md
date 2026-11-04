# beta-FFT clean pretrained label7 audit

generated_at: 2026-06-18 00:37:05 CST

## Conclusion

`Pre_model/label7` is **UNPROVEN_NOT_REUSABLE** for clean split full.

The local checkpoint contains only `net` and `opt`; no args, split list, case IDs, seed, root path, or training metadata were found. The `label7` directory contains only `unet_best_model.pth`. Therefore the full pretrained frame-volume set cannot be completely recovered from local evidence.

Per the requested rule, inability to fully recover the set is not safe by default. The clean full launch script must not use or fall back to official `Pre_model/label7`.
