# beta-FFT ACDC official supervised pretrain recovery

generated_at: 2026-06-18 00:55:01 CST

Recovered official ACDC training entry from README: `cd train_ACDC && python beta_fft_label7.py`.

Recovered supervised pretraining implementation from `train_ACDC/beta_fft_label7.py::pre_train`:

- model: `BCP_net(in_chns=1, class_num=num_classes)`
- optimizer: SGD, lr 0.01, momentum 0.9, weight_decay 0.0001
- loss: official `mix_loss`, optimized as `(loss_dice + loss_ce) / 2`
- training length: `pre_iterations=10000`
- batch size: 24, labeled_bs: 12
- image size: [256, 256]
- validation interval: 200 iterations
- best criterion: validation mean Dice improvement
- checkpoint format: `{'net': model.state_dict(), 'opt': optimizer.state_dict()}`

The official code contains a pretraining function but the local `__main__` has that call commented out. Since all critical logic and parameters are recoverable, a clean explicit-list pretraining entry was added at:

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/pretrain_acdc_clean_supervised.py`

This entry reuses the recovered model/loss/optimizer/schedule/checkpoint format and replaces official front-slice selection with explicit clean labeled list loading.
