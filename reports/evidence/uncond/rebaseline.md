# Re-baseline (queue1.sh)

All at the defaults, 12 held-out prompts x seeds 0-2, on the RTX 2070 SUPER. Scored on the GPU in half precision (`scores.json`).

```
Baseline uncond. Change in the mean over the judges each arm may be reported on, paired on
(prompt, seed); 95% t interval and Wilcoxon p over prompts; prompts improved; time is the median.
arm                                 change      ±      p  wins   n |   b16  vith  sig2 | aes16 aes32  sat clip% |     s  GiB
uncond                             +0.0000 0.0000  1.000  0/12  36 | 0.299 0.274 0.097 |  4.87  5.00 0.41   2.4 |  17.5  5.5
uncond_ens                         +0.0298 0.0170  0.002 10/12  36 | 0.323 0.314 0.124 |  4.97  5.11 0.37   1.4 |  32.0  5.9
turbo                              +0.0848 0.0155  0.000 12/12  36 | 0.352 0.383 0.190 |  7.07  7.51 0.38   1.2 |  34.9  3.2
cc256                              +0.0266 0.0248  0.052 10/12  36 | 0.321 0.308 0.121 |  5.14  5.32 0.51   4.7 |   7.8  5.5
```
