# Search on -uncond with the ensemble guide (queue2.sh)

Scored on the CPU in float32 (`SCORE_DEVICE=cpu`) while the GPU ran the queue. The baseline arms score the same as on the GPU to the third decimal.

```
Baseline uncond_ens. Change in the mean over the judges each arm may be reported on, paired on
(prompt, seed); 95% t interval and Wilcoxon p over prompts; prompts improved; time is the median.
arm                                 change      ±      p  wins   n |   b16  vith  sig2 | aes16 aes32  sat clip% |     s  GiB
uncond_ens                         +0.0000 0.0000  1.000  0/12  36 | 0.323 0.314 0.124 |  4.97  5.11 0.37   1.4 |  32.0  5.9
uncond                             -0.0298 0.0170  0.002  2/12  36 | 0.299 0.274 0.097 |  4.87  5.00 0.41   2.4 |  17.5  5.5
r2_e                               +0.0179 0.0170  0.034  8/12  36 | 0.338 0.332 0.144 |  5.49  5.42 0.36   0.5 |  53.2  5.9
n90_e                              +0.0180 0.0171  0.064  9/12  36 | 0.331 0.338 0.146 |  4.99  5.22 0.32   0.6 |  46.8  5.9
roll3_e                            -0.0105 0.0217  0.677  6/12  36 | 0.309 0.306 0.114 |  4.85  4.96 0.45   1.5 |  45.3  5.9

[exited with code 0]

Aesthetic change per prompt, mean over seeds, 95% t interval, Wilcoxon p, prompts improved:
r2_e vs uncond_ens aesb16: +0.514 ± 0.335, p 0.003, 11/12
r2_e vs uncond_ens aesb32: +0.309 ± 0.271, p 0.034, 8/12
n90_e vs uncond_ens aesb16: +0.012 ± 0.197, p 0.850, 6/12
n90_e vs uncond_ens aesb32: +0.105 ± 0.288, p 1.000, 4/12
r2_e vs n90_e aesb16: +0.502 ± 0.393, p 0.021, 9/12
r2_e vs n90_e aesb32: +0.204 ± 0.367, p 0.204, 7/12
```

## Selection by PE-Core-L-14-336, which reports nothing

```
Baseline uncond_ens. Change in the mean over the judges each arm may be reported on, paired on
(prompt, seed); 95% t interval and Wilcoxon p over prompts; prompts improved; time is the median.
arm                                 change      ±      p  wins   n |   b16  vith  sig2 | aes16 aes32  sat clip% |     s  GiB
uncond_ens                         +0.0000 0.0000  1.000  0/12  36 | 0.323 0.314 0.124 |  4.97  5.11 0.37   1.4 |  32.0  5.9
r2_e                               +0.0179 0.0170  0.034  8/12  36 | 0.338 0.332 0.144 |  5.49  5.42 0.36   0.5 |  53.2  5.9
n90_e                              +0.0180 0.0171  0.064  9/12  36 | 0.331 0.338 0.146 |  4.99  5.22 0.32   0.6 |  46.8  5.9
bo4_e                              +0.0239 0.0088  0.001 11/12  36 | 0.336 0.344 0.152 |  5.21  5.42 0.40   3.0 | 125.3  7.1
fk4_e                              +0.0250 0.0106  0.001 11/12  36 | 0.339 0.341 0.155 |  5.31  5.37 0.42   2.4 | 126.7  7.1
prune_e                            +0.0241 0.0084  0.001 11/12  36 | 0.340 0.342 0.150 |  5.37  5.29 0.39   4.2 | 153.3  7.1

Baseline bo4_e. Change in the mean over the judges each arm may be reported on, paired on
(prompt, seed); 95% t interval and Wilcoxon p over prompts; prompts improved; time is the median.
arm                                 change      ±      p  wins   n |   b16  vith  sig2 | aes16 aes32  sat clip% |     s  GiB
bo4_e                              +0.0000 0.0000  1.000  0/12  36 | 0.336 0.344 0.152 |  5.21  5.42 0.40   3.0 | 125.3  7.1
fk4_e                              +0.0011 0.0058  0.622  7/12  36 | 0.339 0.341 0.155 |  5.31  5.37 0.42   2.4 | 126.7  7.1
prune_e                            +0.0002 0.0038  0.970  6/12  36 | 0.340 0.342 0.150 |  5.37  5.29 0.39   4.2 | 153.3  7.1
r2_e                               -0.0060 0.0131  0.339  3/12  36 | 0.338 0.332 0.144 |  5.49  5.42 0.36   0.5 |  53.2  5.9

bo4_e vs uncond_ens aesb16: +0.237 ± 0.329, p 0.129, 9/12
fk4_e vs uncond_ens aesb16: +0.335 ± 0.232, p 0.027, 11/12
prune_e vs uncond_ens aesb16: +0.394 ± 0.408, p 0.092, 8/12
fk4_e vs bo4_e aesb16: +0.098 ± 0.241, p 0.519, 5/12
r2_e vs bo4_e aesb16: +0.277 ± 0.349, p 0.176, 7/12
```

## Recurrence follow-ups (queue3.sh)

```
Baseline uncond_ens. Change in the mean over the judges each arm may be reported on, paired on
(prompt, seed); 95% t interval and Wilcoxon p over prompts; prompts improved; time is the median.
arm                                 change      ±      p  wins   n |   b16  vith  sig2 | aes16 aes32  sat clip% |     s  GiB
uncond_ens                         +0.0000 0.0000  1.000  0/12  36 | 0.323 0.314 0.124 |  4.97  5.11 0.37   1.4 |  32.0  5.9
r2_e                               +0.0179 0.0170  0.034  8/12  36 | 0.338 0.332 0.144 |  5.49  5.42 0.36   0.5 |  53.2  5.9
n90_e                              +0.0180 0.0171  0.064  9/12  36 | 0.331 0.338 0.146 |  4.99  5.22 0.32   0.6 |  46.8  5.9
r2n90_e                            +0.0137 0.0119  0.042  8/12  36 | 0.331 0.326 0.144 |  5.30  5.30 0.35   0.6 |  77.9  5.9
r3_e                               +0.0232 0.0121  0.003 10/12  36 | 0.332 0.348 0.150 |  5.33  5.39 0.37   1.0 |  74.0  5.9
r2all_e                            +0.0201 0.0164  0.027  9/12  36 | 0.339 0.338 0.143 |  5.44  5.33 0.36   0.5 |  58.3  5.9

Baseline r2_e. Change in the mean over the judges each arm may be reported on, paired on
(prompt, seed); 95% t interval and Wilcoxon p over prompts; prompts improved; time is the median.
arm                                 change      ±      p  wins   n |   b16  vith  sig2 | aes16 aes32  sat clip% |     s  GiB
r2_e                               +0.0000 0.0000  1.000  0/12  36 | 0.338 0.332 0.144 |  5.49  5.42 0.36   0.5 |  53.2  5.9
n90_e                              +0.0001 0.0176  0.733  7/12  36 | 0.331 0.338 0.146 |  4.99  5.22 0.32   0.6 |  46.8  5.9
r2n90_e                            -0.0042 0.0189  0.850  7/12  36 | 0.331 0.326 0.144 |  5.30  5.30 0.35   0.6 |  77.9  5.9
r3_e                               +0.0053 0.0108  0.204  8/12  36 | 0.332 0.348 0.150 |  5.33  5.39 0.37   1.0 |  74.0  5.9
r2all_e                            +0.0022 0.0044  0.380  7/12  36 | 0.339 0.338 0.143 |  5.44  5.33 0.36   0.5 |  58.3  5.9

r2n90_e vs uncond_ens aesb16: +0.329 ± 0.362, p 0.129, 7/12
r3_e vs uncond_ens aesb16: +0.358 ± 0.250, p 0.009, 11/12
r2all_e vs uncond_ens aesb16: +0.461 ± 0.339, p 0.012, 10/12
r2n90_e vs r2_e aesb16: -0.184 ± 0.379, p 0.470, 4/12
r3_e vs r2_e aesb16: -0.156 ± 0.373, p 0.519, 5/12
r2all_e vs r2_e aesb16: -0.052 ± 0.080, p 0.233, 5/12
```

Neither more steps, a third run nor recurring below t 300 beats two runs above t 300 at 60 steps, which is what cgd now does by default with -uncond.
