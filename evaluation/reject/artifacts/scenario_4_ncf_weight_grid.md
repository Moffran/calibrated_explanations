# Scenario 4 — NCF and blend weight grid

Rows: 368

## Key findings

- NCFs tested: default, ensured. Entropy and explicit hinge/margin modes are excluded from this suite.
- accept_rate is the fraction of accepted instances — NOT ICP label-set coverage.
- w >= 0.7 converges NCF behavior; w=0.3 amplifies differences between NCFs where present.
- Accepted accuracy delta is always empirical and benchmarked against the non-reject baseline.

## Outcome snapshot

- **rows**: 368
- **datasets**: 46
- **best_accuracy_delta**: 0.3810
- **ncfs_tested**: ['default', 'ensured']
- **w_values_tested**: [0.3, 0.5, 0.7, 1.0]

## NCF x weight grid (binary)

| ncf | w | mean_accept_rate | mean_accepted_accuracy | mean_accuracy_delta |
|---|---|---|---|---|
| default | 0.3000 | 0.5946 | 0.8904 | 0.0893 |
| default | 0.5000 | 0.5946 | 0.8904 | 0.0893 |
| default | 0.7000 | 0.5946 | 0.8904 | 0.0893 |
| default | 1.0000 | 0.5946 | 0.8904 | 0.0893 |
| ensured | 0.3000 | 0.2266 | 0.7637 | -0.0357 |
| ensured | 0.5000 | 0.2590 | 0.7504 | -0.0507 |
| ensured | 0.7000 | 0.4902 | 0.8517 | 0.0506 |
| ensured | 1.0000 | 0.5946 | 0.8904 | 0.0893 |

## NCF x weight grid (multiclass)

| ncf | w | mean_accept_rate | mean_accepted_accuracy | mean_accuracy_delta |
|---|---|---|---|---|
| default | 0.3000 | 0.5492 | 0.8833 | 0.0993 |
| default | 0.5000 | 0.5492 | 0.8833 | 0.0993 |
| default | 0.7000 | 0.5492 | 0.8833 | 0.0993 |
| default | 1.0000 | 0.5492 | 0.8833 | 0.0993 |
| ensured | 0.3000 | 0.3780 | 0.9283 | 0.1442 |
| ensured | 0.5000 | 0.5065 | 0.9087 | 0.1246 |
| ensured | 0.7000 | 0.5495 | 0.8964 | 0.1123 |
| ensured | 1.0000 | 0.5492 | 0.8833 | 0.0993 |

## Per-dataset accuracy delta (all datasets)

Mean and best accuracy delta across the w grid for each dataset × ncf combination.

| dataset | task_type | ncf | mean_accuracy_delta | best_accuracy_delta | mean_accept_rate |
|---|---|---|---|---|---|
| pc1req | binary | default | 0.3810 | 0.3810 | 0.0952 |
| liver | binary | default | 0.1931 | 0.1931 | 0.4493 |
| german | binary | default | 0.1908 | 0.1908 | 0.2618 |
| spectf | binary | default | 0.1900 | 0.1900 | 0.5741 |
| heartC | binary | default | 0.1878 | 0.1878 | 0.3934 |
| diabetes | binary | default | 0.1597 | 0.1597 | 0.4545 |
| haberman | binary | default | 0.1458 | 0.1458 | 0.2807 |
| hepati | binary | default | 0.0968 | 0.0968 | 0.3226 |
| transfusion | binary | default | 0.0871 | 0.0871 | 0.4950 |
| kc2 | binary | default | 0.0725 | 0.0725 | 0.5541 |
| colic | binary | default | 0.0667 | 0.0667 | 0.6944 |
| vote | binary | default | 0.0643 | 0.0643 | 0.6442 |
| pc4 | binary | default | 0.0639 | 0.0639 | 0.8587 |
| heartH | binary | default | 0.0605 | 0.0605 | 0.8305 |
| je4042 | binary | default | 0.0593 | 0.0593 | 0.5556 |
| kc1 | binary | default | 0.0525 | 0.0525 | 0.2762 |
| je4243 | binary | default | 0.0502 | 0.0502 | 0.2877 |
| iono | binary | default | 0.0429 | 0.0429 | 0.8429 |
| kc3 | binary | default | 0.0427 | 0.0427 | 0.8308 |
| spect | binary | default | 0.0336 | 0.0336 | 0.5682 |
| heartS | binary | default | 0.0285 | 0.0285 | 0.7222 |
| creditA | binary | default | 0.0242 | 0.0242 | 0.9058 |
| ttt | binary | default | 0.0156 | 0.0156 | 0.9688 |
| breast_cancer | binary | default | 0.0078 | 0.0078 | 0.9737 |
| sonar | binary | default | 0.0037 | 0.0037 | 0.6190 |
| wbc | binary | default | 0.0000 | 0.0000 | 1.0000 |
| pc1req | binary | ensured | 0.3810 | 0.3810 | 0.0595 |
| liver | binary | ensured | 0.2657 | 0.2899 | 0.3080 |
| diabetes | binary | ensured | 0.1580 | 0.1597 | 0.3994 |
| heartC | binary | ensured | 0.1358 | 0.1878 | 0.2213 |
| heartH | binary | ensured | 0.1042 | 0.2034 | 0.3602 |
| transfusion | binary | ensured | 0.0871 | 0.0871 | 0.4950 |
| spect | binary | ensured | 0.0623 | 0.1136 | 0.3068 |
| vote | binary | ensured | 0.0539 | 0.0643 | 0.5793 |
| heartS | binary | ensured | 0.0489 | 0.0593 | 0.4537 |
| iono | binary | ensured | 0.0429 | 0.0429 | 0.5679 |
| creditA | binary | ensured | 0.0351 | 0.0461 | 0.6649 |
| colic | binary | ensured | 0.0275 | 0.0667 | 0.6528 |
| breast_cancer | binary | ensured | 0.0172 | 0.0289 | 0.7851 |
| ttt | binary | ensured | 0.0156 | 0.0156 | 0.6276 |
| sonar | binary | ensured | 0.0104 | 0.1190 | 0.3869 |
| wbc | binary | ensured | 0.0010 | 0.0020 | 0.7581 |
| je4243 | binary | ensured | -0.0239 | 0.0502 | 0.2466 |
| haberman | binary | ensured | -0.0405 | 0.1458 | 0.1798 |
| kc1 | binary | ensured | -0.0431 | 0.0525 | 0.1370 |
| je4042 | binary | ensured | -0.0616 | 0.0593 | 0.3287 |
| hepati | binary | ensured | -0.1199 | 0.0968 | 0.1694 |
| german | binary | ensured | -0.1240 | 0.1908 | 0.1806 |
| pc4 | binary | ensured | -0.1275 | 0.0639 | 0.4461 |
| spectf | binary | ensured | -0.1392 | 0.1900 | 0.2870 |
| kc2 | binary | ensured | -0.1470 | 0.0725 | 0.1520 |
| kc3 | binary | ensured | -0.2757 | 0.0467 | 0.4538 |
| cmc | multiclass | default | 0.2625 | 0.2625 | 0.1831 |
| wineW | multiclass | default | 0.2267 | 0.2267 | 0.3602 |
| wineR | multiclass | default | 0.1992 | 0.1992 | 0.2875 |
| whole | multiclass | default | 0.1841 | 0.1841 | 0.1136 |
| vehicle | multiclass | default | 0.1648 | 0.1648 | 0.6118 |
| steel | multiclass | default | 0.1616 | 0.1616 | 0.5835 |
| yeast | multiclass | default | 0.1367 | 0.1367 | 0.2694 |
| glass | multiclass | default | 0.1149 | 0.1149 | 0.3953 |
| ecoli | multiclass | default | 0.0920 | 0.0920 | 0.5735 |
| balance | multiclass | default | 0.0872 | 0.0872 | 0.8640 |
| wave | multiclass | default | 0.0764 | 0.0764 | 0.7230 |
| tae | multiclass | default | 0.0251 | 0.0251 | 0.2903 |
| user | multiclass | default | 0.0160 | 0.0160 | 0.8765 |
| cars | multiclass | default | 0.0132 | 0.0132 | 0.9538 |
| cool | multiclass | default | 0.0123 | 0.0123 | 0.9740 |
| image | multiclass | default | 0.0104 | 0.0104 | 0.9762 |
| vowel | multiclass | default | 0.0036 | 0.0036 | 0.9798 |
| heat | multiclass | default | -0.0002 | -0.0002 | 0.9675 |
| iris | multiclass | default | nan | nan | 0.0000 |
| wine | multiclass | default | nan | nan | 0.0000 |
| cmc | multiclass | ensured | 0.2412 | 0.2983 | 0.1525 |
| wineW | multiclass | ensured | 0.2373 | 0.2699 | 0.3224 |
| vehicle | multiclass | ensured | 0.2180 | 0.2706 | 0.4941 |
| wineR | multiclass | ensured | 0.2147 | 0.2535 | 0.2562 |
| glass | multiclass | ensured | 0.2031 | 0.2326 | 0.2907 |
| whole | multiclass | ensured | 0.1841 | 0.1841 | 0.1136 |
| steel | multiclass | ensured | 0.1752 | 0.2220 | 0.5308 |
| yeast | multiclass | ensured | 0.1685 | 0.1889 | 0.2449 |
| balance | multiclass | ensured | 0.1122 | 0.1520 | 0.7420 |
| tae | multiclass | ensured | 0.1084 | 0.2473 | 0.2500 |
| ecoli | multiclass | ensured | 0.1041 | 0.1176 | 0.4706 |
| wave | multiclass | ensured | 0.0810 | 0.0948 | 0.6915 |
| vowel | multiclass | ensured | 0.0398 | 0.0758 | 0.8295 |
| user | multiclass | ensured | 0.0208 | 0.0399 | 0.7778 |
| cool | multiclass | ensured | 0.0189 | 0.0390 | 0.9383 |
| cars | multiclass | ensured | 0.0178 | 0.0371 | 0.9350 |
| image | multiclass | ensured | 0.0153 | 0.0256 | 0.9356 |
| heat | multiclass | ensured | 0.0015 | 0.0065 | 0.9399 |
| iris | multiclass | ensured | nan | nan | 0.0000 |
| wine | multiclass | ensured | nan | nan | 0.0000 |
