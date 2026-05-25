# Scenario 14 — Routing policy contract validation

Rows: 130

## Key findings

- Validates FLAG / ONLY_ACCEPTED / ONLY_REJECTED routing invariants on binary datasets.
- I1: FLAG rejected mask length == n_test.
- I2: FLAG prediction_set is a (n_test, n_classes) boolean array in result.metadata.
- I3: FLAG original_count == n_test.
- I4: ONLY_ACCEPTED source_indices count matches expected accepted count.
- I5: ONLY_REJECTED source_indices count matches expected rejected count.
- I6: ONLY_ACCEPTED + ONLY_REJECTED source_indices are disjoint and cover all n_test.
- I7: No degraded_mode markers on healthy data.
- Contract passes: 130/130. Failures: 0. Degraded: 0.

## Outcome snapshot

- **rows**: 130
- **datasets**: 26
- **seeds**: 5
- **contract_passes**: 130
- **contract_failures**: 0
- **degraded_rows**: 0
- **invariant_failures**: {'i1_flag_rejected_length': 0, 'i2_flag_prediction_set_shape': 0, 'i3_flag_original_count': 0, 'i4_only_accepted_indices': 0, 'i5_only_rejected_indices': 0, 'i6_index_consistency': 0, 'i7_no_degraded_mode': 0}

## Invariant failure counts

| invariant | failures |
|---|---|
| i1_flag_rejected_length | 0 |
| i2_flag_prediction_set_shape | 0 |
| i3_flag_original_count | 0 |
| i4_only_accepted_indices | 0 |
| i5_only_rejected_indices | 0 |
| i6_index_consistency | 0 |
| i7_no_degraded_mode | 0 |

## Per-dataset contract summary

| dataset | contract_passes | total | any_failure | any_degraded |
|---|---|---|---|---|
| breast_cancer | 5 | 5 | 0 | no |
| colic | 5 | 5 | 0 | no |
| creditA | 5 | 5 | 0 | no |
| diabetes | 5 | 5 | 0 | no |
| german | 5 | 5 | 0 | no |
| haberman | 5 | 5 | 0 | no |
| heartC | 5 | 5 | 0 | no |
| heartH | 5 | 5 | 0 | no |
| heartS | 5 | 5 | 0 | no |
| hepati | 5 | 5 | 0 | no |
| iono | 5 | 5 | 0 | no |
| je4042 | 5 | 5 | 0 | no |
| je4243 | 5 | 5 | 0 | no |
| kc1 | 5 | 5 | 0 | no |
| kc2 | 5 | 5 | 0 | no |
| kc3 | 5 | 5 | 0 | no |
| liver | 5 | 5 | 0 | no |
| pc1req | 5 | 5 | 0 | no |
| pc4 | 5 | 5 | 0 | no |
| sonar | 5 | 5 | 0 | no |
| spect | 5 | 5 | 0 | no |
| spectf | 5 | 5 | 0 | no |
| transfusion | 5 | 5 | 0 | no |
| ttt | 5 | 5 | 0 | no |
| vote | 5 | 5 | 0 | no |
| wbc | 5 | 5 | 0 | no |
