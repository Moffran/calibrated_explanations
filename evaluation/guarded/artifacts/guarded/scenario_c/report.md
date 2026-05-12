# Scenario C — Real-Dataset Guard Retention Benchmark

## Scientific Question

How does conformal guard filtering affect the number of emitted factual and alternative rules across the full dataset universe?  Are guard retention rates stable across diverse real-world datasets at ε=0.10?

**Coverage preservation is not a metric here**: it is structurally invariant under guard filtering (§2.3 of the paper).

## Task-level summary (ε=0.1)

| Task | Reg mode | Expl type | Std rules | Guarded rules | Retention | Fully filtered |
|---|---|---|---|---|---|---|
| binary | cls | alternative | 17.031 | 34.277 | 0.706 | 0.070 |
| binary | cls | factual | 17.125 | 14.668 | 0.889 | 0.087 |
| multiclass | cls | alternative | 20.025 | 62.800 | 0.770 | 0.018 |
| multiclass | cls | factual | 11.829 | 12.986 | 0.904 | 0.062 |
| regression | p25 | alternative | 12.082 | 24.287 | 0.650 | 0.019 |
| regression | p25 | factual | 6.674 | 6.235 | 0.892 | 0.056 |
| regression | p50 | alternative | 12.063 | 23.619 | 0.647 | 0.018 |
| regression | p50 | factual | 6.725 | 6.325 | 0.893 | 0.056 |
| regression | p75 | alternative | 12.246 | 24.606 | 0.653 | 0.019 |
| regression | p75 | factual | 6.830 | 6.397 | 0.894 | 0.056 |
| regression | plain | alternative | 13.526 | 31.959 | 0.698 | 0.014 |
| regression | plain | factual | 7.546 | 6.814 | 0.899 | 0.051 |

## Per-dataset results (ε=0.1)

| Task | Dataset | Reg mode | Expl type | d | Std rules | Guarded rules | Retention | Fully filtered |
|---|---|---|---|---|---|---|---|---|
| binary | diabetes | cls | alternative | 8 | 12.283 | 29.384 | 0.771 | 0.008 |
| binary | diabetes | cls | factual | 8 | 7.178 | 6.853 | 0.899 | 0.049 |
| binary | german | cls | alternative | 27 | 6.848 | 14.762 | 0.762 | 0.046 |
| binary | german | cls | factual | 27 | 14.909 | 12.796 | 0.866 | 0.078 |
| binary | kc1 | cls | alternative | 21 | 20.003 | 35.002 | 0.530 | 0.113 |
| binary | kc1 | cls | factual | 21 | 14.650 | 12.823 | 0.821 | 0.113 |
| binary | pc4 | cls | alternative | 37 | 37.711 | 76.680 | 0.831 | 0.052 |
| binary | pc4 | cls | factual | 37 | 24.226 | 25.316 | 0.897 | 0.064 |
| binary | ttt | cls | alternative | 27 | 8.311 | 15.554 | 0.636 | 0.132 |
| binary | ttt | cls | factual | 27 | 24.662 | 15.554 | 0.960 | 0.132 |
| multiclass | cars | cls | alternative | 6 | 9.027 | 13.384 | 1.000 | 0.000 |
| multiclass | cars | cls | factual | 6 | 5.458 | 5.999 | 1.000 | 0.000 |
| multiclass | cmc | cls | alternative | 9 | 13.083 | 20.188 | 0.824 | 0.018 |
| multiclass | cmc | cls | factual | 9 | 8.462 | 7.427 | 0.874 | 0.114 |
| multiclass | cool | cls | alternative | 8 | 11.604 | 12.019 | 0.612 | 0.011 |
| multiclass | cool | cls | factual | 8 | 6.611 | 7.228 | 0.910 | 0.060 |
| multiclass | heat | cls | alternative | 8 | 16.603 | 17.464 | 0.634 | 0.006 |
| multiclass | heat | cls | factual | 8 | 7.739 | 7.410 | 0.935 | 0.046 |
| multiclass | image | cls | alternative | 19 | 23.709 | 84.110 | 0.794 | 0.007 |
| multiclass | image | cls | factual | 19 | 13.638 | 17.034 | 0.901 | 0.083 |
| multiclass | steel | cls | alternative | 27 | 34.629 | 140.362 | 0.872 | 0.014 |
| multiclass | steel | cls | factual | 27 | 21.660 | 24.768 | 0.919 | 0.050 |
| multiclass | vehicle | cls | alternative | 18 | 26.539 | 68.791 | 0.593 | 0.028 |
| multiclass | vehicle | cls | factual | 18 | 15.690 | 15.402 | 0.863 | 0.079 |
| multiclass | vowel | cls | alternative | 11 | 16.943 | 52.448 | 0.745 | 0.017 |
| multiclass | vowel | cls | factual | 11 | 9.922 | 9.797 | 0.894 | 0.071 |
| multiclass | wave | cls | alternative | 40 | 43.018 | 198.746 | 0.822 | 0.031 |
| multiclass | wave | cls | factual | 40 | 24.734 | 34.008 | 0.873 | 0.058 |
| multiclass | wineR | cls | alternative | 11 | 17.014 | 54.789 | 0.779 | 0.056 |
| multiclass | wineR | cls | factual | 11 | 10.168 | 9.578 | 0.871 | 0.079 |
| multiclass | wineW | cls | alternative | 11 | 17.327 | 58.821 | 0.803 | 0.013 |
| multiclass | wineW | cls | factual | 11 | 10.146 | 9.838 | 0.895 | 0.051 |
| multiclass | yeast | cls | alternative | 8 | 10.802 | 32.473 | 0.760 | 0.012 |
| multiclass | yeast | cls | factual | 8 | 7.717 | 7.342 | 0.918 | 0.052 |
| regression | abalone | p25 | alternative | 8 | 11.162 | 13.864 | 0.349 | 0.074 |
| regression | abalone | p25 | factual | 8 | 5.520 | 5.268 | 0.875 | 0.080 |
| regression | abalone | p50 | alternative | 8 | 9.582 | 12.647 | 0.330 | 0.076 |
| regression | abalone | p50 | factual | 8 | 5.150 | 4.856 | 0.866 | 0.081 |
| regression | abalone | p75 | alternative | 8 | 10.609 | 13.498 | 0.342 | 0.080 |
| regression | abalone | p75 | factual | 8 | 5.542 | 5.133 | 0.872 | 0.080 |
| regression | abalone | plain | alternative | 8 | 14.176 | 23.412 | 0.467 | 0.051 |
| regression | abalone | plain | factual | 8 | 7.910 | 7.086 | 0.903 | 0.071 |
| regression | anacalt | p25 | alternative | 7 | 4.853 | 8.954 | 0.657 | 0.019 |
| regression | anacalt | p25 | factual | 7 | 4.644 | 4.169 | 0.899 | 0.052 |
| regression | anacalt | p50 | alternative | 7 | 6.513 | 10.216 | 0.692 | 0.008 |
| regression | anacalt | p50 | factual | 7 | 6.298 | 5.898 | 0.938 | 0.052 |
| regression | anacalt | p75 | alternative | 7 | 6.513 | 10.216 | 0.692 | 0.008 |
| regression | anacalt | p75 | factual | 7 | 6.298 | 5.898 | 0.938 | 0.052 |
| regression | anacalt | plain | alternative | 7 | 2.534 | 8.477 | 0.651 | 0.028 |
| regression | anacalt | plain | factual | 7 | 3.003 | 2.476 | 0.864 | 0.052 |
| regression | bank8fh | p25 | alternative | 8 | 10.634 | 23.572 | 0.756 | 0.002 |
| regression | bank8fh | p25 | factual | 8 | 5.929 | 5.754 | 0.883 | 0.048 |
| regression | bank8fh | p50 | alternative | 8 | 10.053 | 20.282 | 0.726 | 0.002 |
| regression | bank8fh | p50 | factual | 8 | 5.444 | 5.483 | 0.878 | 0.051 |
| regression | bank8fh | p75 | alternative | 8 | 9.827 | 21.232 | 0.734 | 0.003 |
| regression | bank8fh | p75 | factual | 8 | 5.680 | 5.474 | 0.877 | 0.057 |
| regression | bank8fh | plain | alternative | 8 | 13.300 | 36.216 | 0.825 | 0.001 |
| regression | bank8fh | plain | factual | 8 | 7.739 | 7.126 | 0.903 | 0.046 |
| regression | bank8fm | p25 | alternative | 8 | 11.504 | 26.579 | 0.771 | 0.002 |
| regression | bank8fm | p25 | factual | 8 | 6.310 | 6.062 | 0.889 | 0.059 |
| regression | bank8fm | p50 | alternative | 8 | 11.381 | 27.034 | 0.773 | 0.003 |
| regression | bank8fm | p50 | factual | 8 | 6.464 | 6.169 | 0.890 | 0.062 |
| regression | bank8fm | p75 | alternative | 8 | 12.767 | 32.452 | 0.805 | 0.002 |
| regression | bank8fm | p75 | factual | 8 | 7.143 | 6.654 | 0.898 | 0.064 |
| regression | bank8fm | plain | alternative | 8 | 10.901 | 26.856 | 0.773 | 0.001 |
| regression | bank8fm | plain | factual | 8 | 6.464 | 6.252 | 0.892 | 0.058 |
| regression | bank8nh | p25 | alternative | 8 | 10.543 | 23.843 | 0.734 | 0.003 |
| regression | bank8nh | p25 | factual | 8 | 6.460 | 6.172 | 0.891 | 0.054 |
| regression | bank8nh | p50 | alternative | 8 | 10.086 | 22.724 | 0.724 | 0.002 |
| regression | bank8nh | p50 | factual | 8 | 6.429 | 6.167 | 0.891 | 0.057 |
| regression | bank8nh | p75 | alternative | 8 | 10.536 | 21.417 | 0.707 | 0.004 |
| regression | bank8nh | p75 | factual | 8 | 6.147 | 5.831 | 0.885 | 0.062 |
| regression | bank8nh | plain | alternative | 8 | 13.830 | 36.216 | 0.806 | 0.001 |
| regression | bank8nh | plain | factual | 8 | 7.853 | 7.198 | 0.905 | 0.050 |
| regression | bank8nm | p25 | alternative | 8 | 10.611 | 23.499 | 0.735 | 0.002 |
| regression | bank8nm | p25 | factual | 8 | 6.187 | 6.033 | 0.890 | 0.061 |
| regression | bank8nm | p50 | alternative | 8 | 10.796 | 22.576 | 0.725 | 0.002 |
| regression | bank8nm | p50 | factual | 8 | 6.049 | 5.847 | 0.886 | 0.061 |
| regression | bank8nm | p75 | alternative | 8 | 11.159 | 25.341 | 0.747 | 0.002 |
| regression | bank8nm | p75 | factual | 8 | 6.427 | 6.127 | 0.892 | 0.060 |
| regression | bank8nm | plain | alternative | 8 | 12.123 | 27.970 | 0.767 | 0.002 |
| regression | bank8nm | plain | factual | 8 | 6.802 | 6.574 | 0.898 | 0.059 |
| regression | comp | p25 | alternative | 12 | 18.123 | 42.017 | 0.716 | 0.027 |
| regression | comp | p25 | factual | 12 | 10.012 | 9.663 | 0.901 | 0.057 |
| regression | comp | p50 | alternative | 12 | 16.502 | 34.847 | 0.681 | 0.020 |
| regression | comp | p50 | factual | 12 | 9.566 | 9.277 | 0.897 | 0.058 |
| regression | comp | p75 | alternative | 12 | 17.989 | 42.494 | 0.721 | 0.019 |
| regression | comp | p75 | factual | 12 | 10.400 | 9.868 | 0.903 | 0.056 |
| regression | comp | plain | alternative | 12 | 18.747 | 50.909 | 0.753 | 0.024 |
| regression | comp | plain | factual | 12 | 10.916 | 10.427 | 0.908 | 0.059 |
| regression | concreate | p25 | alternative | 8 | 11.370 | 28.977 | 0.694 | 0.013 |
| regression | concreate | p25 | factual | 8 | 7.268 | 6.643 | 0.897 | 0.046 |
| regression | concreate | p50 | alternative | 8 | 11.532 | 27.947 | 0.685 | 0.014 |
| regression | concreate | p50 | factual | 8 | 7.267 | 6.613 | 0.896 | 0.046 |
| regression | concreate | p75 | alternative | 8 | 11.078 | 28.256 | 0.686 | 0.018 |
| regression | concreate | p75 | factual | 8 | 7.059 | 6.598 | 0.895 | 0.043 |
| regression | concreate | plain | alternative | 8 | 12.896 | 36.412 | 0.737 | 0.010 |
| regression | concreate | plain | factual | 8 | 7.963 | 7.009 | 0.901 | 0.039 |
| regression | cooling | p25 | alternative | 8 | 19.831 | 17.732 | 0.643 | 0.012 |
| regression | cooling | p25 | factual | 8 | 7.804 | 6.524 | 0.925 | 0.051 |
| regression | cooling | p50 | alternative | 8 | 20.214 | 17.801 | 0.646 | 0.010 |
| regression | cooling | p50 | factual | 8 | 7.770 | 6.732 | 0.929 | 0.050 |
| regression | cooling | p75 | alternative | 8 | 19.081 | 17.272 | 0.641 | 0.009 |
| regression | cooling | p75 | factual | 8 | 7.686 | 6.516 | 0.925 | 0.050 |
| regression | cooling | plain | alternative | 8 | 19.564 | 22.129 | 0.686 | 0.009 |
| regression | cooling | plain | factual | 8 | 8.000 | 5.990 | 0.917 | 0.050 |
| regression | deltaA | p25 | alternative | 5 | 7.984 | 17.307 | 0.623 | 0.023 |
| regression | deltaA | p25 | factual | 5 | 4.291 | 3.926 | 0.870 | 0.076 |
| regression | deltaA | p50 | alternative | 5 | 7.983 | 16.839 | 0.617 | 0.021 |
| regression | deltaA | p50 | factual | 5 | 4.323 | 3.944 | 0.869 | 0.079 |
| regression | deltaA | p75 | alternative | 5 | 7.687 | 16.746 | 0.612 | 0.029 |
| regression | deltaA | p75 | factual | 5 | 4.280 | 3.941 | 0.869 | 0.084 |
| regression | deltaA | plain | alternative | 5 | 9.301 | 21.908 | 0.677 | 0.012 |
| regression | deltaA | plain | factual | 5 | 4.873 | 4.352 | 0.880 | 0.071 |
| regression | deltaE | p25 | alternative | 6 | 9.221 | 20.697 | 0.746 | 0.030 |
| regression | deltaE | p25 | factual | 6 | 4.980 | 4.476 | 0.857 | 0.098 |
| regression | deltaE | p50 | alternative | 6 | 8.474 | 19.016 | 0.734 | 0.038 |
| regression | deltaE | p50 | factual | 6 | 4.722 | 4.349 | 0.853 | 0.101 |
| regression | deltaE | p75 | alternative | 6 | 8.474 | 19.016 | 0.734 | 0.038 |
| regression | deltaE | p75 | factual | 6 | 4.722 | 4.349 | 0.853 | 0.101 |
| regression | deltaE | plain | alternative | 6 | 11.420 | 27.962 | 0.790 | 0.007 |
| regression | deltaE | plain | factual | 6 | 5.941 | 5.032 | 0.870 | 0.094 |
| regression | friedm | p25 | alternative | 5 | 8.122 | 25.930 | 0.851 | 0.000 |
| regression | friedm | p25 | factual | 5 | 4.819 | 4.470 | 0.915 | 0.031 |
| regression | friedm | p50 | alternative | 5 | 8.024 | 25.143 | 0.847 | 0.000 |
| regression | friedm | p50 | factual | 5 | 4.840 | 4.476 | 0.915 | 0.031 |
| regression | friedm | p75 | alternative | 5 | 8.068 | 25.458 | 0.849 | 0.000 |
| regression | friedm | p75 | factual | 5 | 4.848 | 4.510 | 0.916 | 0.030 |
| regression | friedm | plain | alternative | 5 | 8.506 | 29.072 | 0.865 | 0.000 |
| regression | friedm | plain | factual | 5 | 4.998 | 4.582 | 0.917 | 0.028 |
| regression | heating | p25 | alternative | 8 | 20.333 | 18.226 | 0.635 | 0.010 |
| regression | heating | p25 | factual | 8 | 7.762 | 6.667 | 0.925 | 0.051 |
| regression | heating | p50 | alternative | 8 | 21.994 | 19.477 | 0.655 | 0.011 |
| regression | heating | p50 | factual | 8 | 7.919 | 7.139 | 0.930 | 0.050 |
| regression | heating | p75 | alternative | 8 | 20.986 | 18.876 | 0.649 | 0.009 |
| regression | heating | p75 | factual | 8 | 7.866 | 6.913 | 0.927 | 0.050 |
| regression | heating | plain | alternative | 8 | 19.564 | 22.191 | 0.685 | 0.009 |
| regression | heating | plain | factual | 8 | 8.000 | 6.002 | 0.914 | 0.050 |
| regression | housing | p25 | alternative | 9 | 11.000 | 25.840 | 0.801 | 0.010 |
| regression | housing | p25 | factual | 9 | 6.300 | 6.870 | 0.940 | 0.040 |
| regression | housing | p50 | alternative | 9 | 11.510 | 23.240 | 0.785 | 0.010 |
| regression | housing | p50 | factual | 9 | 6.000 | 6.770 | 0.939 | 0.040 |
| regression | housing | p75 | alternative | 9 | 12.060 | 21.190 | 0.767 | 0.020 |
| regression | housing | p75 | factual | 9 | 6.290 | 6.730 | 0.939 | 0.040 |
| regression | housing | plain | alternative | 9 | 13.160 | 31.660 | 0.830 | 0.010 |
| regression | housing | plain | factual | 9 | 7.270 | 8.070 | 0.948 | 0.040 |
| regression | kin8fh | p25 | alternative | 8 | 11.528 | 33.794 | 0.790 | 0.004 |
| regression | kin8fh | p25 | factual | 8 | 7.262 | 6.614 | 0.876 | 0.051 |
| regression | kin8fh | p50 | alternative | 8 | 11.030 | 33.152 | 0.790 | 0.004 |
| regression | kin8fh | p50 | factual | 8 | 6.968 | 6.576 | 0.876 | 0.053 |
| regression | kin8fh | p75 | alternative | 8 | 11.817 | 34.733 | 0.799 | 0.004 |
| regression | kin8fh | p75 | factual | 8 | 7.133 | 6.629 | 0.877 | 0.051 |
| regression | kin8fh | plain | alternative | 8 | 13.169 | 43.447 | 0.832 | 0.004 |
| regression | kin8fh | plain | factual | 8 | 7.926 | 7.049 | 0.884 | 0.044 |
| regression | kin8fm | p25 | alternative | 8 | 12.934 | 37.980 | 0.825 | 0.002 |
| regression | kin8fm | p25 | factual | 8 | 7.509 | 6.936 | 0.898 | 0.041 |
| regression | kin8fm | p50 | alternative | 8 | 12.711 | 36.376 | 0.819 | 0.002 |
| regression | kin8fm | p50 | factual | 8 | 7.411 | 6.828 | 0.896 | 0.037 |
| regression | kin8fm | p75 | alternative | 8 | 13.147 | 38.053 | 0.825 | 0.002 |
| regression | kin8fm | p75 | factual | 8 | 7.594 | 7.008 | 0.899 | 0.037 |
| regression | kin8fm | plain | alternative | 8 | 13.914 | 43.091 | 0.843 | 0.002 |
| regression | kin8fm | plain | factual | 8 | 7.916 | 7.169 | 0.901 | 0.033 |
| regression | kin8nh | p25 | alternative | 8 | 10.666 | 29.782 | 0.791 | 0.003 |
| regression | kin8nh | p25 | factual | 8 | 6.413 | 6.498 | 0.907 | 0.030 |
| regression | kin8nh | p50 | alternative | 8 | 10.419 | 27.997 | 0.780 | 0.004 |
| regression | kin8nh | p50 | factual | 8 | 6.227 | 6.354 | 0.905 | 0.034 |
| regression | kin8nh | p75 | alternative | 8 | 10.836 | 31.187 | 0.796 | 0.003 |
| regression | kin8nh | p75 | factual | 8 | 6.782 | 6.753 | 0.910 | 0.030 |
| regression | kin8nh | plain | alternative | 8 | 13.732 | 45.071 | 0.852 | 0.003 |
| regression | kin8nh | plain | factual | 8 | 7.979 | 7.323 | 0.917 | 0.024 |
| regression | kin8nm | p25 | alternative | 8 | 11.417 | 30.892 | 0.826 | 0.002 |
| regression | kin8nm | p25 | factual | 8 | 6.532 | 6.592 | 0.918 | 0.027 |
| regression | kin8nm | p50 | alternative | 8 | 11.347 | 31.689 | 0.830 | 0.001 |
| regression | kin8nm | p50 | factual | 8 | 6.716 | 6.801 | 0.920 | 0.023 |
| regression | kin8nm | p75 | alternative | 8 | 11.809 | 34.522 | 0.841 | 0.002 |
| regression | kin8nm | p75 | factual | 8 | 7.161 | 6.939 | 0.921 | 0.024 |
| regression | kin8nm | plain | alternative | 8 | 13.703 | 45.282 | 0.874 | 0.001 |
| regression | kin8nm | plain | factual | 8 | 7.937 | 7.381 | 0.925 | 0.022 |
| regression | laser | p25 | alternative | 4 | 7.003 | 11.713 | 0.456 | 0.044 |
| regression | laser | p25 | factual | 4 | 3.842 | 3.383 | 0.885 | 0.072 |
| regression | laser | p50 | alternative | 4 | 6.910 | 11.482 | 0.451 | 0.042 |
| regression | laser | p50 | factual | 4 | 3.794 | 3.347 | 0.884 | 0.071 |
| regression | laser | p75 | alternative | 4 | 7.098 | 10.983 | 0.438 | 0.048 |
| regression | laser | p75 | factual | 4 | 3.742 | 3.240 | 0.879 | 0.074 |
| regression | laser | plain | alternative | 4 | 7.364 | 13.622 | 0.491 | 0.038 |
| regression | laser | plain | factual | 4 | 4.000 | 3.544 | 0.890 | 0.067 |
| regression | mg | p25 | alternative | 6 | 9.800 | 16.259 | 0.441 | 0.008 |
| regression | mg | p25 | factual | 6 | 5.553 | 4.797 | 0.887 | 0.057 |
| regression | mg | p50 | alternative | 6 | 9.286 | 14.480 | 0.412 | 0.014 |
| regression | mg | p50 | factual | 6 | 5.331 | 4.528 | 0.879 | 0.067 |
| regression | mg | p75 | alternative | 6 | 9.730 | 15.897 | 0.435 | 0.010 |
| regression | mg | p75 | factual | 6 | 5.497 | 4.810 | 0.886 | 0.054 |
| regression | mg | plain | alternative | 6 | 10.486 | 20.289 | 0.492 | 0.001 |
| regression | mg | plain | factual | 6 | 5.949 | 5.301 | 0.896 | 0.048 |
| regression | mortage | p25 | alternative | 15 | 24.528 | 32.526 | 0.338 | 0.056 |
| regression | mortage | p25 | factual | 15 | 13.648 | 11.518 | 0.874 | 0.079 |
| regression | mortage | p50 | alternative | 15 | 24.426 | 31.847 | 0.334 | 0.056 |
| regression | mortage | p50 | factual | 15 | 14.000 | 11.988 | 0.879 | 0.079 |
| regression | mortage | p75 | alternative | 15 | 24.719 | 32.843 | 0.340 | 0.057 |
| regression | mortage | p75 | factual | 15 | 13.999 | 12.163 | 0.880 | 0.079 |
| regression | mortage | plain | alternative | 15 | 25.672 | 37.591 | 0.367 | 0.057 |
| regression | mortage | plain | factual | 15 | 14.479 | 12.729 | 0.884 | 0.078 |
| regression | plastic | p25 | alternative | 2 | 3.482 | 3.660 | 0.312 | 0.003 |
| regression | plastic | p25 | factual | 2 | 1.949 | 1.679 | 0.862 | 0.041 |
| regression | plastic | p50 | alternative | 2 | 3.533 | 3.759 | 0.318 | 0.000 |
| regression | plastic | p50 | factual | 2 | 1.966 | 1.703 | 0.864 | 0.039 |
| regression | plastic | p75 | alternative | 2 | 3.526 | 3.737 | 0.317 | 0.001 |
| regression | plastic | p75 | factual | 2 | 1.958 | 1.706 | 0.864 | 0.040 |
| regression | plastic | plain | alternative | 2 | 3.582 | 3.813 | 0.321 | 0.000 |
| regression | plastic | plain | factual | 2 | 2.000 | 1.714 | 0.864 | 0.040 |
| regression | puma8fh | p25 | alternative | 8 | 10.392 | 28.220 | 0.807 | 0.002 |
| regression | puma8fh | p25 | factual | 8 | 6.213 | 6.298 | 0.906 | 0.037 |
| regression | puma8fh | p50 | alternative | 8 | 10.108 | 22.943 | 0.773 | 0.002 |
| regression | puma8fh | p50 | factual | 8 | 5.356 | 5.760 | 0.897 | 0.049 |
| regression | puma8fh | p75 | alternative | 8 | 11.031 | 27.632 | 0.804 | 0.001 |
| regression | puma8fh | p75 | factual | 8 | 5.950 | 6.317 | 0.904 | 0.040 |
| regression | puma8fh | plain | alternative | 8 | 13.827 | 44.784 | 0.869 | 0.001 |
| regression | puma8fh | plain | factual | 8 | 7.948 | 7.310 | 0.917 | 0.031 |
| regression | puma8fm | p25 | alternative | 8 | 11.524 | 32.647 | 0.824 | 0.002 |
| regression | puma8fm | p25 | factual | 8 | 6.593 | 6.489 | 0.923 | 0.032 |
| regression | puma8fm | p50 | alternative | 8 | 11.857 | 33.222 | 0.826 | 0.001 |
| regression | puma8fm | p50 | factual | 8 | 6.917 | 6.839 | 0.926 | 0.022 |
| regression | puma8fm | p75 | alternative | 8 | 11.510 | 34.138 | 0.830 | 0.007 |
| regression | puma8fm | p75 | factual | 8 | 6.806 | 6.673 | 0.925 | 0.027 |
| regression | puma8fm | plain | alternative | 8 | 13.031 | 40.930 | 0.855 | 0.002 |
| regression | puma8fm | plain | factual | 8 | 7.570 | 7.299 | 0.931 | 0.018 |
| regression | puma8nh | p25 | alternative | 8 | 9.048 | 24.249 | 0.745 | 0.006 |
| regression | puma8nh | p25 | factual | 8 | 5.412 | 5.733 | 0.879 | 0.049 |
| regression | puma8nh | p50 | alternative | 8 | 9.223 | 21.730 | 0.726 | 0.006 |
| regression | puma8nh | p50 | factual | 8 | 5.256 | 5.671 | 0.881 | 0.052 |
| regression | puma8nh | p75 | alternative | 8 | 10.359 | 27.280 | 0.768 | 0.004 |
| regression | puma8nh | p75 | factual | 8 | 5.864 | 6.080 | 0.888 | 0.044 |
| regression | puma8nh | plain | alternative | 8 | 12.420 | 39.610 | 0.828 | 0.000 |
| regression | puma8nh | plain | factual | 8 | 7.576 | 6.999 | 0.900 | 0.042 |
| regression | puma8nm | p25 | alternative | 8 | 11.708 | 34.761 | 0.826 | 0.001 |
| regression | puma8nm | p25 | factual | 8 | 6.788 | 6.524 | 0.904 | 0.024 |
| regression | puma8nm | p50 | alternative | 8 | 11.448 | 36.056 | 0.831 | 0.006 |
| regression | puma8nm | p50 | factual | 8 | 6.923 | 6.464 | 0.903 | 0.024 |
| regression | puma8nm | p75 | alternative | 8 | 10.697 | 32.726 | 0.816 | 0.002 |
| regression | puma8nm | p75 | factual | 8 | 6.339 | 6.259 | 0.900 | 0.033 |
| regression | puma8nm | plain | alternative | 8 | 10.376 | 30.320 | 0.805 | 0.001 |
| regression | puma8nm | plain | factual | 8 | 6.086 | 6.073 | 0.898 | 0.029 |
| regression | quakes | p25 | alternative | 3 | 3.842 | 7.281 | 0.548 | 0.023 |
| regression | quakes | p25 | factual | 3 | 2.391 | 2.279 | 0.880 | 0.072 |
| regression | quakes | p50 | alternative | 3 | 4.239 | 7.973 | 0.587 | 0.011 |
| regression | quakes | p50 | factual | 3 | 2.577 | 2.430 | 0.888 | 0.057 |
| regression | quakes | p75 | alternative | 3 | 4.959 | 8.413 | 0.595 | 0.009 |
| regression | quakes | p75 | factual | 3 | 2.530 | 2.481 | 0.891 | 0.056 |
| regression | quakes | plain | alternative | 3 | 5.401 | 12.637 | 0.686 | 0.000 |
| regression | quakes | plain | factual | 3 | 3.000 | 2.699 | 0.900 | 0.050 |
| regression | stock | p25 | alternative | 9 | 14.392 | 20.331 | 0.360 | 0.030 |
| regression | stock | p25 | factual | 9 | 8.320 | 7.252 | 0.886 | 0.058 |
| regression | stock | p50 | alternative | 9 | 14.780 | 20.132 | 0.360 | 0.022 |
| regression | stock | p50 | factual | 9 | 8.426 | 6.927 | 0.881 | 0.058 |
| regression | stock | p75 | alternative | 9 | 14.901 | 20.313 | 0.363 | 0.022 |
| regression | stock | p75 | factual | 9 | 8.328 | 7.132 | 0.885 | 0.058 |
| regression | stock | plain | alternative | 9 | 15.690 | 24.930 | 0.406 | 0.017 |
| regression | stock | plain | factual | 9 | 8.919 | 7.877 | 0.895 | 0.046 |
| regression | treasury | p25 | alternative | 15 | 23.969 | 23.918 | 0.251 | 0.079 |
| regression | treasury | p25 | factual | 15 | 13.262 | 10.461 | 0.842 | 0.094 |
| regression | treasury | p50 | alternative | 15 | 24.857 | 25.474 | 0.263 | 0.078 |
| regression | treasury | p50 | factual | 15 | 14.142 | 11.567 | 0.856 | 0.094 |
| regression | treasury | p75 | alternative | 15 | 24.534 | 26.104 | 0.268 | 0.081 |
| regression | treasury | p75 | factual | 15 | 13.916 | 11.473 | 0.854 | 0.096 |
| regression | treasury | plain | alternative | 15 | 25.416 | 29.451 | 0.290 | 0.079 |
| regression | treasury | plain | factual | 15 | 14.238 | 12.046 | 0.859 | 0.094 |
| regression | wineRed | p25 | alternative | 11 | 16.071 | 31.369 | 0.678 | 0.044 |
| regression | wineRed | p25 | factual | 11 | 7.867 | 8.280 | 0.898 | 0.066 |
| regression | wineRed | p50 | alternative | 11 | 15.716 | 34.531 | 0.699 | 0.037 |
| regression | wineRed | p50 | factual | 11 | 8.439 | 8.809 | 0.904 | 0.058 |
| regression | wineRed | p75 | alternative | 11 | 15.716 | 34.531 | 0.699 | 0.037 |
| regression | wineRed | p75 | factual | 11 | 8.439 | 8.809 | 0.904 | 0.058 |
| regression | wineRed | plain | alternative | 11 | 19.863 | 55.147 | 0.786 | 0.030 |
| regression | wineRed | plain | factual | 11 | 10.936 | 10.042 | 0.915 | 0.056 |
| regression | wineWhite | p25 | alternative | 11 | 14.591 | 30.182 | 0.654 | 0.036 |
| regression | wineWhite | p25 | factual | 11 | 7.663 | 8.249 | 0.879 | 0.071 |
| regression | wineWhite | p50 | alternative | 11 | 14.198 | 30.711 | 0.663 | 0.042 |
| regression | wineWhite | p50 | factual | 11 | 7.964 | 8.452 | 0.883 | 0.073 |
| regression | wineWhite | p75 | alternative | 11 | 14.198 | 30.711 | 0.663 | 0.042 |
| regression | wineWhite | p75 | factual | 11 | 7.964 | 8.452 | 0.883 | 0.073 |
| regression | wineWhite | plain | alternative | 11 | 20.581 | 57.183 | 0.786 | 0.028 |
| regression | wineWhite | plain | factual | 11 | 10.972 | 9.871 | 0.899 | 0.067 |
| regression | wizmir | p25 | alternative | 9 | 14.430 | 30.570 | 0.606 | 0.044 |
| regression | wizmir | p25 | factual | 9 | 8.057 | 7.240 | 0.876 | 0.079 |
| regression | wizmir | p50 | alternative | 9 | 15.273 | 32.452 | 0.619 | 0.028 |
| regression | wizmir | p50 | factual | 9 | 8.546 | 7.626 | 0.881 | 0.073 |
| regression | wizmir | p75 | alternative | 9 | 14.446 | 30.112 | 0.601 | 0.026 |
| regression | wizmir | p75 | factual | 9 | 8.156 | 7.224 | 0.875 | 0.073 |
| regression | wizmir | plain | alternative | 9 | 14.597 | 34.084 | 0.630 | 0.026 |
| regression | wizmir | plain | factual | 9 | 8.314 | 7.431 | 0.879 | 0.073 |

## Execution summary

- Total datasets evaluated: 79
- Skipped (too small): 30
- Errors during execution: 1

**Seeds:** 3  **Cal sizes:** [100, 300, 500]  **k:** 5  **normalize_guard:** True  **merge_adjacent:** False

## Metric definitions

| Metric | Definition |
|---|---|
| `mean_standard_rules_per_instance` | Mean rule count from `explain_factual` / `explore_alternatives` |
| `mean_guarded_rules_per_instance` | Mean `intervals_emitted` per test instance |
| `guard_retention_rate` | `intervals_emitted / (intervals_emitted + guard_removed)` over factual bins only |
| `fraction_instances_fully_filtered` | Fraction of test instances with 0 guarded rules |
