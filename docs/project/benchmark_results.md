# Benchmark synthétique FFT vs wavelet (WBS 6.2.3)

Ce rapport versionné documente une comparaison rapide entre FFT et ondelettes,
ainsi qu'entre LDA/Logistic et le classifieur centroid maison, sur un dataset
synthétique équilibré.

| features | classifier | accuracy | train_s | predict_s |
|---|---|---|---|---|
| fft | lda | 1.000 | 0.0067 | 0.000076 |
| fft | logistic | 1.000 | 0.0181 | 0.000098 |
| fft | centroid | 1.000 | 0.0018 | 0.000036 |
| wavelet | lda | 1.000 | 0.0750 | 0.003634 |
| wavelet | logistic | 1.000 | 0.0727 | 0.001370 |
| wavelet | centroid | 1.000 | 0.1038 | 0.001249 |
