EGM Analyzer — synthetic validation report
fs = 2000.0 Hz, T = 500.0 ms, N = 1000

[PASS] Vpp(unipolar) vs prescribed amplitude
       max |Vpp_computed − Vpp_true| (mV) = 0.000000e+00  (tolerance < 1.0e-02)
[PASS] Vpp(bipolar) vs analytical max-min
       max |Vpp_computed − Vpp_true| (mV) = 0.000000e+00  (tolerance < 1.0e-09)
[PASS] LAT vs analytical inflexion-point location
       max |LAT_computed − LAT_true| (ms) = 1.213203e-01  (tolerance < 5.0e-01)
[PASS] Vpp(omni) orientation-independence (rigid rotation)
       max |Vpp_omni(rot θ) − Vpp_omni(0)| (mV) = 4.440892e-16  (tolerance < 1.0e-02)
[PASS] ROR ≈ 0 for purely linear propagation
       |ROR| = 0.000000e+00  (tolerance < 1.0e-06)
[PASS] ROR ≈ 1 for purely circular propagation
       |ROR − 1| = 1.236173e-06  (tolerance < 5.0e-03)

Summary: 6/6 cases passed.
