# The Sharpe ratio haircut

Harvey and Liu (2015) turn a Sharpe ratio into a p-value, correct the p-value for the number of tests, and turn the result back into a Sharpe ratio. The difference is the haircut.

1. Correct the annual Sharpe ratio for autocorrelation, by Lo (2002).
2. Convert it to a t-statistic on monthly observations.
3. Take a two-sided p-value from a Student t with M − 1 degrees of freedom.
4. Adjust it for multiple testing.
5. Invert the adjusted p-value to a Sharpe ratio.

| Method | Controls | Adjustment |
|---|---|---|
| Bonferroni | Familywise error | p × n_tests, exact |
| Holm | Familywise error | Stepwise, simulated |
| BHY | False discovery rate | Stepwise, simulated |

Holm and BHY need the p-values of the other tests, which you rarely have. So they're simulated from the Harvey–Liu–Zhu model of published factors, with a chosen average correlation, and the median adjusted p-value is used.

The haircut is not a fixed percentage. In Harvey and Liu's example, a 1.0 annual Sharpe ratio over ten years, with autocorrelation 0.1 and 100 tests, loses 74.6% under Bonferroni, 71.3% under Holm and 52.0% under BHY. Strong Sharpe ratios lose little, and weak ones lose nearly everything, so the common rule of cutting by half is too harsh on the first and too kind to the second.

The DSR and the haircut correct for the same search in different ways. The DSR uses the spread of your actual trials. The haircut needs only the count, which makes it usable on a published number, at the price of assuming that the rest of your search looks like the published factor zoo.

The [calculator](../calculator.md) computes the Bonferroni haircut in the browser.
