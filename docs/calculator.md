# Sharpe-haircut calculator

Enter a backtest and the search behind it. The results update as you type, and nothing leaves your browser.

<form id="sharpe-calc" class="calc" autocomplete="off">
  <fieldset>
    <legend>The backtest</legend>
    <div class="calc-grid">
      <label>Annual Sharpe ratio
        <input name="sr_annual" type="number" step="any" value="2.5">
      </label>
      <label>Length
        <input name="sample_length" type="number" step="any" min="0" value="5">
      </label>
      <label>Length is in
        <select name="length_unit">
          <option value="years" selected>Years</option>
          <option value="n_obs">Observations</option>
        </select>
      </label>
      <label>Frequency
        <select name="periods_per_year">
          <option value="252" selected>Daily, 252 per year</option>
          <option value="365">Daily, 365 per year</option>
          <option value="52">Weekly</option>
          <option value="12">Monthly</option>
        </select>
      </label>
      <label>Skew
        <input name="skew" type="number" step="any" value="-3">
      </label>
      <label>Kurtosis (normal = 3)
        <input name="kurtosis" type="number" step="any" min="1" value="10">
      </label>
      <label>Autocorrelation
        <input name="autocorr" type="number" step="any" min="-1" max="1" value="0">
      </label>
    </div>
  </fieldset>
  <fieldset>
    <legend>The search</legend>
    <div class="calc-grid">
      <label>Trials, N
        <input name="n_trials" type="number" step="any" min="1" value="100">
      </label>
      <label>Variance of trial Sharpe ratios, annual
        <input name="var_trials_annual" type="number" step="any" min="0" value="0.5">
      </label>
      <label>Number of tests
        <input name="n_tests" type="number" step="1" min="1" value="100">
      </label>
    </div>
  </fieldset>
  <noscript>This calculator needs JavaScript.</noscript>
  <p class="calc-problems" data-problems role="status"></p>
  <table aria-live="polite">
    <caption>Results</caption>
    <thead>
      <tr><th scope="col">Measure</th><th scope="col">Value</th><th scope="col">Meaning</th></tr>
    </thead>
    <tbody>
      <tr><th scope="row">PSR(0)</th><td><output data-metric="psr0"></output></td><td>Chance the true Sharpe ratio is above 0, with no allowance for the search.</td></tr>
      <tr><th scope="row">SR₀, annual</th><td><output data-metric="sr0Annual"></output></td><td>The best Sharpe ratio to expect from N trials with no edge.</td></tr>
      <tr><th scope="row">DSR</th><td><output data-metric="dsr"></output></td><td>Chance the true Sharpe ratio is above SR₀. The usual bar is 0.95.</td></tr>
      <tr><th scope="row">MinTRL</th><td><output data-metric="minTrlYears"></output></td><td>Track record needed for PSR(0) to reach 0.95.</td></tr>
      <tr><th scope="row">MinBTL</th><td><output data-metric="minBtlYears"></output></td><td>Backtest length at which N trials with no edge stop reaching this Sharpe ratio.</td></tr>
      <tr><th scope="row">Haircut Sharpe, annual</th><td><output data-metric="haircutSrAnnual"></output></td><td>The Sharpe ratio left after a Bonferroni correction.</td></tr>
      <tr><th scope="row">Haircut</th><td><output data-metric="haircutPct"></output></td><td>The share of the Sharpe ratio lost.</td></tr>
    </tbody>
  </table>
</form>

## Notes

Autocorrelation enters only the haircut, by Lo (2002). The first five measures take the Sharpe ratio as given.

The haircut uses monthly observations, as Harvey and Liu do, so the sample needs at least two whole months.

Variance is of the annual Sharpe ratios across your trials. With a trial log, `overfitlab trials report` gives it to you.

Python and this page agree within 1e-6 on every case in the test fixture. Holm and BHY need simulation, so they're in the [Python library](api.md#haircut) and not here. The theory is on the [haircut](theory/haircut.md) and [DSR](theory/psr-dsr.md) pages.
