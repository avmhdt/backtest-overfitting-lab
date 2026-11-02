// Sharpe-haircut calculator. A port of the matching overfitlab functions, checked against
// them by js/test/sharpe_calc.test.mjs. No dependencies and no network calls.

const EULER_GAMMA = 0.5772156649015329;
const SQRT_2PI = Math.sqrt(2 * Math.PI);
const ALPHA = 0.05;

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];

function normPdf(x) {
  return Math.exp(-0.5 * x * x) / SQRT_2PI;
}

// Marsaglia's series near the centre, the continued fraction of the upper tail beyond it.
function normUpperTail(x) {
  let fraction = 0;
  for (let k = 200; k >= 1; k -= 1) {
    fraction = k / (x + fraction);
  }
  return normPdf(x) / (x + fraction);
}

export function normCdf(x) {
  if (Math.abs(x) >= 3) {
    return x > 0 ? 1 - normUpperTail(x) : normUpperTail(-x);
  }
  let term = x;
  let sum = x;
  for (let k = 1; k < 200 && Math.abs(term) > 1e-18; k += 1) {
    term *= (x * x) / (2 * k + 1);
    sum += term;
  }
  return 0.5 + normPdf(x) * sum;
}

function normLowerTail(z) {
  return z >= 3 ? normUpperTail(z) : normCdf(-z);
}

// Solves for the upper quantile from Abramowitz and Stegun 26.2.23 with Halley steps.
function upperQuantile(tail) {
  if (tail === 0.5) {
    return 0;
  }
  const lower = Math.min(tail, 1 - tail);
  const t = Math.sqrt(-2 * Math.log(lower));
  let z =
    t -
    (2.515517 + t * (0.802853 + t * 0.010328)) /
      (1 + t * (1.432788 + t * (0.189269 + t * 0.001308)));
  for (let k = 0; k < 6; k += 1) {
    const step = (normLowerTail(z) - lower) / normPdf(z);
    z += step / (1 - (z * step) / 2);
  }
  return tail < 0.5 ? z : -z;
}

export function normPpf(p) {
  if (!(p > 0 && p < 1)) {
    return Number.NaN;
  }
  return p < 0.5 ? -upperQuantile(p) : upperQuantile(1 - p);
}

function logGamma(x) {
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  const shifted = x - 1;
  const t = shifted + 7.5;
  let sum = LANCZOS[0];
  for (let k = 1; k < LANCZOS.length; k += 1) {
    sum += LANCZOS[k] / (shifted + k);
  }
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t) - t + Math.log(sum);
}

// Lentz's continued fraction for the incomplete beta function.
function betaFraction(x, a, b) {
  const tiny = 1e-300;
  let c = 1;
  let d = 1 - ((a + b) * x) / (a + 1);
  d = 1 / (Math.abs(d) < tiny ? tiny : d);
  let h = d;
  for (let m = 1; m <= 5000; m += 1) {
    const even = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m));
    d = 1 + even * d;
    d = 1 / (Math.abs(d) < tiny ? tiny : d);
    c = 1 + even / c;
    c = Math.abs(c) < tiny ? tiny : c;
    h *= d * c;
    const odd = (-(a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1));
    d = 1 + odd * d;
    d = 1 / (Math.abs(d) < tiny ? tiny : d);
    c = 1 + odd / c;
    c = Math.abs(c) < tiny ? tiny : c;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-16) {
      break;
    }
  }
  return h;
}

function regularizedBeta(x, a, b) {
  if (x <= 0) {
    return 0;
  }
  if (x >= 1) {
    return 1;
  }
  const front = Math.exp(
    logGamma(a + b) -
      logGamma(a) -
      logGamma(b) +
      a * Math.log(x) +
      b * Math.log1p(-x),
  );
  if (x < (a + 1) / (a + b + 2)) {
    return (front * betaFraction(x, a, b)) / a;
  }
  return 1 - (front * betaFraction(1 - x, b, a)) / b;
}

function studentUpperTail(df, t) {
  return 0.5 * regularizedBeta(df / (df + t * t), df / 2, 0.5);
}

export function studentCdf(df, t) {
  return t >= 0 ? 1 - studentUpperTail(df, t) : studentUpperTail(df, -t);
}

// Bisection on the upper tail, which keeps its accuracy for very small probabilities.
function studentUpperQuantile(df, tail) {
  if (tail >= 0.5) {
    return 0;
  }
  let high = 1;
  while (studentUpperTail(df, high) > tail) {
    high *= 2;
  }
  let low = 0;
  for (let k = 0; k < 200 && high - low > 4 * Number.EPSILON * high; k += 1) {
    const middle = 0.5 * (low + high);
    if (studentUpperTail(df, middle) > tail) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return 0.5 * (low + high);
}

export function studentPpf(df, p) {
  return p >= 0.5 ? studentUpperQuantile(df, 1 - p) : -studentUpperQuantile(df, p);
}

function sharpeStd(sr, nObs, skew, kurtosis) {
  const term = 1 - skew * sr + ((kurtosis - 1) / 4) * sr * sr;
  return term > 0 ? Math.sqrt(term / (nObs - 1)) : null;
}

function maxOrderStatistic(nTrials) {
  if (nTrials === 1) {
    return 0;
  }
  return (
    (1 - EULER_GAMMA) * upperQuantile(1 / nTrials) +
    EULER_GAMMA * upperQuantile(1 / (nTrials * Math.E))
  );
}

function ar1Scale(rho, q) {
  const term = 1 + ((2 * rho) / (1 - rho)) * (1 - (1 - rho ** q) / (q * (1 - rho)));
  return term ** -0.5;
}

function bonferroniHaircut(srAnnual, nObs, q, nTests, autocorr) {
  const months = Math.floor((nObs * 12) / q);
  if (months < 2) {
    return null;
  }
  const corrected = autocorr === 0 || q === 1 ? srAnnual : srAnnual * ar1Scale(autocorr, q);
  if (corrected <= 0) {
    return { srAnnual: 0, pct: 100 };
  }
  const tstat = (corrected / Math.sqrt(12)) * Math.sqrt(months);
  const df = months - 1;
  const adjusted = Math.min(nTests * 2 * studentUpperTail(df, tstat), 1);
  const tail = adjusted > 0 ? Math.min(tstat, studentUpperQuantile(df, adjusted / 2)) : tstat;
  const haircutAnnual = (tail / Math.sqrt(months)) * Math.sqrt(12);
  return { srAnnual: haircutAnnual, pct: (100 * (corrected - haircutAnnual)) / corrected };
}

function inputProblems(input, nObs) {
  const problems = [];
  const finite = (name) => Number.isFinite(input[name]);
  const names = ["sr_annual", "periods_per_year", "skew", "kurtosis"];
  names.push("n_trials", "var_trials_annual", "n_tests", "autocorr");
  if (!names.every(finite) || !Number.isFinite(nObs)) {
    problems.push("Fill in every field with a number.");
    return problems;
  }
  if (input.periods_per_year <= 0) {
    problems.push("Frequency must be positive.");
  }
  if (nObs < 2) {
    problems.push("The sample needs at least two observations.");
  }
  if (input.kurtosis < 1) {
    problems.push("Kurtosis is non-excess, so it can't be below 1. A normal distribution has 3.");
  }
  if (input.n_trials < 1) {
    problems.push("Trials must be at least 1.");
  }
  if (input.var_trials_annual < 0) {
    problems.push("The variance of trial Sharpe ratios can't be negative.");
  }
  if (input.n_tests < 1 || !Number.isInteger(input.n_tests)) {
    problems.push("Tests must be a whole number of at least 1.");
  }
  if (Math.abs(input.autocorr) >= 1) {
    problems.push("Autocorrelation must be between -1 and 1.");
  }
  return problems;
}

// Inputs are annual: `sr_annual`, `var_trials_annual`, and `years` or `n_obs`.
export function calculate(input) {
  const q = input.periods_per_year;
  const nObs = input.n_obs ?? Math.round(input.years * q);
  const empty = {
    psr0: null,
    sr0Annual: null,
    dsr: null,
    minTrlObs: null,
    minTrlYears: null,
    minBtlYears: null,
    haircutSrAnnual: null,
    haircutPct: null,
  };
  const problems = inputProblems(input, nObs);
  if (problems.length > 0) {
    return { ...empty, problems };
  }

  const out = { ...empty, problems };
  const { skew, kurtosis, n_trials: nTrials } = input;
  const sr = input.sr_annual / Math.sqrt(q);
  const maxStat = maxOrderStatistic(nTrials);
  out.sr0Annual = Math.sqrt(input.var_trials_annual) * maxStat;

  const std = sharpeStd(sr, nObs, skew, kurtosis);
  if (std === null) {
    problems.push("This Sharpe ratio, skew and kurtosis give a negative variance.");
  } else {
    out.psr0 = normCdf(sr / std);
    out.dsr = normCdf((sr - out.sr0Annual / Math.sqrt(q)) / std);
    if (sr > 0) {
      const z = upperQuantile(ALPHA);
      out.minTrlObs = 1 + (std * std * (nObs - 1) * z * z) / (sr * sr);
      out.minTrlYears = out.minTrlObs / q;
    }
  }
  if (input.sr_annual > 0) {
    out.minBtlYears = (maxStat / input.sr_annual) ** 2;
  }
  const haircut = bonferroniHaircut(input.sr_annual, nObs, q, input.n_tests, input.autocorr);
  if (haircut === null) {
    problems.push("The haircut needs a sample of at least two months.");
  } else {
    out.haircutSrAnnual = haircut.srAnnual;
    out.haircutPct = haircut.pct;
  }
  return out;
}

const FORMATS = {
  psr0: (v) => v.toFixed(4),
  sr0Annual: (v) => v.toFixed(3),
  dsr: (v) => v.toFixed(4),
  minTrlYears: (v) => `${v.toFixed(1)} years`,
  minBtlYears: (v) => `${v.toFixed(1)} years`,
  haircutSrAnnual: (v) => v.toFixed(3),
  haircutPct: (v) => `${v.toFixed(1)}%`,
};

function readForm(form) {
  const number = (name) => {
    const text = form.elements[name].value.trim();
    return text === "" ? Number.NaN : Number(text);
  };
  const input = { periods_per_year: number("periods_per_year") };
  for (const name of ["sr_annual", "skew", "kurtosis", "n_trials", "var_trials_annual"]) {
    input[name] = number(name);
  }
  input.n_tests = number("n_tests");
  input.autocorr = number("autocorr");
  input[form.elements.length_unit.value] = number("sample_length");
  return input;
}

function render(form) {
  const result = calculate(readForm(form));
  for (const [key, format] of Object.entries(FORMATS)) {
    const target = form.querySelector(`[data-metric="${key}"]`);
    target.textContent = result[key] === null ? "n/a" : format(result[key]);
  }
  form.querySelector("[data-problems]").textContent = result.problems.join(" ");
}

if (typeof document !== "undefined") {
  const form = document.getElementById("sharpe-calc");
  if (form) {
    form.addEventListener("input", () => render(form));
    form.addEventListener("submit", (event) => event.preventDefault());
    render(form);
  }
}
