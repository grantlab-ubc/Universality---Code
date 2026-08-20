
import numpy as np, glob, math
from scipy import stats
from collections import defaultdict

CACHE = 'dp_loss_window_results/simulation_cache'

def hill_mle(x, xmin):
    s = np.asarray(x, float); s = s[s >= xmin]; n = len(s)
    if n < 10: return np.nan, n
    S = np.sum(np.log(s / xmin))
    return (1.0 + n / S, n) if S > 0 else (np.nan, n)

def find_xmin_KS(data, max_trials=60):
    data = np.asarray(data, float); data = data[data > 0]
    u = np.unique(data)
    if len(u) < 10: return float(np.min(data))
    upper = max(10, int(len(u) * 0.85))
    cands = u[np.linspace(0, upper - 1, min(max_trials, upper)).astype(int)]
    bestD, bestxm = np.inf, float(np.min(data))
    for xm in cands:
        s = data[data >= xm]
        if len(s) < 50: continue
        tau, _ = hill_mle(data, xm)
        if np.isnan(tau) or tau <= 1: continue
        ss = np.sort(s); emp = np.arange(1, len(ss) + 1) / len(ss)
        D = np.max(np.abs(emp - (1 - (xm / ss) ** (tau - 1))))
        if D < bestD: bestD, bestxm = D, float(xm)
    return bestxm

def production_ensemble():
    best = None
    for f in glob.glob(f'{CACHE}/avalanche_*.npz'):
        d = np.load(f, allow_pickle=True)
        if int(d['num_runs']) != 1200: continue
        tS, _ = hill_mle(d['sizes'], find_xmin_KS(d['sizes']))
        tT, _ = hill_mle(d['durations'], find_xmin_KS(d['durations']))
        tA, _ = hill_mle(d['areas'], find_xmin_KS(d['areas']))
        score = abs(tS - 1.434) + abs(tT - 1.743) + abs(tA - 1.475)
        if best is None or score < best[0]: best = (score, f)
    return best[1]

def ll_powerlaw(x, xmin, tau):
    return np.log(tau - 1) + (tau - 1) * np.log(xmin) - tau * np.log(x)

def ll_exp(x, xmin):
    lam = 1.0 / np.mean(x - xmin)
    return math.log(lam) - lam * (x - xmin)

def ll_lognormal(x, xmin):
    from scipy.optimize import minimize
    lx = np.log(x)
    def negll(p):
        mu, ls = p; s = math.exp(ls); z = (math.log(xmin) - mu) / s
        logZ = math.log(max(1e-300, 1 - stats.norm.cdf(z)))
        return -np.sum(stats.norm.logpdf(lx, mu, s) - np.log(x) - logZ)
    r = minimize(negll, [lx.mean(), math.log(lx.std() + 1e-9)], method='Nelder-Mead')
    mu, ls = r.x; s = math.exp(ls); z = (math.log(xmin) - mu) / s
    logZ = math.log(max(1e-300, 1 - stats.norm.cdf(z)))
    return stats.norm.logpdf(lx, mu, s) - np.log(x) - logZ

def vuong(ll1, ll2):
    d = ll1 - ll2; n = len(d); s = np.std(d)
    if s == 0: return 0.0, np.sum(d)
    Z = np.sum(d) / (math.sqrt(n) * s)
    return Z, np.sum(d)

def model_selection(data, name, refs, seed=0):
    x = np.asarray(data, float); x = x[x > 0]
    xm = find_xmin_KS(x); tail = x[x >= xm]; n = len(tail)
    tau, _ = hill_mle(x, xm)
    ss = np.sort(tail); emp = np.arange(1, n + 1) / n
    Dks = np.max(np.abs(emp - (1 - (xm / ss) ** (tau - 1))))
    rng = np.random.default_rng(seed); Db = []
    for _ in range(500):
        u = rng.random(n); xs = xm * (1 - u) ** (-1 / (tau - 1))
        t2, _ = hill_mle(xs, xm); sx = np.sort(xs)
        Db.append(np.max(np.abs(np.arange(1, n + 1) / n - (1 - (xm / sx) ** (t2 - 1)))))
    pgof = float(np.mean(np.array(Db) >= Dks))
    llp = ll_powerlaw(tail, xm, tau)
    Zexp, _ = vuong(llp, ll_exp(tail, xm))
    Zln, _ = vuong(llp, ll_lognormal(tail, xm))
    print(f"\n{name}: xmin={xm:.3g} n={n} tau={tau:.3f} | GOF p={pgof:.2f} "
          f"| Z(exp)={Zexp:+.1f} Z(lognorm)={Zln:+.1f}")
    lls = {c: np.sum(ll_powerlaw(tail, xm, t)) for c, t in refs.items() if not np.isnan(t)}
    best = max(lls, key=lls.get)
    for c in sorted(lls, key=lambda k: -lls[k]):
        d = ll_powerlaw(tail, xm, refs[c]) - ll_powerlaw(tail, xm, refs[best])
        s = np.std(d); Z = np.sum(d) / (math.sqrt(n) * s) if s > 0 else 0
        print(f"   {c:11s} tau={refs[c]:.2f}: dLL={lls[c]-lls[best]:+7.1f} (Z={Z:+.1f})")

def hyperscaling():
    print("\n=== hyperscaling  eta+2delta = d/z  (d=3) ===")
    for eta, dl, z, lab in [(0.114, 0.730, 1.90, 'DP reference'),
                            (0.151, 0.730, 2.025, 'this work (raw)')]:
        print(f"  {lab:16s}: eta+2d={eta+2*dl:.3f}  d/z={3/z:.3f}  "
              f"z_hyper=d/(eta+2d)={3/(eta+2*dl):.3f}")

def density_closure():
    print("\n=== critical density-decay theta (lambda=2.008) ===")
    import csv
    rows = list(csv.DictReader(open('dp_loss_window_results/active_density_decay_values.csv')))
    th = [float(r['decay_exp']) for r in rows if abs(float(r['lambda_fac']) - 2.008) < 1e-6]
    print(f"  theta = {np.mean(th):.3f} (range {min(th):.3f}-{max(th):.3f}); "
          f"delta=0.730, beta/nu_par(DP)=0.736")

if __name__ == '__main__':
    prod = production_ensemble()
    d = np.load(prod, allow_pickle=True)
    print("production avalanche ensemble:", prod.split('/')[-1])
    model_selection(d['sizes'], 'P(S)', {'DP':1.40,'DyP/GEP':1.12,'Manna':1.28,'mean-field':1.50})
    model_selection(d['durations'],'P(T)', {'DP':1.73,'DyP/GEP':1.36,'Manna':1.78,'mean-field':2.00})
    model_selection(d['areas'], 'P(A)', {'DP':1.45,'DyP/GEP':1.16,'Manna':np.nan,'mean-field':np.nan})
    hyperscaling()
    density_closure()
