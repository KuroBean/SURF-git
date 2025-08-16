import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

# --- helpers ---
def _smooth(y, window_samples=11, polyorder=2):
    window_samples = max(3, int(window_samples) | 1)
    polyorder = min(polyorder, window_samples - 2)
    return savgol_filter(y, window_samples, polyorder)

def _auto_polarity(sig):
    return np.abs(np.min(sig)) > np.abs(np.max(sig))  # True if negative-going dominates

def _baseline(time, sig, window_s=5e-4):
    t0 = time[0]
    m = (time - t0) <= window_s
    if not np.any(m):
        m = slice(0, max(5, min(50, len(sig)//20)))
    return float(np.mean(sig[m]))

def _sech2_model(t, c, a, d):
    # y(t) = 0.5*c*sech^2( sqrt(c/2)*(t-a) ) + d
    z = np.sqrt(np.maximum(c, 1e-16)/2.0) * (t - a)
    sech = 1.0/np.cosh(z)
    return 0.5*c*(sech**2) + d

def _fit_sech2(time, sig, a_guess, d_fixed, fit_half_window_s=1e-4):
    # restrict to a local window around peak
    lo, hi = a_guess - fit_half_window_s, a_guess + fit_half_window_s
    m = (time >= lo) & (time <= hi)
    T, Y = time[m], sig[m]
    if len(T) < 5:
        raise RuntimeError("Not enough samples in fit window; increase fit_half_window_s.")

    # amplitude + width guesses
    A0 = max(float(np.max(Y) - d_fixed), 1e-6)  # ensure positive
    width = max((T[-1] - T[0]) / 10.0, 1e-6)
    # FWHM ≈ 1.763/k, k = sqrt(c/2) ⇒ c ≈ 2*(1.763/width)^2
    c0_shape = 2.0 * (1.763/width)**2
    # but peak amplitude ≈ 0.5*c ⇒ match scale loosely
    c0_amp = 2.0 * A0
    c0 = max(min(c0_shape, 1e6), 1e-6)  # keep sane
    c0 = max(min(0.5*(c0 + c0_amp), 1e12), 1e-12)  # blend guesses, clamp

    # initial a from detected peak (requested change)
    a0 = float(a_guess)

    # fit with d fixed: wrap model
    def f_wrap(t, c, a):
        return _sech2_model(t, c, a, d_fixed)

    # bounds: positive c, a within a generous window
    pad = 2*fit_half_window_s
    bounds = ([1e-12, T[0]-pad], [1e12, T[-1]+pad])

    popt, pcov = curve_fit(f_wrap, T, Y, p0=[c0, a0], bounds=bounds, maxfev=50000)
    c_hat, a_hat = popt
    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf)) if np.ndim(pcov)==2 else [np.nan, np.nan]
    uc, ua = float(perr[0]), float(perr[1])

    Yfit = f_wrap(T, c_hat, a_hat)
    return {
        "c": float(c_hat), "u_c": uc,
        "a": float(a_hat), "u_a": ua,
        "d": float(d_fixed),
        "T": T, "Y": Y, "Yfit": Yfit
    }

# --- main ---
def compute_pulse_velocity_sech2(
    csv_path,
    distance,
    sep=",", skiprows=0,
    time_col_index=0, ch1_col_index=1, ch2_col_index=2,
    smooth=True, smooth_window_samples=15, smooth_polyorder=2,
    prominence=0.05, height=None,
    fit_half_window_s=1e-4, baseline_window_s=5e-4,
    plot=True
):
    """
    Velocity via sech^2 fits around high-prominence peaks.
    Now: a_guess is the detected peak time; returns all fit params with 1σ.
    """
    df = pd.read_csv(csv_path, sep=sep, skiprows=skiprows, header=None)
    t  = pd.to_numeric(df.iloc[:, time_col_index], errors="coerce").to_numpy()
    v1 = pd.to_numeric(df.iloc[:, ch1_col_index], errors="coerce").to_numpy()
    v2 = pd.to_numeric(df.iloc[:, ch2_col_index], errors="coerce").to_numpy()

    good = np.isfinite(t) & np.isfinite(v1) & np.isfinite(v2)
    t, v1, v2 = t[good], v1[good], v2[good]
    order = np.argsort(t)
    t, v1, v2 = t[order], v1[order], v2[order]
    if len(t) < 5:
        raise ValueError("Too few samples.")

    v1p = _smooth(v1, smooth_window_samples, smooth_polyorder) if smooth else v1
    v2p = _smooth(v2, smooth_window_samples, smooth_polyorder) if smooth else v2

    # polarity for detection only
    s1 = -v1p if _auto_polarity(v1p) else v1p
    s2 = -v2p if _auto_polarity(v2p) else v2p

    # baselines (on original)
    d1 = _baseline(t, v1, baseline_window_s)
    d2 = _baseline(t, v2, baseline_window_s)

    # find earliest high-prominence peak on each channel
    pk1, _ = find_peaks(s1, prominence=prominence, height=height)
    pk2, _ = find_peaks(s2, prominence=prominence, height=height)
    if len(pk1)==0 or len(pk2)==0:
        raise RuntimeError("No peaks found. Increase smoothing or adjust 'prominence'/'height'.")

    a1_guess = float(t[int(pk1[0])])
    a2_guess = float(t[int(pk2[0])])

    # fit around each peak using the identified peak time as a0
    fit1 = _fit_sech2(t, v1, a_guess=a1_guess, d_fixed=d1, fit_half_window_s=fit_half_window_s)
    fit2 = _fit_sech2(t, v2, a_guess=a2_guess, d_fixed=d2, fit_half_window_s=fit_half_window_s)

    a1, ua1, c1, uc1, d1_fit = fit1["a"], fit1["u_a"], fit1["c"], fit1["u_c"], fit1["d"]
    a2, ua2, c2, uc2, d2_fit = fit2["a"], fit2["u_a"], fit2["c"], fit2["u_c"], fit2["d"]

    delta_t  = a2 - a1
    if delta_t <= 0:
        raise RuntimeError(f"Non-positive Δt ({delta_t}). Check channel ordering or detection settings.")
    u_delta_t = float(np.sqrt(ua1**2 + ua2**2))

    velocity   = distance / delta_t
    u_velocity = float(np.abs(distance) * u_delta_t / (delta_t**2))

    result = {
        "channel_1": {"c": c1, "u_c": uc1, "a": a1, "u_a": ua1, "d": d1_fit},
        "channel_2": {"c": c2, "u_c": uc2, "a": a2, "u_a": ua2, "d": d2_fit},
        "delta_t": delta_t, "u_delta_t": u_delta_t,
        "velocity": velocity, "u_velocity": u_velocity,
        "fit1": fit1, "fit2": fit2
    }

    # summary printout
    print("Channel 1 fit: c = {:.6e} ± {:.1e}, a = {:.9e} ± {:.1e}, d = {:.6e}".format(c1, uc1, a1, ua1, d1_fit))
    print("Channel 2 fit: c = {:.6e} ± {:.1e}, a = {:.9e} ± {:.1e}, d = {:.6e}".format(c2, uc2, a2, ua2, d2_fit))
    print("Δt = {:.9e} ± {:.1e} s".format(delta_t, u_delta_t))
    print("v  = {:.6g} ± {:.2g} (units/s)".format(velocity, u_velocity))

    if plot:
        # overlay local fits for both channels
        for label, sig, fit in [("ch1", v1, fit1), ("ch2", v2, fit2)]:
            T, Y, Yfit = fit["T"], fit["Y"], fit["Yfit"]
            a = fit["a"]; d = fit["d"]
            lo, hi = a - 3*fit_half_window_s, a + 3*fit_half_window_s
            m = (t >= lo) & (t <= hi)
            plt.figure(figsize=(8,4))
            plt.plot(t[m], sig[m], label=f"{label} window", alpha=0.6)
            plt.scatter(T, Y, s=12, label="fit samples")
            plt.plot(T, Yfit, label="sech² fit", linewidth=2)
            plt.axvline(a, color="red", linestyle="--", label=f"a = {a:.6e} ± {fit['u_a']:.1e}s")
            plt.axhline(d, color="gray", linestyle=":", label="d (baseline)")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage")
            plt.title(f"{label}: sech² fit around detected peak")
            plt.legend()
            plt.tight_layout()
            plt.show()

    return result


# ---------------- Example usage ----------------
res = compute_pulse_velocity_sech2(
    r'.\1D chain tension changing exp\10N\scope_12.csv', 
    distance=0.018*8,
    sep=",",
    skiprows=0,
    smooth=True,
    smooth_window_samples=15,
    smooth_polyorder=2,
    prominence=0.02,       # set this high to isolate the main pulse
    height=None,           # optionally require an absolute height
    fit_half_window_s=3e-4,
    baseline_window_s=5e-4,
    plot=True
)
