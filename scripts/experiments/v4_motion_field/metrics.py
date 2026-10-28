"""World-metre vector metrics. No models, optimizers, or inferred GT identities."""
import numpy as np

EPS_M = 1e-6  # 0.001 mm: numerical direction-definition threshold, not contact/motion label.


def rms(x):
    return float(np.sqrt(np.mean(np.sum(np.asarray(x, dtype=np.float64)**2, axis=-1))))


def vector_comparison(normal, oracle, epsilon=EPS_M):
    n, o = np.asarray(normal, dtype=np.float64), np.asarray(oracle, dtype=np.float64)
    if n.shape != o.shape or n.ndim != 2 or n.shape[1] != 3 or not len(n):
        raise ValueError('Aligned nonempty [N,3] fields required')
    if not np.isfinite(n).all() or not np.isfinite(o).all():
        raise ValueError('Nonfinite motion')
    a, b = np.linalg.norm(n, axis=1), np.linalg.norm(o, axis=1)
    diff = np.linalg.norm(n-o, axis=1)
    defined = (a > epsilon) & (b > epsilon)
    cos = np.full(len(n), np.nan)
    cos[defined] = np.clip(np.sum(n[defined]*o[defined], axis=1)/(a[defined]*b[defined]), -1, 1)
    angles = np.degrees(np.arccos(cos))
    # Exact squared error decomposition, including zero vectors (direction term then zero).
    magnitude_energy = float(np.mean((a-b)**2))
    direction_energy = float(np.mean(np.maximum(2*(a*b-np.sum(n*o, axis=1)), 0)))
    total = float(np.mean(diff**2))
    mn, mo = n.mean(0), o.mean(0)
    out = dict(normal_rms_mm=rms(n)*1000, oracle_rms_mm=rms(o)*1000,
               difference_rms_mm=rms(n-o)*1000, difference_p95_mm=float(np.quantile(diff,.95))*1000,
               normal_over_oracle_rms=rms(n)/rms(o) if rms(o)>epsilon else None,
               defined_direction_count=int(defined.sum()), undefined_direction_count=int((~defined).sum()),
               direction_cosine_mean=float(cos[defined].mean()) if defined.any() else None,
               direction_angle_median_deg=float(np.median(angles[defined])) if defined.any() else None,
               direction_angle_p95_deg=float(np.quantile(angles[defined],.95)) if defined.any() else None,
               magnitude_error_energy_fraction=magnitude_energy/total if total>1e-24 else None,
               direction_error_energy_fraction=direction_energy/total if total>1e-24 else None,
               mean_translation_difference_mm=float(np.linalg.norm(mn-mo))*1000,
               normal_nonrigid_rms_mm=rms(n-mn)*1000, oracle_nonrigid_rms_mm=rms(o-mo)*1000,
               residual_difference_rms_mm=rms((n-mn)-(o-mo))*1000)
    for name, v in [('normal',mn),('oracle',mo)]:
        for axis,val in zip('xyz',v):out[f'{name}_mean_{axis}_mm']=float(val)*1000
    return out, dict(difference_mm=diff*1000, angle_deg=angles, direction_defined=defined)


def neighbor_metrics(initial, displacement, ids, valid):
    x, u = np.asarray(initial,dtype=np.float64), np.asarray(displacement,dtype=np.float64)
    d0=np.linalg.norm(x[:,None]-x[ids],axis=-1)
    valid=valid & (d0>1e-12)
    du=u[:,None]-u[ids]
    d1=np.linalg.norm((x+u)[:,None]-(x+u)[ids],axis=-1)
    ratio=d1[valid]/d0[valid]
    return dict(neighbor_motion_difference_rms_mm=rms(du[valid])*1000,
                neighbor_motion_difference_p95_mm=float(np.quantile(np.linalg.norm(du[valid],axis=-1),.95))*1000,
                neighbor_distance_change_rms_mm=float(np.sqrt(np.mean((d1[valid]-d0[valid])**2)))*1000,
                ratio_p05=float(np.quantile(ratio,.05)),ratio_p95=float(np.quantile(ratio,.95)),
                distortion_p95=float(np.quantile(abs(ratio-1),.95)),edge_count=int(valid.sum()))
