import numpy as np

from .odl_config import DEFAULT_DEVICE

def solve_linear_system(A, b):

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()

    rs = np.sqrt((A ** 2).sum(axis=1))
    rs[(rs == 0) | ~np.isfinite(rs)] = 1.0
    cs = np.sqrt((A ** 2).sum(axis=0))
    cs[(cs == 0) | ~np.isfinite(cs)] = 1.0

    As = (A / rs[:, None]) / cs[None, :]
    bs = b / rs

    with np.errstate(all="ignore"):
        try:
            y = np.linalg.solve(As, bs)
        except np.linalg.LinAlgError:
            y = np.linalg.lstsq(As, bs, rcond=None)[0]
    if not np.all(np.isfinite(y)):
        y = np.linalg.lstsq(As, bs, rcond=None)[0]

    return y / cs

def solve_linear_static_direct(system):

    K = np.array(system["KL_red"], dtype=float)

    if np.linalg.norm(K - K.T, "fro") <= 1.0e-10 * max(np.linalg.norm(K, "fro"), 1.0):
        K = 0.5 * (K + K.T)
    f = np.asarray(system["f_red"], dtype=float).ravel()

    displacement_red = solve_linear_system(K, f)
    displacement_full = system["transform"] @ displacement_red
    residual_norm = float(np.linalg.norm(K @ displacement_red - f))

    return {
        "displacement_red": displacement_red,
        "displacement_full": displacement_full,
        "residual_norm": residual_norm,
    }

def lm_optimize(residual_fn, x0, max_iter=300, tol=1.0e-8,
                damping_init=1.0e-3, damping_factor=10.0,
                damping_min=1.0e-10, damping_max=1.0e10,
                verbose=False):
    import torch
    torch.set_default_dtype(torch.float64)

    x = x0.clone().detach()
    device = x.device
    n = x.numel()
    eye = torch.eye(n, device=device)

    damping = damping_init
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    history = []

    it = 0
    for it in range(max_iter):
        F = residual_fn(x).detach()
        current_loss = 0.5 * float(torch.dot(F, F))
        history.append(current_loss)

        if float(torch.norm(F)) < tol:
            break

        J = torch.autograd.functional.jacobian(residual_fn, x, vectorize=True).detach()
        JTr = J.T @ F
        if float(torch.norm(JTr)) < tol * 10:
            break
        JTJ = J.T @ J

        step_accepted = False
        for _attempt in range(10):
            H = JTJ + damping * eye
            try:
                delta = torch.linalg.solve(H, -JTr)
            except Exception:
                damping = min(damping * damping_factor, damping_max)
                continue

            x_new = x + delta
            F_new = residual_fn(x_new).detach()
            new_loss = 0.5 * float(torch.dot(F_new, F_new))

            actual_reduction = current_loss - new_loss
            predicted_reduction = -0.5 * float(delta @ (2.0 * JTr + JTJ @ delta))
            rho = actual_reduction / predicted_reduction if predicted_reduction > 0 else -1.0

            if rho > 0.75:
                damping = max(damping / 3.0, damping_min)
                step_accepted = True
            elif rho > 0.25:
                step_accepted = True
            else:
                damping = min(damping * damping_factor, damping_max)

            if step_accepted:
                x = x_new
                if new_loss < best_loss:
                    best_loss = new_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if verbose:
                    print(f"    [LM] it={it} ||F||={float(torch.norm(F_new)):.3e} mu={damping:.1e} rho={rho:.3f}")
                break

        if not step_accepted:
            if patience_counter > max_patience:
                break

    rn = float(torch.norm(residual_fn(x).detach()))
    return {
        "x": x.detach(),
        "residual_norm": rn,
        "n_iter": it + 1,
        "loss_history": history,
    }

def gn_optimize(residual_fn, x0, max_iter=200, variant="original", verbose=False):
    if str(variant).lower() in ("q2", "aligned", "beam"):
        return _gn_optimize_q2(residual_fn, x0, max_iter=max_iter, verbose=verbose)
    return _gn_optimize_original(residual_fn, x0, max_iter=max_iter, verbose=verbose)


def _gn_optimize_original(residual_fn, x0, max_iter=200, tol=1.0e-9, rel_tol=1.0e-12,
                          reg=1.0e-10, max_halving=25, verbose=False):
    import torch
    torch.set_default_dtype(torch.float64)
    x = x0.clone().detach()
    device = x.device
    n = x.numel()
    eye = torch.eye(n, device=device)
    F = residual_fn(x).detach()
    cost = float(torch.dot(F, F))
    ref = max(cost ** 0.5, 1.0)
    rel = 1.0
    history = [cost]
    it = 0
    while it < max_iter:
        it += 1
        if cost ** 0.5 <= tol * ref:
            break
        J = torch.autograd.functional.jacobian(residual_fn, x, vectorize=True).detach()
        JTJ = J.T @ J
        JTr = J.T @ F
        A = JTJ + reg * torch.diag(torch.diagonal(JTJ)) + 1.0e-14 * eye
        try:
            delta = torch.linalg.solve(A, -JTr)
        except Exception:
            break
        alpha = 1.0
        improved = False
        for _ in range(max_halving):
            F_new = residual_fn(x + alpha * delta).detach()
            cost_new = float(torch.dot(F_new, F_new))
            if cost_new < cost:
                rel = (cost - cost_new) / max(cost, 1.0e-300)
                x = x + alpha * delta
                F = F_new
                cost = cost_new
                history.append(cost)
                improved = True
                break
            alpha *= 0.5
        if verbose:
            print(f"    [GN-orig] it={it} ||F||={cost ** 0.5:.3e} alpha={alpha:.2e}")
        if not improved or rel < rel_tol:
            break
    return {"x": x.detach(), "residual_norm": cost ** 0.5, "n_iter": it, "loss_history": history}


def _gn_optimize_q2(residual_fn, x0, max_iter=200, tol=1.0e-10, damping=0.1, verbose=False):
    import torch
    torch.set_default_dtype(torch.float64)
    x = x0.clone().detach()
    device = x.device
    n = x.numel()
    eye = torch.eye(n, device=device)
    history = []
    it = 0
    for it in range(max_iter):
        F = residual_fn(x).detach()
        rnorm = float(torch.norm(F))
        history.append(rnorm ** 2)

        if rnorm < tol:
            break

        J = torch.autograd.functional.jacobian(residual_fn, x, vectorize=True).detach()
        JTJ = J.T @ J
        JTr = J.T @ F
        A = JTJ + damping * torch.diag(torch.diagonal(JTJ)) + 1.0e-10 * eye
        try:
            delta = torch.linalg.solve(A, -JTr)
        except Exception:
            damping *= 10.0
            continue

        alpha = 1.0
        beta = 0.5
        c = 0.1
        rnsq = rnorm ** 2
        for _ in range(20):
            F_try = residual_fn(x + alpha * delta).detach()
            if float(torch.dot(F_try, F_try)) < rnsq * (1.0 - c * alpha):
                break
            alpha *= beta

        x = x + alpha * delta
        new_norm = float(torch.norm(residual_fn(x).detach()))
        if new_norm < rnorm:
            damping *= 0.7
        else:
            damping *= 2.0
        if verbose:
            print(f"    [GN-q2] it={it} ||F||={new_norm:.3e} alpha={alpha:.2e} damping={damping:.1e}")

    rn = float(torch.norm(residual_fn(x).detach()))
    return {"x": x.detach(), "residual_norm": rn, "n_iter": it + 1, "loss_history": history}

def lbfgs_residual(residual_fn, x0, max_iter=50000, lr=0.8, history_size=200,
                   tolerance_grad=1.0e-16, tolerance_change=1.0e-16,
                   perturbation_enabled=True, perturbation_threshold=5.0e-4,
                   perturbation_patience=600, perturbation_scale=5.0e-3,
                   perturbation_scale_increment=0.25, precondition=True, verbose=False):
    import torch
    torch.set_default_dtype(torch.float64)
    x0d = x0.clone().detach()
    with torch.no_grad():
        F0 = residual_fn(x0d)
        denom = max(float(torch.dot(F0, F0)), 1.0e-300)

    if precondition:
        J0 = torch.autograd.functional.jacobian(residual_fn, x0d, vectorize=True).detach()
        cs = torch.sqrt((J0 * J0).sum(dim=0))
        cs[(cs == 0) | ~torch.isfinite(cs)] = 1.0
    else:
        cs = torch.ones_like(x0d)

    p = torch.zeros_like(x0d).requires_grad_(True)
    history = []
    total_iters = 0
    restart_count = 0

    while total_iters < max_iter:
        opt = torch.optim.LBFGS(
            [p], lr=lr, max_iter=max_iter - total_iters, history_size=history_size,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
            line_search_fn="strong_wolfe")
        flat_count = 0
        last_loss = float('inf')
        need_restart = False

        def closure():
            nonlocal flat_count, last_loss, need_restart, total_iters
            opt.zero_grad()
            F = residual_fn(x0d + p / cs)
            loss = torch.dot(F, F) / denom
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p], max_norm=100.0)
            loss_val = float(loss.item())
            history.append(loss_val)
            total_iters += 1
            if perturbation_enabled and len(history) > 1:
                rel = abs(loss_val - last_loss) / max(abs(last_loss), 1.0e-10)
                if rel < perturbation_threshold:
                    flat_count += 1
                else:
                    flat_count = 0
                if flat_count >= perturbation_patience:
                    need_restart = True
                    raise StopIteration("perturbation restart")
            last_loss = loss_val
            if total_iters >= max_iter:
                raise StopIteration("max iterations")
            return loss

        try:
            opt.step(closure)
            break
        except StopIteration:
            if need_restart and perturbation_enabled and total_iters < max_iter:
                restart_count += 1
                current_scale = perturbation_scale * (1.0 + perturbation_scale_increment * (restart_count - 1))
                with torch.no_grad():
                    x_cur = x0d + p / cs
                    std = torch.std(x_cur).item()
                    noise_scale = current_scale * std if std > 0 else current_scale
                    p.add_(cs * torch.randn_like(p) * noise_scale)
                continue
            break
        except Exception:
            opt_adam = torch.optim.Adam([p], lr=0.01)
            for _ in range(min(max_iter - total_iters, 1000)):
                opt_adam.zero_grad()
                F = residual_fn(x0d + p / cs)
                loss = torch.dot(F, F) / denom
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p], max_norm=1.0e3)
                opt_adam.step()
                history.append(float(loss.item()))
                total_iters += 1
            break

    with torch.no_grad():
        x_final = x0d + p / cs
        rn = float(torch.norm(residual_fn(x_final)))
    if verbose:
        print(f"    [LBFGS] iters={total_iters} restarts={restart_count} precond={bool(precondition)} ||F||={rn:.3e}")
    return {"x": x_final.detach(), "residual_norm": rn, "n_iter": total_iters, "loss_history": history}

class _MGConverged(Exception):
    pass


def build_mg_prolongations(n_fine, coarse_levels=(2, 3, 5, 8, 13), n_fields=5):
    from .dq_core import chebyshev_lobatto_nodes, interpolation_row

    n_fine = int(n_fine)
    coarse = sorted({int(l) for l in coarse_levels if 2 <= int(l) < n_fine})
    stages_levels = [coarse[:3], coarse, coarse + [n_fine]]
    stages = []
    for st in stages_levels:
        if st and (not stages or st != stages[-1]):
            stages.append(st)

    xf = chebyshev_lobatto_nodes(n_fine, 1.0)

    def _level_block(l):
        if l == n_fine:
            P2 = np.eye(n_fine * n_fine)
        else:
            xc = chebyshev_lobatto_nodes(l, 1.0)
            P1 = np.vstack([interpolation_row(xc, t) for t in xf])
            P2 = np.kron(P1, P1)
        nf, nc = P2.shape
        B = np.zeros((n_fields * nf, n_fields * nc))
        for f in range(n_fields):
            B[f * nf:(f + 1) * nf, f * nc:(f + 1) * nc] = P2
        return B

    blocks = {l: _level_block(l) for l in stages[-1]}
    return [np.hstack([blocks[l] for l in st]) for st in stages]


def lbfgs_residual_mg(residual_fn, x0, prolongations, stage_budgets,
                      lr=1.0, history_size=200, tol_residual=1.0e-5,
                      restart_scale=0.0, restart_gate=None, restart_patience=600,
                      restart_threshold=5.0e-4, restart_seed=None, restart_chunk=25,
                      verbose=False):
    import torch
    torch.set_default_dtype(torch.float64)
    x = x0.clone().detach()
    with torch.no_grad():
        F0 = residual_fn(x)
        denom0 = max(float(torch.dot(F0, F0)), 1.0e-300)
    history = []
    total_evals = 0
    restart_on = restart_scale is not None and float(restart_scale) > 0.0
    kicks = 0
    suppressed = 0
    gen = None
    if restart_on:
        if restart_seed is None:
            raise ValueError("lbfgs_residual_mg: restart_scale>0 需要显式 restart_seed (复现性)")
        gen = torch.Generator()
        gen.manual_seed(int(restart_seed))

    for P_np, budget in zip(prolongations, stage_budgets):
        if budget <= 0:
            continue
        with torch.no_grad():
            rn_in = float(torch.norm(residual_fn(x)))
        if rn_in < tol_residual:
            break
        denom_stage = max(rn_in ** 2, 1.0e-300)
        Pt = torch.as_tensor(np.asarray(P_np, dtype=float), device=x.device)
        J0 = torch.autograd.functional.jacobian(residual_fn, x, vectorize=True).detach()
        cs = torch.sqrt(((J0 @ Pt) ** 2).sum(dim=0))
        cs[(cs == 0) | ~torch.isfinite(cs)] = 1.0
        del J0
        x_base = x
        c = torch.zeros(Pt.shape[1], dtype=torch.float64, device=x.device, requires_grad=True)
        opt = torch.optim.LBFGS([c], lr=lr,
                                max_iter=(int(restart_chunk) if restart_on else budget),
                                history_size=history_size,
                                tolerance_grad=1.0e-16, tolerance_change=1.0e-16,
                                line_search_fn="strong_wolfe")
        state = {"evals": 0, "best": float("inf"), "best_c": None}
        flat = {"count": 0, "last": float("inf")}

        def closure():
            opt.zero_grad()
            F = residual_fn(x_base + Pt @ (c / cs))
            loss = torch.dot(F, F) / denom_stage
            loss.backward()
            state["evals"] += 1
            lv = float(loss.item())
            history.append(lv * (denom_stage / denom0))
            if lv < state["best"]:
                state["best"] = lv
                state["best_c"] = c.detach().clone()
            if restart_on:
                rel = abs(lv - flat["last"]) / max(abs(flat["last"]), 1.0e-10)
                flat["count"] = flat["count"] + 1 if rel < restart_threshold else 0
                flat["last"] = lv
            if (lv * denom_stage) ** 0.5 < tol_residual or state["evals"] >= budget:
                raise _MGConverged
            return loss

        if not restart_on:
            try:
                opt.step(closure)
            except _MGConverged:
                pass
        else:
            stage_done = False
            while not stage_done:
                try:
                    opt.step(closure)
                except _MGConverged:
                    stage_done = True
                    break
                if flat["count"] >= restart_patience:
                    rn_cur = (flat["last"] * denom_stage) ** 0.5
                    if restart_gate is None or rn_cur > restart_gate:
                        kicks += 1
                        with torch.no_grad():
                            sd = float(torch.std(c))
                            noise = torch.randn(c.shape, generator=gen,
                                                dtype=torch.float64).to(c.device)
                            c.add_(noise * float(restart_scale) * (sd if sd > 0.0 else 1.0))
                    else:
                        suppressed += 1
                    flat["count"] = 0
                    flat["last"] = float("inf")
        total_evals += state["evals"]
        with torch.no_grad():
            if state["best_c"] is not None:
                x = (x_base + Pt @ (state["best_c"] / cs)).detach()
            rn = float(torch.norm(residual_fn(x)))
        del Pt
        if verbose:
            print(f"    [LBFGS-MG] stage_cols={int(P_np.shape[1])} evals={state['evals']} ||F||={rn:.3e}")
        if rn < tol_residual:
            break

    with torch.no_grad():
        rn = float(torch.norm(residual_fn(x)))
    out = {"x": x, "residual_norm": rn, "n_iter": total_evals, "loss_history": history}
    if restart_on:
        out.update({"mg_restart_kicks": kicks, "mg_restart_suppressed": suppressed,
                    "mg_restart_scale": float(restart_scale),
                    "mg_restart_gate": (None if restart_gate is None else float(restart_gate)),
                    "mg_restart_seed": int(restart_seed)})
        if verbose:
            print("    [LBFGS-MG] restart: kicks=%d suppressed=%d (scale=%g, gate=%s, seed=%d)"
                  % (kicks, suppressed, restart_scale, restart_gate, restart_seed))
    return out


def lbfgs_energy(energy_fn, x0, max_iter=3000, lr=1.0, history_size=100,
                 tolerance_grad=1.0e-14, tolerance_change=1.0e-20, precondition=True,
                 verbose=False):
    import torch
    torch.set_default_dtype(torch.float64)
    x0d = x0.clone().detach()
    if precondition:
        H = torch.autograd.functional.hessian(energy_fn, x0d)
        cs = torch.sqrt(torch.abs(torch.diagonal(H)))
        cs[(cs == 0) | ~torch.isfinite(cs)] = 1.0
    else:
        cs = torch.ones_like(x0d)
    p = torch.zeros_like(x0d).requires_grad_(True)
    opt = torch.optim.LBFGS(
        [p], lr=lr, max_iter=max_iter, history_size=history_size,
        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        e = energy_fn(x0d + p / cs)
        e.backward()
        return e

    opt.step(closure)
    x = (x0d + p / cs).detach()
    xr = x.clone().requires_grad_(True)
    e = energy_fn(xr)
    rn = float(torch.norm(torch.autograd.grad(e, xr)[0]))
    n_iter = int(opt.state[opt._params[0]].get("n_iter", -1))
    if verbose:
        print(f"    [LBFGS-energy] iters={n_iter} ||grad||={rn:.3e}")
    return {"x": x, "residual_norm": rn, "n_iter": n_iter}

def residual_solve(residual_fn, x0, optimizer="lm", max_iter=None,
                   gn_variant="original", lbfgs_precondition=True,
                   lbfgs_mg_prolongations=None,
                   lbfgs_mg_stage_fractions=(0.15, 0.25, 0.60),
                   lbfgs_mg_tol=1.0e-5,
                   lbfgs_mg_restart_scale=0.0, lbfgs_mg_restart_gate=None,
                   lbfgs_mg_restart_patience=600, lbfgs_mg_restart_threshold=5.0e-4,
                   lbfgs_mg_restart_seed=None, verbose=False):
    o = str(optimizer).lower()
    if o == "lm":
        return lm_optimize(residual_fn, x0, verbose=verbose,
                           **({"max_iter": max_iter} if max_iter is not None else {}))
    if o == "gn":
        return gn_optimize(residual_fn, x0, variant=gn_variant, verbose=verbose,
                           **({"max_iter": max_iter} if max_iter is not None else {}))
    if o in ("lbfgs", "l-bfgs"):
        lb_iter = max_iter if max_iter is not None else 50000
        if lbfgs_mg_prolongations is not None:
            n_st = len(lbfgs_mg_prolongations)
            fr = list(lbfgs_mg_stage_fractions) or [1.0]
            if len(fr) < n_st:
                fr += [fr[-1]] * (n_st - len(fr))
            fr = fr[:n_st]
            s = float(sum(fr)) or 1.0
            budgets = [max(1, int(round(lb_iter * f / s))) for f in fr]
            return lbfgs_residual_mg(residual_fn, x0, lbfgs_mg_prolongations, budgets,
                                     tol_residual=lbfgs_mg_tol,
                                     restart_scale=lbfgs_mg_restart_scale,
                                     restart_gate=lbfgs_mg_restart_gate,
                                     restart_patience=lbfgs_mg_restart_patience,
                                     restart_threshold=lbfgs_mg_restart_threshold,
                                     restart_seed=lbfgs_mg_restart_seed, verbose=verbose)
        return lbfgs_residual(residual_fn, x0, max_iter=lb_iter,
                              precondition=lbfgs_precondition, verbose=verbose)
    raise ValueError(f"optimizer must be 'lbfgs'/'gn'/'lm', got {optimizer!r}")

def _row_col_scaling(A):

    rs = np.sqrt((A ** 2).sum(axis=1))
    rs[(rs == 0) | ~np.isfinite(rs)] = 1.0
    cs = np.sqrt((A ** 2).sum(axis=0))
    cs[(cs == 0) | ~np.isfinite(cs)] = 1.0
    return rs, cs

def odl_linear_solve(system, max_iter=4000, lr=1.0, history_size=100,
                      tolerance_grad=1.0e-12, tolerance_change=1.0e-18,
                      device=None, verbose=True, record_history=True):

    import torch
    torch.set_default_dtype(torch.float64)
    if device is None:
        device = DEFAULT_DEVICE

    K = np.array(system["KL_red"], dtype=float)
    f = np.asarray(system["f_red"], dtype=float).ravel()
    rs, cs = _row_col_scaling(K)
    As = (K / rs[:, None]) / cs[None, :]
    bs = f / rs
    bs_norm_sq = float(bs @ bs) or 1.0

    As_t = torch.as_tensor(As, device=device)
    bs_t = torch.as_tensor(bs, device=device)
    denom = torch.as_tensor(bs_norm_sq, device=device)
    y = torch.zeros(As.shape[1], device=device, requires_grad=True)

    history = []

    opt = torch.optim.LBFGS(
        [y], lr=lr, max_iter=max_iter, history_size=history_size,
        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = (As_t @ y - bs_t).pow(2).sum() / denom
        loss.backward()
        if record_history:
            history.append(float(loss.item()))
        return loss

    opt.step(closure)

    final_loss = float(((As_t @ y - bs_t).pow(2).sum() / denom).item())
    y_np = y.detach().cpu().numpy()
    displacement_red = y_np / cs
    displacement_full = system["transform"] @ displacement_red
    residual_norm = float(np.linalg.norm(K @ displacement_red - f))
    n_iter = int(opt.state[opt._params[0]].get("n_iter", -1))

    if verbose:
        print(f"  [ODL] L-BFGS iters={n_iter}, relative loss={final_loss:.3e}, "
              f"residual norm={residual_norm:.3e}")

    return {
        "displacement_red": displacement_red,
        "displacement_full": displacement_full,
        "residual_norm": residual_norm,
        "final_loss": final_loss,
        "n_iter": n_iter,
        "loss_history": history,
    }

def odl_energy_solve(system, max_iter=3000, lr=1.0, history_size=100,
                      tolerance_grad=1.0e-12, tolerance_change=1.0e-20,
                      device=None, verbose=True, record_history=True):

    import torch
    torch.set_default_dtype(torch.float64)
    if device is None:
        device = DEFAULT_DEVICE

    K = np.array(system["KL_red"], dtype=float)
    f = np.asarray(system["f_red"], dtype=float).ravel()
    s = np.sqrt(np.abs(np.diag(K)))
    s[(s == 0) | ~np.isfinite(s)] = 1.0
    Ktil = (K / s[:, None]) / s[None, :]
    ftil = f / s
    f_norm = float(np.linalg.norm(f)) or 1.0

    Ktil_t = torch.as_tensor(Ktil, device=device)
    ftil_t = torch.as_tensor(ftil, device=device)
    z = torch.zeros(K.shape[1], device=device, requires_grad=True)

    history = []

    opt = torch.optim.LBFGS(
        [z], lr=lr, max_iter=max_iter, history_size=history_size,
        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        Kz = Ktil_t @ z
        loss = 0.5 * torch.dot(z, Kz) - torch.dot(z, ftil_t)
        loss.backward()
        if record_history:
            history.append(float(loss.item()))
        return loss

    opt.step(closure)

    z_np = z.detach().cpu().numpy()
    displacement_red = z_np / s
    displacement_full = system["transform"] @ displacement_red
    residual_norm = float(np.linalg.norm(K @ displacement_red - f))
    rel_residual = residual_norm / f_norm
    energy = float((0.5 * z_np @ (Ktil @ z_np) - z_np @ ftil))
    n_iter = int(opt.state[opt._params[0]].get("n_iter", -1))

    if verbose:
        print(f"  [ODL-energy] L-BFGS iters={n_iter}, energy={energy:.6e}, "
              f"rel residual={rel_residual:.3e}")

    return {
        "displacement_red": displacement_red,
        "displacement_full": displacement_full,
        "residual_norm": residual_norm,
        "rel_residual": rel_residual,
        "energy": energy,
        "n_iter": n_iter,
        "loss_history": history,
    }
