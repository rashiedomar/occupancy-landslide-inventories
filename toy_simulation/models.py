"""
Models compared in the toy experiment.

  NaiveAll      : standard practice. One classifier, label = "recorded in any inventory",
                  inputs = terrain + accessibility features. Map made with real features.
  NaiveTerrain  : same, but accessibility features dropped (the "just remove road
                  distance" fix that Steger et al. showed is confounded).
  Occupancy     : OURS. Two heads.
                    psi-head  f(terrain)       -> P(landslide really happened)
                    p-head    g(access)_k      -> P(inventory k recorded it | it happened)
                  trained with the occupancy marginal likelihood over K inventories.
                  At test time only psi is used.
  Oracle        : classifier trained on the TRUE landslides z (upper bound; impossible
                  in real life).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(d_in, d_out, hidden=64, depth=2, dropout=0.0):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------- baselines
class BinaryClassifier(nn.Module):
    def __init__(self, d_in, **kw):
        super().__init__()
        self.net = mlp(d_in, 1, **kw)

    def forward(self, x):
        return self.net(x).squeeze(-1)          # logit

    def loss(self, x, y):
        return F.binary_cross_entropy_with_logits(self(x), y.float())

    @torch.no_grad()
    def predict(self, x):
        return torch.sigmoid(self(x))


# ----------------------------------------------------------------------------- monotone detection
class MonotoneDetection(nn.Module):
    """Additive, sign-constrained detection model (a monotone GAM per inventory).

        logit p_k(x) = b_k + sum_j  s_j * sum_m softplus(a_kjm) * sigmoid(softplus(c_kjm) * (x_j - t_m))

    s_j in {+1, -1, 0} is the KNOWN direction of feature j's effect on detection
    (e.g. -1 for distance to road: being farther from a road never makes a landslide MORE
    likely to be reported). Each inventory can still learn a zero effect. Additive -> the
    learned curves can be plotted and checked against what we know about each observer.
    """

    def __init__(self, signs, K, n_basis=8, lo=-2.5, hi=2.5):
        super().__init__()
        self.register_buffer("signs", torch.tensor(signs, dtype=torch.float32))
        self.register_buffer("knots", torch.linspace(lo, hi, n_basis))
        d = len(signs)
        self.a = nn.Parameter(torch.randn(K, d, n_basis) * 0.1 - 2.0)   # amplitudes (softplus)
        self.c = nn.Parameter(torch.zeros(K, d, n_basis))                # slopes (softplus)
        self.b = nn.Parameter(torch.zeros(K))

    def component(self, x):
        """(N, K, d) per-feature contributions to the logit."""
        u = torch.sigmoid(F.softplus(self.c)[None] * 2.0 * (x[:, None, :, None] - self.knots))
        return (F.softplus(self.a)[None] * u).sum(-1) * self.signs

    def forward(self, x):
        return self.b + self.component(x).sum(-1)


class SpatialMonotoneDetection(nn.Module):
    """MonotoneDetection on the access features + a smooth spatial field per inventory that can
    absorb unobserved, spatially structured reporting effort (local media, NGO presence...).

    Input columns: [access features (len(signs)) ..., spatial basis (n_knots)], where the basis is
    precomputed once with rbf_basis(): phi_r(loc) = exp(-|loc - knot_r|^2 / (2 h^2)).
    Field_k(loc) = phi(loc) @ w_k, Gaussian prior w ~ N(0, sigma^2) added to the loss as a penalty
    (scaled per training cell so it acts like a proper prior).
    """

    def __init__(self, signs, K, n_knots, sigma=0.5, n_train=1):
        super().__init__()
        self.mono = MonotoneDetection(signs, K)
        self.d = len(signs)
        self.w = nn.Parameter(torch.zeros(n_knots, K))
        self.prior_scale = 1.0 / (2 * sigma ** 2 * n_train)

    def field(self, basis):
        return basis @ self.w

    def forward(self, x):
        return self.mono(x[:, :self.d]) + self.field(x[:, self.d:])

    def penalty(self):
        return self.prior_scale * (self.w ** 2).sum()


def rbf_basis(loc, knots, bandwidth):
    """Gaussian radial basis, (N, n_knots). loc, knots in the same units (e.g. km)."""
    d2 = ((loc[:, None, :] - knots[None]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * bandwidth ** 2)).astype(np.float32)


class PsiWithRoadCut(nn.Module):
    """Susceptibility = MLP(terrain) + a short-range, non-negative road-cut term.

    logit psi = f(terrain) + softplus(a) * exp(-d_road_km / lam),   lam in [0.05, 0.30] km.
    Encodes 'scale separation': real road-cut effects act within a few hundred metres, whereas
    reporting bias acts over kilometres. Last input column must be RAW distance to road in km.
    """

    def __init__(self, d_terrain, lam_range=(0.05, 0.30), **kw):
        super().__init__()
        self.f = mlp(d_terrain, 1, **kw)
        self.a = nn.Parameter(torch.tensor(-1.0))
        self.lam_raw = nn.Parameter(torch.tensor(0.0))
        self.lo, self.hi = lam_range

    def lam(self):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.lam_raw)

    def forward(self, x):
        return self.f(x[:, :-1]) + F.softplus(self.a) * torch.exp(-x[:, -1:] / self.lam())


# ----------------------------------------------------------------------------- ours
class OccupancyNet(nn.Module):
    """Deep occupancy-detection model for landslide inventories.

    log L_i = logaddexp(
        log psi_i + sum_k m_ik [ y_ik log p_ik + (1 - y_ik) log(1 - p_ik) ],   # landslide present
        log(1 - psi_i) + log 1[ all observed y_ik == 0 ]                       # absent
    )
    m_ik = 1 if inventory k surveyed cell i (coverage mask), else 0.
    """

    def __init__(self, d_site, d_obs, K, p_head=None, fixed_p=None, psi_head=None, false_pos=None, **kw):
        """p_head : optional custom detection module (e.g. MonotoneDetection); default = MLP.
        fixed_p: optional list of length K; a number fixes that inventory's detection prob
                 (e.g. 0.99 for an exhaustively mapped anchor area), None = learned."""
        super().__init__()
        self.psi_head = psi_head if psi_head is not None else mlp(d_site, 1, **kw)
        self.p_head = p_head if p_head is not None else mlp(d_obs, K, **kw)
        self.K = K
        fp = [float("nan") if v is None else float(np.log(v / (1 - v))) for v in (fixed_p or [None] * K)]
        self.register_buffer("fixed_logit", torch.tensor(fp, dtype=torch.float32))
        # optional false positives: P(record | no landslide) = f_k, learned per inventory.
        # false_pos: list of length K with True (learn f_k) / False (f_k = 0, e.g. exhaustive anchor survey)
        self.false_pos = false_pos
        if false_pos is not None:
            self.fp_logit = nn.Parameter(torch.full((K,), -4.0))
            self.register_buffer("fp_mask", torch.tensor([float(v) for v in false_pos]))

    def logits(self, x_site, x_obs):
        lp = self.p_head(x_obs)
        if lp.shape[-1] < self.K:      # custom head models only the learned inventories
            full = self.fixed_logit.expand(lp.shape[0], -1).clone()
            full[:, torch.isnan(self.fixed_logit)] = lp
            lp = full
        else:
            lp = torch.where(torch.isnan(self.fixed_logit), lp, self.fixed_logit)
        return self.psi_head(x_site).squeeze(-1), lp

    def loss(self, x_site, x_obs, Y, M):
        """Y, M: (N, K) observations and coverage masks."""
        lpsi_logit, lp_logit = self.logits(x_site, x_obs)
        log_psi = F.logsigmoid(lpsi_logit)
        log_1m_psi = F.logsigmoid(-lpsi_logit)
        log_p = F.logsigmoid(lp_logit)
        log_1m_p = F.logsigmoid(-lp_logit)
        Yf, Mf = Y.float(), M.float()
        obs_ll = (Mf * (Yf * log_p + (1 - Yf) * log_1m_p)).sum(-1)
        present = log_psi + obs_ll
        never_seen = ((Yf * Mf).sum(-1) == 0)
        if self.false_pos is None:
            absent = torch.where(never_seen, log_1m_psi, torch.full_like(log_1m_psi, -1e4))
        else:
            lf, l1f = F.logsigmoid(self.fp_logit), F.logsigmoid(-self.fp_logit)
            fp_ll = (Mf * self.fp_mask * (Yf * lf + (1 - Yf) * l1f)).sum(-1)
            # inventories without false positives (mask 0) still forbid records where there is no landslide
            hard = (Yf * Mf * (1 - self.fp_mask)).sum(-1) > 0
            absent = torch.where(hard, torch.full_like(log_1m_psi, -1e4), log_1m_psi + fp_ll)
        nll = -(torch.logaddexp(present, absent)).mean()
        if hasattr(self.p_head, "penalty"):
            nll = nll + self.p_head.penalty()
        return nll

    @torch.no_grad()
    def predict(self, x_site):
        return torch.sigmoid(self.psi_head(x_site).squeeze(-1))

    @torch.no_grad()
    def predict_detection(self, x_obs):
        return torch.sigmoid(self.p_head(x_obs))


# ----------------------------------------------------------------------------- spillover false positives
class SpilloverOccupancy(nn.Module):
    """Occupancy model whose false positives come mostly from NEIGHBOURING landslides
    (positional mismatch between inventories), plus a small background rate:

        P(record_ik | z_i = 0) = sigmoid(a_k) * q_i + sigmoid(b_k),
        q_i = 1 - prod_{j in 8-neighbours(i)} (1 - psi_j)   (prob. that a neighbour has a landslide)

    psi is evaluated on ALL cells each step (features only, no labels), so neighbours of training cells
    may lie in validation/test blocks without leaking labels. Loss is computed on row subsets `idx`.
    fp_mask[k] = 0 disables false positives for inventory k (e.g. the exhaustive anchor survey).
    """

    def __init__(self, psi_head, p_head, K, X_site, X_obs, Y, M, nbr, fixed_p=None, fp_mask=None):
        super().__init__()
        self.psi_head, self.p_head, self.K = psi_head, p_head, K
        for n, v in [("X_site", X_site), ("X_obs", X_obs), ("Y", Y.float()), ("M", M.float()), ("nbr", nbr)]:
            self.register_buffer(n, v)
        fp = [float("nan") if v is None else float(np.log(v / (1 - v))) for v in (fixed_p or [None] * K)]
        self.register_buffer("fixed_logit", torch.tensor(fp, dtype=torch.float32))
        self.register_buffer("fp_mask", torch.tensor([1.0] * K if fp_mask is None else [float(v) for v in fp_mask]))
        self.a = nn.Parameter(torch.full((K,), -1.0))     # spillover strength
        self.b = nn.Parameter(torch.full((K,), -5.0))     # background false-positive rate

    def det_logits(self, xo):
        lp = self.p_head(xo)
        full = self.fixed_logit.expand(lp.shape[0], -1).clone()
        full[:, torch.isnan(self.fixed_logit)] = lp
        return full

    def loss(self, idx):
        lpsi_all = self.psi_head(self.X_site).squeeze(-1)
        psi_all = torch.sigmoid(lpsi_all)
        nb = self.nbr[idx]
        ok = nb >= 0
        one_m = torch.where(ok, 1 - psi_all[nb.clamp(min=0)], torch.ones_like(nb, dtype=psi_all.dtype))
        q = 1 - torch.exp(torch.log(one_m.clamp(min=1e-6)).sum(-1))
        f = (torch.sigmoid(self.a)[None] * q[:, None] + torch.sigmoid(self.b)[None]).clamp(1e-6, 1 - 1e-4)
        lpsi = lpsi_all[idx]
        lp = self.det_logits(self.X_obs[idx])
        Y, M = self.Y[idx], self.M[idx]
        present = F.logsigmoid(lpsi) + (M * (Y * F.logsigmoid(lp) + (1 - Y) * F.logsigmoid(-lp))).sum(-1)
        fp_ll = (M * self.fp_mask * (Y * torch.log(f) + (1 - Y) * torch.log1p(-f))).sum(-1)
        hard = (Y * M * (1 - self.fp_mask)).sum(-1) > 0
        absent = torch.where(hard, torch.full_like(lpsi, -1e4), F.logsigmoid(-lpsi) + fp_ll)
        return -(torch.logaddexp(present, absent)).mean()

    @torch.no_grad()
    def predict(self, x_site):
        return torch.sigmoid(self.psi_head(x_site).squeeze(-1))

    @torch.no_grad()
    def predict_detection(self, x_obs):
        return torch.sigmoid(self.p_head(x_obs))


def neighbour_index(mask):
    """(N, 8) indices of the 8 neighbours of each masked cell, in mask (row-major) order; -1 if outside."""
    H, W = mask.shape
    pos = -np.ones((H, W), np.int64)
    pos[mask] = np.arange(mask.sum())
    rr, cc = np.nonzero(mask)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r2, c2 = rr + dr, cc + dc
            inside = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
            v = np.full(len(rr), -1, np.int64)
            v[inside] = pos[r2[inside], c2[inside]]
            out.append(v)
    return np.stack(out, 1)


# ----------------------------------------------------------------------------- training
def train(model, loss_fn, steps=1500, lr=5e-3, wd=1e-4, seed=0, verbose=False,
          val_loss_fn=None, eval_every=25, patience=300):
    """Full-batch AdamW. If val_loss_fn is given, keep the weights with the best
    validation loss (early stopping). The validation loss must only use data that
    would be available in real life (observed inventories), never the hidden truth."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, since = float("inf"), None, 0
    for t in range(steps):
        model.train()
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        if val_loss_fn is not None and t % eval_every == 0:
            model.eval()
            with torch.no_grad():
                v = val_loss_fn().item()
            if v < best - 1e-6:
                best, since = v, 0
                best_state = {k: x.clone() for k, x in model.state_dict().items()}
            else:
                since += eval_every
                if since >= patience:
                    break
        if verbose and t % 300 == 0:
            print(f"  step {t:5d}  loss {loss.item():.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.stopped_at = t
    return model
