# Numerical Methods for the 1D Fokker-Planck Equation

*Detailed notes on discretizing an aFokker-Planck equation in two
different ways, and why both ways agree.*

These notes assume you've seen single-variable calculus (Taylor series,
integration by parts, the fundamental theorem of calculus) and have some
exposure to PDEs and basic probability (expectation, variance, the normal
distribution). Nothing about finite differences, finite volumes, or
stochastic calculus is assumed.

This file is the detailed companion to the project's top-level `README.md`.
The README gives the quick-reference version of everything derived here -
the equations you need to *use* the solvers; this file derives *why* each
piece is the way it is.

---

## 1. Setup and notation

We're solving

$$
   \partial_t p = -\partial_x( b(x,t) p ) + \partial_{xx}^2( D(x,t) p ) = -\partial_x J(x,t) \qquad (1), 
$$

for a density $p(x, t) \geq 0$ with $\int p(x,t) dx = 1$ (or less, if probability can
leave through the boundary, as we will see). The function $J$ is the **probability current**: the equation is a continuity equation,
identical in form to conservation of mass. This is not
a cosmetic rewriting — it is the entire reason the numerical scheme conserves mass/probability *exactly* 
(at least when we do not use Dirichlet boundary conditions).

You may be more used to seeing the diffusion term written $\partial_x (D \partial_x p)$
instead. The two are **the same equation only when $D$ is constant** — they
differ by a term proportional to $\partial_x D$ whenever $D$ varies with $x$. §2.4
explains why this project deliberately uses the $\partial_x ^2 (Dp)$ form: it is
the one that makes Parts I and II describe *exactly* the same physics with
no hidden correction terms, for any $D(x,t)$.

There are two equally valid ways to think about what this equation
describes:

- **Eulerian picture**: Fix locations in space and watch the density
  $p(x, t)$ flow past them. This is the picture
  the PDE itself is written in.
- **Lagrangian picture**: Instead of tracking the whole probability $p(x,t)$, track individual
  *particles* moving randomly, and let the density be whatever
  distribution those particles happen to have at time $t$. A particle's
  position $X_t$ obeys the stochastic differential equation (SDE)
  
  $$dX_t = b(X_t, t) dt + \sqrt{2 D(X_t, t)} dW_t \qquad (2)$$
  
  where $W_t$ is a Brownian motion. The resulting density, in the limit of an infinite
  number of tracked particles, is precisely $p(x,t)$.

Part I discretizes the Eulerian PDE directly (finite volumes + Euler time
stepping). Part II discretizes the Lagrangian SDE instead (Euler-Maruyama)
and derives, from scratch, why the resulting particle cloud reproduces the
*same* $p(x, t)$. That derivation is the mathematical justification for
"just simulate a lot of particles and histogram them".

---

## Part I — The Eulerian route: finite volumes

*(We use "cells" and "faces" throughout, as is standard for finite-volume
schemes, even though in 1D these are simply intervals and points: a
"cell" is an interval, a "face" is one of its two endpoints.)*

### 1.1 Cell averages give an *exact* identity 

We solve (1) on a domain $I$ (an interval) :

```

         dx
I=    |-----|-----|-----| .... |-----|
      x₁    x₂    x₃    x₄     xₙ    xₙ₊₁
         I₁    I₂    I₃           Iₙ
``` 

Averaging $p(x,t)$ over the $i$-th cell (denoted as $I_i$, since in th 1D case it is an interval),
we obtain the averaged values  

$$
p_t [i] = \frac{1}{dx}\int_{I_i} p(x,t) dx,
$$

which satisfy

$$
\partial_t p_t [i] =dx^{-1} \int_{I_i} \partial_t p(x,t) dx = -dx^{-1}\int_{I_i} \partial_x J(x,t) dx =- \frac{J_t[x_{i+1}]-J_t[x_i]}{dx} \qquad (3),
$$

where 

$$J_t [x_i] = J_t \text{ evaluated at } x_i.$$


Summing over every cell has a telescopic effect: every interior face flux appears
once with a `+` sign (leaving its left cell) and once with a `-` sign
(entering its right cell), so all interior terms cancel and only the two
domain-boundary fluxes survive:

$$
\frac{d}{dt} \text{Total mass at time t} = \frac{d}{dt} \int_I p(x,t) dx=J_t[x_1] - J_t[x_{n+1}] \qquad (4).
$$

That is, positive fluxes at the left extreme of the domain makes mass increase (there is a positive
mass flowing into the domain from the outside), while the same on the right extreme makes mass decrease
(mass is escaping the domain). The same, with opposite effects, holds for negative fluxes.



This is why we use a flux-form (finite-volume) scheme instead of
differencing the PDE directly: **mass conservation is a structural property
of the discretization** (up to suitable boundary conditions), not something that has to be checked.

### 1.2 Reconstructing the flux at a face

On each face $x_i$ , we need to approximate the value of the flux by using the known
averaged values of $p(x,t), b(x,t)$ and $D(x,t)$. In particular, we choose the following
discretization

$$
J_t[x_i] = \frac{b_t[i+1] - b_t[i]}{dx}p_{\text{upwind}} - \frac{D_t[i+1]p_t[i+1] - D_t[i]p_t[i]}{\text{distance}} \qquad (5)
$$


where 

- $b_t[i], b_t[i+1], D_t[i], D_t[i+1], p_t[i] , p_t[i+1]$ denote the averaged values on $I_i$ and $I_{i+1}$ at time $t$.


- The advective term is **upwinded**. `p_upwind` is $p_t[i]$ if `b_face >= 0`
  (flow left-to-right) and $p_t[i+1]$ otherwise. Upwinding helps the explicit scheme remain stable when 
   advection dominates diffusion (see §1.4 for the details); a centered advection scheme which uses the average values of
  $p_t [i]$ and $p_t [i+1]$ would oscillate once advection dominates diffusion.
- The diffusive term is a centered difference of $D p$ between the two
  cells either side of the $i$-th face. 
- `distance` is either $dx$ or $dx/2$, depending on the boundary conditions. See §1.3.

In the codes, instead of using the index $i$, we use the notations $p_L, p_R$, which
denote the averaged values of $p(x,t)$ on the face on the left and on the right of the boundary point $x_i$
considered, respectively. Because both terms are linear in 
$(p_L, p_R)$, we can write  $J = c_L * p_L + c_R * p_R$ for
coefficients $(c_L, c_R)$ computed once per face (`_face_coeffs`). The entire
operator is assembled directly into a sparse matrix, face by face.

### 1.3 Boundary conditions, and why each rule is the *right* rule

- **Periodic**: the last and first cells are neighbors; treat that face
  exactly like an interior one.
- **(Homogeneous) Neumann**: physically, "reflecting"
  means *zero probability crosses this wall*. The correct discretization is therefore simply
  $J = 0$ at the boundary face. 
- **(Homogeneous) Dirichlet**: the density is fixed to $0$ *exactly at the
  face*, which sits $dx/2$ from the adjacent cell center, not a full $dx$,
  which is the spacing you'd use between two real cell centers.

### 1.4 Time discretization: stability analysis of forward and backward Euler \& numerical diffusion

Since the coefficients depend on space and time, a classical `Von Neumann` analysis is not directly applicable. 
Instead, we use a frozen-coefficient argument: at each grid point and time level, the coefficients are treated as 
locally constant and the corresponding Fourier amplification factor is computed. 
This yields a local stability criterion that must be enforced uniformly over the computational domain. For this reason, we 
also perform the error analysis using `Taylor` expansion using such frozen coefficients.
With $b,D$ constant, the Fokker-Planck equation reduces to 

$$
\partial_t p + b\partial_x p = D\partial_x^2 p. \qquad (6)
$$

We discretize on a uniform grid $x_j = j\Delta x$, $t^n = n\Delta t$,
writing $p_j^n$ for the numerical approximation to $p(x_j,t^n)$. (To avoid
a clash with the imaginary unit $i$ below, we index grid points by $j$ in
this section, rather than $i$ as in §1.1–§1.3). Taking
$b \ge 0$ for concreteness (the case $b<0$ is symmetric, using the
downwind neighbor and $|b|$ throughout), the forward-Euler, upwind +
centered-diffusion scheme reads

$$
\mathcal{N}_{\Delta t,\Delta x}[p(x_j, t_n)]=\frac{p_j^{n+1}-p_j^n}{\Delta t} + b\frac{p_j^n - p_{j-1}^n}{\Delta x} = D\frac{p_{j+1}^n - 2p_j^n + p_{j-1}^n}{\Delta x^2}. \quad (FE)
$$

#### 1.4.1 Taylor expansion analysis: consistency and truncation error

Given a discrete scheme $\mathcal{N}_{\Delta t,\Delta x}[p]$ intended to
approximate a PDE, the **local truncation error** at
$(x_j,t^n)$ is the residual obtained by substituting the *exact* solution $p$
of the continuous problem into the discrete equations:

$$
\tau_j^n := \mathcal{N}_{\Delta t,\Delta x} [p (x_j,t^n)].
$$

The scheme is **consistent** if $\tau_j^n \to 0$ as $\Delta t,\Delta x \to
0$ independently, and is of order $(p,q)$ in time and space if $\tau_j^n =
O(\Delta t^p) + O(\Delta x^q)$.

Apply this to (FE): substitute the exact solution $p(x,t)$ of (6) into
the discrete operator and Taylor-expand every term about $(x_j,t^n)$
(all derivatives below evaluated there):

$$
\frac{p(x_j,t^{n+1})-p(x_j,t^n)}{\Delta t} = \partial_t p + \frac{\Delta t}{2}\partial_t^2 p + O(\Delta t^2),
$$

$$
\frac{p(x_j,t^n)-p(x_{j-1},t^n)}{\Delta x} = \partial_x p - \frac{\Delta x}{2}\partial_x^2 p + O(\Delta x^2),
$$

$$
\frac{p(x_{j+1},t^n)-2p(x_j,t^n)+p(x_{j-1},t^n)}{\Delta x^2} = \partial_x^2 p + O(\Delta x^2).
$$

Substituting into $\tau_j^n$:

$$
\tau_j^n = \underbrace{\Big[\partial_t p + b\partial_x p - D\partial_x^2 p\Big]}_{=0,\ p \text{ solves } (6)} + \frac{\Delta t}{2}\partial_t^2 p - \frac{b\Delta x}{2}\partial_x^2 p  + O(\Delta t^2,\Delta x^2).
$$

The bracket vanishes because $p$ is the exact solution, leaving

$$
\tau_j^n = \frac{\Delta t}{2}\partial_t^2 p - \frac{b\Delta x}{2}\partial_x^2 p + O(\Delta t^2, \Delta x^2). 
$$

We substitute $\partial_t p = - b \partial_x p + D\partial_{xx}p$ twice in the duble time derivative to find:


$$\tau_j^n = \frac{b}{2} (b \Delta t - \Delta x) \partial_x^2 p  + O(\Delta t, \Delta x ^2).$$


Two conclusions follow:

**Consistency and order.** Since $\tau_j^n \to 0$ as $\Delta t,\Delta x \to
0$, the scheme is consistent with (6), and $\tau_j^n = O(\Delta t) +
O(\Delta x)$: first order in time and first order in space (the
diffusive term alone contributes a second-order error; upwinding is what
degrades the spatial order to one, we see shortly why we still decided to do the upwinding).

**Artificial diffusion.** The term $\frac{b}{2} (b \Delta t - \Delta x) \partial_x^2 p$ in has exactly the algebraic form of a diffusive term. Evaluated on
a smooth solution, the scheme therefore behaves, to leading order, as if
it were consistent not with $D$ but with an effective diffusivity

$$
D_{\text{eff}} = D -\frac{b}{2} (b \Delta t - \Delta x) = D + \frac{b}{2}(1- \lambda),
$$

with $\lambda$ the CFL number (introduced in the next section). It is interesting to notice that, if $b$ is constant, then
the numerical diffusion is eliminated considering $\Delta x = b \Delta t$. In a sense, contrary to what one might think,
sending $\Delta t$ or $\Delta x$ to $0$ my not always be the best choice, since it induces artificial diffusion. 
If we repeat the analysis for the backward Euler method w simply have to evaluate the right hand side
of (FE) at $t^{n+1}$ and the numerical diffusivity would be the same with $- \Delta t$ in place of $\Delta t$. Hence, the
effective diffusion coefficient would be

$$ D_{eff} = D+ \frac{b}{2} (b \Delta t + \Delta x) ,$$

with no possibility of making the artificial diffusion disappear.
**What this does and does not show.** This computation establishes
consistency and the order of accuracy: holding $\Delta t,\Delta x$ fixed
and letting them shrink independently, the residual of the *exact*
solution in the discrete equations vanishes at the stated rate. It says
**nothing** about whether small perturbations grow or decay under
repeated application of the scheme. A scheme can be perfectly consistent
(even high order) and still be unusable because errors amplify at every
step; that is a separate question, answered only by the stability analysis of the next section.


#### 1.4.2 Von Neumann stability analysis


We define the following quantities:

$$
\lambda := \frac{|b|\Delta t}{\Delta x} \qquad(\text{CFL number}),
$$

$$
\mu := \frac{D\Delta t}{\Delta x^2} \qquad(\text{diffusion number}).
$$



We look for solutions of (FE) of the form $p_j^n = \xi^n e^{i (j \Delta x) k}= \xi^n e^{i \theta j }$, where $\theta = k \Delta x \in (-\pi, \pi]$ is the phase advanced 
per grid point by wavenumber $k$. That is, we study how the numerical method (forward Euler in this case)
transforms a single spatial Fourier mode of $p(x,t)$. 


Substituting into (FE) we have:


$$ \frac{\xi^{n+1} e^{i (j \Delta x) k} - \xi^{n} e^{i (j \Delta x) k} }{\Delta t} + b \frac{\xi^{n} e^{i (j \Delta x) k}-\xi^{n} e^{i ((j-1) \Delta x) k}  }{\Delta x} =  D \frac{\xi^{n} e^{i ((j+1) \Delta x) k} -2\xi^{n} e^{i (j \Delta x) k} + \xi^{n} e^{i ((j-1) \Delta x) k}}{\Delta x^2},$$

which can be rewritten as

$$
\xi^{n+1} = \left[1 - \lambda\big(1-e^{-i\theta}\big) + \mu\big(e^{i\theta}-2+e^{-i\theta}\big)\right] \xi^n \implies \frac{\xi^{n+1}}{\xi^n}= \xi(\theta), \text{ an amplification factor}.
$$

Using $1-e^{-i\theta} = 2\sin^2(\theta/2) + i\sin\theta$ and
$e^{i\theta}-2+e^{-i\theta} = -4\sin^2(\theta/2)$:

$$
\xi(\theta) = 1 - 2(\lambda+2\mu)\sin^2 \Big(\frac{\theta}{2}\Big) - i\lambda\sin\theta.
$$

**Stability condition.** The scheme is stable iff $|\xi(\theta)|\le 1$ for
*every* $\theta$, otherwise that Fourier component of any
perturbation (round-off, or a mismatch in initial data) is amplified by a
factor $|\xi|>1$ at every step, so an arbitrarily small error grows like
$|\xi|^n \to \infty$. It can be found easily that the **exact** stability condition for (FE) is

$$
\boxed{\ \lambda + 2\mu \le 1\ } \qquad\Longleftrightarrow\qquad \frac{|b|\Delta t}{\Delta x} + \frac{2D\Delta t}{\Delta x^2} \le 1.
$$

This single bound recovers both limiting cases derived separately
elsewhere in these notes: setting $\mu=0$ (pure advection) gives $\lambda
\le 1$; setting $\lambda=0$ (pure diffusion) gives $\mu \le 1/2$.

#### Remark: the Péclet number

Define the **Péclet number**

$$
Pe_h := \frac{|b|\Delta x}{D}.
$$

A one-line calculation relates it to $\lambda,\mu$ exactly:

$$
Pe_h = \frac{\lambda}{\mu}.
$$

Unlike $\lambda,\mu$, $Pe_h$ does not depend on $\Delta t$ at
all: it is a purely spatial statement about how well the mesh resolves
the *local* competition between advection and diffusion, independent of
any time-stepping choice. For a *centered* difference
of the advective term (not the upwind scheme used here), large
$Pe_h$ produces genuine non-physical oscillations: centered
advection is non-monotone once $Pe_h$ exceeds roughly 2,
regardless of the time step. Because this project upwinds (§1.2), it is immune
to that particular pathology. Instead, large $Pe_h$ 
here signals that the artificial diffusion may become large compared to the true $D$: 
the effective diffusion $D_{eff}$ can be rewritten as

$$
D_{eff} = D\left(1 + \frac{Pe_h}{2}  (1- \lambda)\right).
$$

#### 1.4.3 Comparison of forward and backward Euler

**Forward Euler** is only *conditionally* stable: §1.4.2 shows the exact
threshold $\lambda+2\mu\le 1$.

**Backward Euler** evaluates the right-hand side of (FE) at the new time
level:

$$
\frac{p_j^{n+1}-p_j^n}{\Delta t} + b\frac{p_j^{n+1}-p_{j-1}^{n+1}}{\Delta x} = D\frac{p_{j+1}^{n+1}-2p_j^{n+1}+p_{j-1}^{n+1}}{\Delta x^2} \quad (BE).
$$

The same substitution $p_j^n = \xi^n e^{i\theta j}$ now gives $\xi$ on the
*left*-hand side of the equation defining it:

$$
\xi\Big[1 + 2a\sin^2 \Big(\frac{\theta}{2}\Big) + i\lambda\sin\theta\Big] = 1, \qquad a=\lambda+2\mu,
$$

so

$$
|\xi(\theta)|^2 = \frac{1}{\Big[1+2a\sin^2(\theta/2)\Big]^2 + \lambda^2\sin^2\theta}.
$$

Since $a,\lambda \ge 0$, the denominator is $\ge 1$ for every $\theta$,
hence $|\xi(\theta)|\le 1$ **for every** $\lambda,\mu\ge0$: that is,
backward Euler is unconditionally stable. There is no threshold on $\Delta t, \Delta x$ to violate.

This robustness is purely about **stability**, not accuracy: backward Euler is still
only first order in $\Delta t$, so a large $\Delta t$ still produces a
large truncation error even though the scheme will not diverge. More
importantly, the spatial part of the truncation error comes entirely from
the spatial discretization and does not involve the time integrator at
all. Switching to backward Euler removes the $\Delta t$ vs $\Delta x$
stability constraint, but it does nothing to reduce artificial diffusion:
$\Delta x$ still has to be small enough, relative to $|b|$ and $D$, for the
solution to be *accurate*, whichever time integrator is used.

#### 1.4.4 Synthesis

The analyses above answer different questions, and none can
substitute for another:

- **Taylor expansion (§1.4.1)** proves *consistency* and gives the *order
  of accuracy* ($O(\Delta t) + O(\Delta x)$ here). It says nothing about error 
  growth over many steps.
- **Von Neumann analysis (§1.4.2)** gives an exact threshold, $\lambda + 2\mu \le 1$ for forward
  Euler, none for backward Euler.

---

## Part II — The Lagrangian route: Euler-Maruyama

### 2.1 Trading a field for an ensemble of paths

Instead of tracking $p(x,t)$ on a grid, imagine releasing many independent
particles, each moving under the SDE

$$
dX_t = b(X_t, t)dt + \sigma(X_t, t)dW_t, \quad \sigma = \sqrt{2D(x,t)}
$$

and ask: at time $t$, what does the *distribution* of $X_t$ look like
across the whole ensemble? The claim is that this distribution is exactly $p(x, t)$, the solution of
the Fokker-Planck equation from Part I. If that's true, we never need to
build a grid or a matrix; we just need to simulate paths and make a
histogram.

### 2.2 The one new rule of stochastic calculus

The $W_t$ is a stochastic process called `Brownian motion`, characterized by: $W_0 = 0$, independent increments, and
$W_{t+dt} - W_t \sim N(0, dt)$. You can think that $dW = W_{t+dt} - W_t$.

Here is the fact that makes stochastic calculus different from ordinary
calculus: $(dW)^2 \approx dt$, not a smaller, negligible quantity. Indeed $E[(dW)^2] = Var(dW) = dt$ by definition, 
and one can show that $Var((dW)^2) = O(dt^2)$, so as $dt \to 0$, $(dW)^2$ concentrates around its
mean $dt$ with vanishing relative spread. This single replacement rule is the core of **Ito's
calculus**.

### 2.3 Ito's formula: the stochastic chain rule

Take any smooth function $f$ and Taylor-expand $f_t=f(X_{t+dt})$ around $X_t$,
keeping terms up to the order that matters ($dt$ and $dW^2 $, both of size
$dt$; drop $dt^2$, $dt dW$, and higher):

$$
df_t = f_t'(X_t) dX_t + \frac{1}{2}f_t''(X_t) (dX_t)^2 + ...
$$

With $dX_t = b(X_t,t)dt + \sigma(X_t,t)dW_t$:

$$
E[(dX_t)^2] = \sigma(X_t,t)^2E[(dW_t)^2] + 2b(X_t,t)\sigma(X_t,t)dtE[dW_t] + b(X_t,t)^2 dt^2 \approx \sigma(X_t,t)^2dt \quad \text{(using } E[(dW_t)^2] \approx dt\text{)}
$$

So Ito's formula reads

$$
df_t = \left[ b(X_t,t) f_t'(X_t) + D(X_t,t) f_t''(X_t) \right] dt + \sigma(X_t,t) f'(X_t)dW_t
$$

Define the **generator** of the process,

$$
L f = b f' + D f''
$$

so that  we can write 

$$df_t = (Lf_t)dt + \sigma(X_t)f_t''dW_t.$$

### 2.4 From one path to the whole density: deriving Fokker-Planck

Take expectations of Ito's formula. Since $E[dW_t] = 0$ (Brownian increments
average to zero), the stochastic term cancels out, and exchanging the order of derivative and integration (i.e. expected value):

$$
\frac{d}{dt}E[f(X_t)] = E[ L f(X_t) ] = E[ b(X_t,t) f'(X_t) + D(X_t,t) f''(X_t) ].
$$

By definition, the density $p(x,t)$ is a function such that $E[f(X_t)] = \int f(x) p(x,t) dx$. Differentiate the left side 
under the integral sign, and rewrite the right side as an integral
against $p(x,t)$ too:

$$
\int f \partial_t p(x,t)dx =  \int [ b (x,t) f'(x,t) + D(x,t) f''(x,t) ] p(x,t) dx.
$$

Integrate the right-hand side by parts (once for the $b f' p$ term, twice
for the $D f'' p$ term), throwing the derivatives onto $p$ instead of $f$. Denote the domain of integration as $I=(\alpha,\beta)$ , 
with $\alpha,\beta$ eventually equal to $\pm \infty$, we have:

$$
\int_{I} (b p) f' dx = [(bp) f]_\alpha ^{\beta} -  \int_{I} (b p)' f dx 
$$

and 

$$
\int_{I} (D p) f'' dx = [(Dp) f']_\alpha ^{\beta} - [(Dp)' f]_\alpha ^\beta  + \int_{I} (D p)'' f dx. 
$$

We finally have:


$$
\int_I f \partial_t p dx = \int_I f [ -(bp)' + (Dp)'' ]  dx + \text{ Boundary terms (B.T.)} \qquad \text{ for every f smooth} \quad (7).
$$

First of all, let us see the effect of the boundary conditions considered: periodic, homogeneous Neumann, homogeneous Dirichlet.
The boundary terms are:

$$ B.T. = [(bp) f]_\alpha ^{\beta} + [(Dp) f']_\alpha ^{\beta} - [(Dp)' f]_\alpha ^\beta.$$

Hence:
  -  Considering periodic BC the B.T. vanish in a straightforward way.
  -  Considering homogeneous Neumann BC, we impose that $J(x,t)= bp - \partial_x (Dp)$ vanishes at $\alpha$ and $\beta$. In this case
     it still holds that the B.T. is zero, provided we restrict to functions $f$ such that $f'(\alpha)= f' (\beta)=0$. This is not
     restrictive for our next claim
  -  Considering homogeneous Dirichlet conditions, we are left with a term $[(b p) -(Dp)' f]_\alpha ^\beta$. This residual term is 
     responsible for the eventual loss of mass in this case.
  

Since (7) holds for every smooth function $f$ such that $f'(\alpha)= f'(\beta)=0$, by the fundamental lemma of calculus of variations:

$$\partial_t p (x,t) + \partial_x (b(x,t)p(x,t)) - \partial_x ^2 (D(x,t)p(x,t))=0,$$

which is exactly Fokker-Planck equation from Part I. In the cse of homogeneous Dirichlet BC, we still get that $p(x,t)$ solves Fokker-Planck
at interior points (for instance restricting to compactly supported test functions $f$), but considering $J= (bp) - (Dp)'$:


$$\frac{d}{dt} \int_I p(x,t)= \int_I \partial_x  J(x,t)dx= -[J(x,t)]_{\alpha} ^\beta \leq 0 \implies \text{ The mass of p decreases.}$$


### 2.5 Discretizing the SDE: Euler-Maruyama

Just as forward Euler discretizes an ODE by freezing the right-hand side at
the start of the step, **Euler-Maruyama** discretizes the SDE the same way:

$$
X_{n+1} = X_n + b(X_n, t_n) dt + \sqrt{2 D(X_n, t_n) dt}  Z_n, \quad \text{ with }  Z_n \sim N(0,1)
$$

Two different notions of convergence matter here, and
they behave very differently:

- **Strong error** (path-wise accuracy, `E|X_n - X_{t_n}|`): Euler-Maruyama
  is only order `1/2` (noticeably worse than the ODE case) because the
  random increment itself carries $O(\sqrt{dt})$ fluctuations that no
  amount of Taylor-expansion cleverness removes at fixed $dt$.
- **Weak error** (accuracy of *statistics*, e.g. `|E[f(X_n)] - E[f(X_{t_n})]|`):
  order $1$, the same as forward Euler for ODEs. The reason is that the random noise is "filtered out"
  thanks to Ito's formula.

Since a density estimated from a histogram is exactly a statistic
(essentially $E[\mathbf{1}_{x \in \text{bin}}]$ for each bin), it is the **weak** error that
controls how good the reconstructed $p(x,t)$ is, which is the more
forgiving of the two, and part of why the simple Euler-Maruyama scheme
(rather than something more elaborate) is a reasonable choice.

### 2.6 Why simulate particles at all — robustness to rough coefficients

Part I's flux formulas consider differences of $b$ and $D$ between neighboring cells; this
implicitly assumes they vary smoothly at the grid scale. If $b$ or $D$ are
discontinuous (e.g. diffusion jumping across a material interface) or only
known empirically/noisily, that differencing has no clean meaning, and the
finite-volume solution can pick up spurious oscillations near the
irregularity. Euler-Maruyama, by contrast, only ever *evaluates* $b$ and
$D$ at particle positions: it never differentiates them. The price is Monte Carlo noise, i.e. an
error $O(1/sqrt(n_{trials}))$ by the central limit theorem.

### 2.7 Boundary conditions as particle rules

The Eulerian boundary conditions of Part I have exact particle counterparts,
and it's worth seeing why each pairing holds:

| PDE condition | Particle rule | Why it matches |
|---|---|---|
| Periodic | wrap position around | the domain is a ring; a particle leaving one side *is*, physically, the same particle entering the other |
| Neumann (`J=0`) | reflect off the wall | zero probability current means no particle may actually cross; folding its position back in enforces this pathwise |
| Dirichlet (fixed `g`) | remove particle on exit | the PDE lets mass drain away at a fixed boundary value; a particle that exits has, by construction, left the domain's probability budget for good |

Since both solvers start from the *same* histogram-derived initial
density (`fp1d/initial_conditions.py`), the two descriptions can be run
side by side on identical initial data and compared directly — which is
exactly the cross-validation this project's test suite performs.


## 3. Comparing the two pictures

| | Finite volume (Eulerian) | Euler-Maruyama (Lagrangian) |
|---|---|---|
| Unknown | cell-average density on a grid | positions of many particles |
| Error source | flux approximation (`O(dx)` here) + time truncation | statistical (Monte Carlo) noise, `O(1/sqrt(n_trials))` |
| Cost  | grid resolution `n`, linear solve (implicit) | number of trajectories, number of steps |
| Handles rough `b, D`? | degrades (needs smoothness at grid scale) | robust (never differentiates coefficients) |
| Exactly conserves mass? | yes, structurally (telescoping flux sum) | yes in expectation; finite-sample histograms carry sampling noise |
| Best suited to | smooth, well-characterized coefficients, when you want a numerically precise field | irregular/discontinuous coefficients, or when trajectory-level information (not just the density) is of interest |

## 4. Where to go from here

- **Finite volume methods, general theory**: LeVeque, *Finite Volume
  Methods for Hyperbolic Problems*.
- **Fokker-Planck equation, physics perspective**: Risken, *The
  Fokker-Planck Equation*; Gardiner, *Stochastic Methods*.
- **Numerical SDEs, including convergence orders**: Kloeden & Platen,
  *Numerical Solution of Stochastic Differential Equations*.
- **Ito calculus and Ito's formula**: any
  introductory stochastic calculus text (e.g. Øksendal, *Stochastic
  Differential Equations*, Chapter 3–4 or Baldi, *Stochastic Calculus*) works through the argument above
  with full rigor.
