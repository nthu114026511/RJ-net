$
\rho = \frac{\rho_0}{J} \\
\Leftrightarrow M := \rho J \\
\dot{M} = 0 \Leftrightarrow M|_{t_0} = \text{const.} \\
(\rho J)^{\cdot} = (\rho_t + \nabla \cdot (\rho \vec{u})) \underset{\neq 0}{J} \Leftrightarrow \rho_t + \nabla \cdot (\rho \vec{u}) = 0
$

$
\underline{\text{How about } \dot{M} \neq 0? \quad \text{e.g. Mass-action?}} \\
\\
\rho_t + \nabla \cdot (\rho \vec{u}) = \sum \underline{\vec{r}} \\
\\
\text{Kinematics: } \rho = \rho_0 + \sum \vec{R} \\
\\
M = \underline{m(t)} = \int_{\Omega} \rho(x,t) dx \stackrel{\text{C.V.T.}}{=} \int_{\Omega_{\bar{x}}} \underbrace{\rho(\underline{x(\bar{x},t)},t) \det\left[ \frac{\partial x}{\partial \bar{x}} \right]}_{\rho_0(\bar{x}), J(\bar{x},t)} d\bar{x} \\
\\
\text{if } ||| \qquad x=x(\bar{x},t)
$

$
\underline{m|_{t=0}} = \int_{\Omega_{\bar{x}}} \underbrace{\rho(\bar{x}, 0)}_{= \rho_0(\bar{x}) = \rho_0} d\bar{x} \qquad \boxed{\rho = \frac{\rho_0}{J}}
$

$
\boxed{\rho = \rho_0 + \Sigma \vec{R}} \\
m(t) = \int_{\Omega} \rho(x,t) dx = \int_{\Omega_{\bar{x}}} (\rho_0(\bar{x}) + \Sigma \vec{R}(t, \bar{x})) d\bar{x} \\
\begin{array}{ccc}
\downarrow \frac{d}{dt} & & \downarrow \frac{d}{dt} \\
\int_{\Omega_{\bar{x}}} (\rho_t + \nabla \cdot (\rho u)) J d\bar{x} & & \int_{\Omega_{\bar{x}}} \Sigma \vec{r} \, d\bar{x} \\
 & & \| \\
= \int_{\Omega} \underline{\rho_t + \nabla \cdot (\rho u)} \, dx & & \int_{\Omega} \underline{\Sigma \vec{r}} \, dx
\end{array} \\
\rho_t + \nabla \cdot (\rho u) = \Sigma \vec{r} \quad , \quad \vec{r} = \vec{r}(\vec{\rho})
$

e.g: $\vec{\rho} = \begin{bmatrix} B \\ R \\ E \end{bmatrix}$

$\vec{u}_B \equiv \vec{u}_R \equiv \vec{u}_E \equiv 0. \quad (ODE)$

? $\vec{r} = \vec{r}_{mn}(\rho) \qquad$ cf $\qquad \vec{u} = \vec{u}_{mn}(\rho)$

$\dot{B} = r B\left(1-\frac{B}{K}\right) = rB - \frac{r}{K} B^2$

$B \underset{k_{-}}{\stackrel{k_{+}}{\rightleftharpoons}} 2B$

$\dot{B} = ^{(+1)}(k_{+}B - k_{-}B^2)$

$
\text{For } n=0,1,2 \dots \\
u^{n+1} = u_{NN}(\rho^n) \\
r_i^{n+1} = r_{i,NN}^n(\vec{\rho}^n, x) \\
R_i^{n+1} = R_i^n + \Delta t (r_i^n) \\
J^{n+1} = J^n + \Delta t (\nabla \cdot u) J^n - u \cdot \nabla J \\
\rho^{n+1} = \frac{\rho_0 + R^{n+1}}{J^{n+1}}
$

$
\rho_t = \Delta \rho + (k^+ \rho - k^- \rho^2) \\
r(\rho) = k^+ \rho - k^- \rho^2 \\
\underline{x_t = \frac{x^{n+1}-x^n}{\Delta t} \approx u} \\
R_t = \frac{R^{n+1}-R^n}{\Delta t} \approx r_{NN} \\
r_{NN} = r(\rho) \\
\underline{\underline{R(t=0)=0.}} \\
V_{PINN} \\
\underline{\underline{\text{Wei}}}: \quad \underline{\rho_t = \Delta \rho + \rho - \rho^2} \quad \underline{\rho_{t=0} = \rho_0}
$