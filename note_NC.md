$
\begin{array}{c}
\boxed{(x, \rho^n) \rightarrow \text{NN} \rightarrow u^n} \\
\Big\downarrow \\
\boxed{J^{n+1} = J^n + \Delta t (\nabla \cdot u^n)J^n} \\
\Big\downarrow \\
\boxed{\rho^{n+1} = \frac{\rho_0}{J^{n+1}}} \\
\Big\downarrow \\
\boxed{\text{Loss} = \sum_{n=0}^{NT-1} [ ||\rho_t^{n+1} - \rho_{xx}^{n+1}||^2 + \rho^{n+1}(0,t)^2 + \rho^{n+1}(1,t)^2 ]}
\end{array}
$