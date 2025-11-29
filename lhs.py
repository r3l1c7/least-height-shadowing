"""
Least-Height Shadowing Algorithm

A method for controlling height explosion in arithmetic dynamics by
projecting iterates onto nearby low-height rationals.

Author: [YOUR NAME]
License: MIT
"""

from fractions import Fraction
import math
from typing import Callable, List, Dict, Optional


def height(z: Fraction) -> float:
    """
    Logarithmic Weil height of a rational number.
    
    For z = p/q in lowest terms: h(z) = log(max(|p|, |q|))
    
    Args:
        z: A rational number as a Fraction
        
    Returns:
        The logarithmic height (base e)
    """
    if z == 0:
        return 0.0
    return math.log(max(abs(z.numerator), z.denominator))


def cost(y: Fraction, z: Fraction, lam: float, beta: float) -> float:
    """
    Cost functional for the least-height projection.
    
    J(y, z) = ||y - z||² + λ·exp(β·h(z))
    
    Args:
        y: Target value (exact iterate)
        z: Candidate approximation
        lam: Height penalty coefficient (λ)
        beta: Height penalty exponent (β)
        
    Returns:
        Total cost combining accuracy and complexity penalty
    """
    accuracy_cost = (float(y) - float(z)) ** 2
    height_penalty = lam * math.exp(beta * height(z))
    return accuracy_cost + height_penalty


def least_height_projection(
    y: Fraction,
    lam: float,
    beta: float,
    q_max: int = 200,
    neighbor_range: int = 5,
) -> Fraction:
    """
    Compute the height-regularized projection Π(y).
    
    Finds: argmin_{z ∈ ℚ} [ ||y - z||² + λ·exp(β·h(z)) ]
    
    by searching rationals with denominators 1 to q_max.
    
    Args:
        y: Target value to approximate
        lam: Height penalty coefficient (smaller = more accurate, higher heights)
        beta: Height penalty exponent (smaller = gentler penalty curve)
        q_max: Maximum denominator to consider
        neighbor_range: How many numerators to check around the optimal
        
    Returns:
        The optimal low-height rational approximation
    """
    y_float = float(y)
    best_z = Fraction(round(y_float), 1)
    best_cost = cost(y, best_z, lam, beta)

    for q in range(1, q_max + 1):
        p_center = round(y_float * q)
        for delta in range(-neighbor_range, neighbor_range + 1):
            p = p_center + delta
            z = Fraction(p, q)
            c = cost(y, z, lam, beta)
            if c < best_cost:
                best_cost = c
                best_z = z

    return best_z


def iterate_exact(
    f: Callable[[Fraction], Fraction],
    x0: Fraction,
    steps: int
) -> List[Fraction]:
    """
    Iterate a map exactly over rationals.
    
    Warning: Heights will explode exponentially for most maps.
    
    Args:
        f: The rational map to iterate
        x0: Initial condition
        steps: Number of iterations
        
    Returns:
        List of orbit points [x0, f(x0), f²(x0), ...]
    """
    orbit = [x0]
    x = x0
    for _ in range(steps):
        x = f(x)
        orbit.append(x)
    return orbit


def iterate_shadow(
    f: Callable[[Fraction], Fraction],
    x0: Fraction,
    steps: int,
    lam: float,
    beta: float,
    q_max: int = 200,
    neighbor_range: int = 5,
) -> List[Fraction]:
    """
    Iterate with least-height shadowing: x_{n+1} = Π(f(xₙ))
    
    Args:
        f: The rational map to iterate
        x0: Initial condition
        steps: Number of iterations
        lam: Height penalty coefficient
        beta: Height penalty exponent
        q_max: Maximum denominator for projection
        neighbor_range: Search width for numerators
        
    Returns:
        List of shadow orbit points
    """
    orbit = [x0]
    x = x0
    for _ in range(steps):
        y = f(x)
        x = least_height_projection(y, lam, beta, q_max, neighbor_range)
        orbit.append(x)
    return orbit


def simulate(
    x0: Fraction,
    steps: int,
    lam: float,
    beta: float,
    f_map: Callable[[Fraction], Fraction],
    q_max: int = 200,
    neighbor_range: int = 5,
) -> Dict:
    """
    Run both exact and shadow iterations and compare.
    
    Args:
        x0: Initial condition
        steps: Number of iterations
        lam: Height penalty coefficient
        beta: Height penalty exponent
        f_map: The rational map to iterate
        q_max: Maximum denominator for projection
        neighbor_range: Search width for numerators
        
    Returns:
        Dictionary containing:
        - exact_orbit: List of exact iterates
        - shadow_orbit: List of shadow iterates
        - heights_exact: Heights of exact orbit
        - heights_shadow: Heights of shadow orbit
        - errors: |exact - shadow| at each step
    """
    exact_orbit = [x0]
    shadow_orbit = [x0]
    heights_exact = [height(x0)]
    heights_shadow = [height(x0)]
    errors = [0.0]

    x_exact = x0
    x_shadow = x0

    for _ in range(steps):
        # Exact iteration
        x_exact = f_map(x_exact)
        exact_orbit.append(x_exact)
        heights_exact.append(height(x_exact))

        # Shadow iteration
        y = f_map(x_shadow)
        x_shadow = least_height_projection(y, lam, beta, q_max, neighbor_range)
        shadow_orbit.append(x_shadow)
        heights_shadow.append(height(x_shadow))

        # Tracking error
        errors.append(abs(float(x_exact) - float(x_shadow)))

    return {
        'exact_orbit': exact_orbit,
        'shadow_orbit': shadow_orbit,
        'heights_exact': heights_exact,
        'heights_shadow': heights_shadow,
        'errors': errors,
    }


# =============================================================================
# Example Maps
# =============================================================================

def quadratic_map(c: Fraction) -> Callable[[Fraction], Fraction]:
    """Return f(x) = x² + c"""
    def f(x: Fraction) -> Fraction:
        return x * x + c
    return f


def logistic_map(r: Fraction) -> Callable[[Fraction], Fraction]:
    """Return f(x) = r·x·(1-x)"""
    def f(x: Fraction) -> Fraction:
        return r * x * (Fraction(1) - x)
    return f


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LEAST-HEIGHT SHADOWING DEMO")
    print("=" * 70)
    
    # Quadratic map f(x) = x² + 1/4
    f = quadratic_map(Fraction(1, 4))
    x0 = Fraction(1, 3)
    
    results = simulate(
        x0=x0,
        steps=15,
        lam=1e-8,
        beta=0.3,
        f_map=f,
        q_max=300,
        neighbor_range=5,
    )
    
    print(f"\nMap: f(x) = x² + 1/4")
    print(f"Initial: x₀ = {x0}")
    print(f"Parameters: λ = 1e-8, β = 0.3, q_max = 300\n")
    
    print(f"{'Step':<6} {'Shadow Value':<20} {'Exact Height':<15} {'Shadow Height':<15} {'Error':<12}")
    print("-" * 70)
    
    for i in range(len(results['shadow_orbit'])):
        shadow = results['shadow_orbit'][i]
        h_ex = results['heights_exact'][i]
        h_sh = results['heights_shadow'][i]
        err = results['errors'][i]
        
        # Format shadow value
        if h_sh < 10:
            sv = f"{shadow.numerator}/{shadow.denominator}"
        else:
            sv = f"{float(shadow):.6f}"
        
        print(f"{i:<6} {sv:<20} {h_ex:<15.1f} {h_sh:<15.2f} {err:<12.2e}")
    
    print("\n" + "=" * 70)
    h_final_exact = results['heights_exact'][-1]
    h_max_shadow = max(results['heights_shadow'])
    compression = h_final_exact / h_max_shadow
    
    print(f"Final exact height:    {h_final_exact:,.0f}")
    print(f"Max shadow height:     {h_max_shadow:.2f}")
    print(f"Height compression:    {compression:,.0f}×")
    print(f"Max tracking error:    {max(results['errors']):.2e}")
    print("=" * 70)
