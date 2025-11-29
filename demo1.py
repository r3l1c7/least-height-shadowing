#!/usr/bin/env python3

from fractions import Fraction
import math

# ------------------------------------------------------------
# Height function on rationals
# ------------------------------------------------------------

def height_log(frac: Fraction) -> float:
    num = frac.numerator
    den = frac.denominator
    return math.log(max(abs(num), den))


# ------------------------------------------------------------
# Cost functional and least-height projection
# ------------------------------------------------------------

def cost(y: Fraction, z: Fraction, lam: float, beta: float) -> float:
    diff = float(y - z)
    return diff * diff + lam * math.exp(beta * height_log(z))


def least_height_projection(
    y: Fraction,
    lam: float,
    beta: float,
    q_max: int = 200,
    neighbor_range: int = 5,
) -> Fraction:
    y_float = float(y)
    best_z = None
    best_c = None

    for q in range(1, q_max + 1):
        p_center = int(round(y_float * q))
        for delta in range(-neighbor_range, neighbor_range + 1):
            p = p_center + delta
            z = Fraction(p, q)
            c = cost(y, z, lam, beta)
            if best_c is None or c < best_c:
                best_c = c
                best_z = z

    return best_z


# ------------------------------------------------------------
# Example map: logistic map f(x) = 4 x (1 - x)
# ------------------------------------------------------------

def f_logistic(x: Fraction) -> Fraction:
    # r = 3.9 instead of 4 - avoids the 0 fixed point trap
    return Fraction(39, 10) * x * (1 - x)


def f_quadratic(x: Fraction) -> Fraction:
    # x^2 + 1/4 - classic height explosion example
    return x * x + Fraction(1, 4)


# ------------------------------------------------------------
# Simulation and reporting
# ------------------------------------------------------------

def simulate(
    x0: Fraction,
    steps: int,
    lam: float,
    beta: float,
    q_max: int = 200,
    neighbor_range: int = 5,
    f_map = None,
):
    if f_map is None:
        f_map = f_quadratic
        
    shadow_orbit = [x0]
    exact_orbit = [x0]
    errors = [0.0]
    heights_exact = [height_log(x0)]
    heights_shadow = [height_log(x0)]

    x_shadow = x0
    x_exact = x0

    for n in range(steps):
        x_exact = f_map(x_exact)
        exact_orbit.append(x_exact)
        heights_exact.append(height_log(x_exact))

        y = f_map(x_shadow)
        z = least_height_projection(y, lam, beta, q_max, neighbor_range)
        x_shadow = z
        shadow_orbit.append(x_shadow)
        heights_shadow.append(height_log(x_shadow))

        err = abs(float(x_exact - x_shadow))
        errors.append(err)

    return exact_orbit, shadow_orbit, heights_exact, heights_shadow, errors


def format_fraction(frac: Fraction, max_chars: int = 30, height_threshold: float = 50.0) -> str:
    h = height_log(frac)
    if h > height_threshold:
        return f"{float(frac):.6g}"
    s = f"{frac.numerator}/{frac.denominator}"
    if len(s) > max_chars:
        return s[:max_chars - 3] + "..."
    return s


def run_experiment():
    print("=" * 85)
    print("EXPERIMENT A: Quadratic map f(x) = x² + 1/4")
    print("=" * 85)
    
    x0 = Fraction(1, 3)
    steps = 15

    lam = 1e-8
    beta = 0.3
    q_max = 300
    neighbor_range = 5

    exact_orbit, shadow_orbit, h_exact, h_shadow, errors = simulate(
        x0, steps, lam, beta, q_max, neighbor_range, f_quadratic
    )

    print(f"x0 = {x0}, lambda = {lam}, beta = {beta}, q_max = {q_max}")
    print()
    print(f"{'n':>2}  {'exact_x':>20}  {'shadow_x':>20}  {'h_exact':>10}  {'h_shadow':>8}  {'error':>12}")
    print("-" * 85)
    
    for n in range(0, steps + 1):
        ex = format_fraction(exact_orbit[n])
        sh = format_fraction(shadow_orbit[n])
        print(
            f"{n:2d}  {ex:>20}  {sh:>20}  {h_exact[n]:10.1f}  {h_shadow[n]:8.2f}  {errors[n]:12.2e}"
        )
    
    print()
    print(f"Final exact height:  {h_exact[-1]:,.0f}")
    print(f"Max shadow height:   {max(h_shadow):.2f}")
    print(f"Height compression:  {h_exact[-1] / max(max(h_shadow), 0.1):,.0f}x")
    print(f"Max tracking error:  {max(errors):.2e}")
    
    print()
    print("=" * 85)
    print("EXPERIMENT B: Logistic map f(x) = 3.9·x·(1-x)")
    print("=" * 85)
    
    x0 = Fraction(1, 5)
    steps = 12
    
    lam = 1e-9
    beta = 0.2
    q_max = 500
    neighbor_range = 8

    exact_orbit, shadow_orbit, h_exact, h_shadow, errors = simulate(
        x0, steps, lam, beta, q_max, neighbor_range, f_logistic
    )

    print(f"x0 = {x0}, lambda = {lam}, beta = {beta}, q_max = {q_max}")
    print()
    print(f"{'n':>2}  {'exact_x':>20}  {'shadow_x':>20}  {'h_exact':>10}  {'h_shadow':>8}  {'error':>12}")
    print("-" * 85)
    
    for n in range(0, steps + 1):
        ex = format_fraction(exact_orbit[n])
        sh = format_fraction(shadow_orbit[n])
        print(
            f"{n:2d}  {ex:>20}  {sh:>20}  {h_exact[n]:10.1f}  {h_shadow[n]:8.2f}  {errors[n]:12.2e}"
        )
    
    print()
    print(f"Final exact height:  {h_exact[-1]:,.0f}")
    print(f"Max shadow height:   {max(h_shadow):.2f}")
    print(f"Height compression:  {h_exact[-1] / max(max(h_shadow), 0.1):,.0f}x")
    print(f"Max tracking error:  {max(errors):.2e}")


if __name__ == "__main__":
    run_experiment()
