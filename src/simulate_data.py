import numpy as np
import pandas as pd


def simulate_equipment(n_units=120, min_cycles=110, max_cycles=260, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    for unit_id in range(1, n_units + 1):
        life = int(rng.integers(min_cycles, max_cycles + 1))
        base_temp = rng.normal(72, 2)
        base_vibration = rng.normal(0.22, 0.02)
        base_pressure = rng.normal(100, 2)
        base_current = rng.normal(35, 1.5)

        for cycle in range(1, life + 1):
            age = cycle / life
            nonlinear = age ** 2
            temperature = base_temp + 18 * nonlinear + rng.normal(0, 1.1)
            vibration = base_vibration + 0.75 * nonlinear + rng.normal(0, 0.045)
            pressure = base_pressure - 18 * nonlinear + rng.normal(0, 1.6)
            motor_current = base_current + 9 * nonlinear + rng.normal(0, 1.1)
            efficiency = 0.95 - 0.28 * nonlinear + rng.normal(0, 0.018)
            rows.append({
                "unit_id": unit_id, "cycle": cycle, "temperature": temperature,
                "vibration": vibration, "pressure": pressure,
                "motor_current": motor_current, "efficiency": efficiency,
                "rul": life - cycle,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = simulate_equipment()
    df.to_csv("data/simulated_equipment.csv", index=False)
    print(df.head())
    print(f"Generated {len(df):,} rows across {df['unit_id'].nunique()} units.")
