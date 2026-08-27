"""Generate all synthetic datasets used by the practice projects.

Run once from the scikit-learn/ root:
    python data/raw/generate_datasets.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_student_performance(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    study_hours     = rng.normal(5, 2.5, n).clip(0, 12)
    attendance_pct  = rng.normal(75, 15, n).clip(40, 100)
    prev_score      = rng.normal(65, 12, n).clip(30, 100)
    parent_edu      = rng.choice(['none', 'high_school', 'bachelors', 'masters'], n,
                                  p=[0.1, 0.4, 0.35, 0.15])
    has_tutoring    = rng.choice([0, 1], n, p=[0.65, 0.35])
    internet_access = rng.choice([0, 1], n, p=[0.2, 0.8])
    gender          = rng.choice(['M', 'F'], n)

    # Missing values
    study_hours_m = study_hours.copy()
    study_hours_m[rng.choice(n, 80, replace=False)] = np.nan
    prev_score_m = prev_score.copy()
    prev_score_m[rng.choice(n, 60, replace=False)] = np.nan

    edu_map = {'none': 0, 'high_school': 1, 'bachelors': 2, 'masters': 3}
    edu_num = pd.Series(parent_edu).map(edu_map)

    noise = rng.normal(0, 5, n)
    final_score = (
        0.35 * study_hours +
        0.25 * (attendance_pct / 10) +
        0.30 * (prev_score / 10) +
        2.0  * edu_num +
        3.0  * has_tutoring +
        1.5  * internet_access +
        noise
    ).clip(0, 100)

    df = pd.DataFrame({
        'study_hours_per_day': study_hours_m.round(2),
        'attendance_pct':      attendance_pct.round(1),
        'prev_score':          prev_score_m.round(1),
        'parent_education':    parent_edu,
        'has_tutoring':        has_tutoring,
        'internet_access':     internet_access,
        'gender':              gender,
        'final_score':         final_score.round(1),
    })
    path = OUTPUT_DIR / 'student_performance.csv'
    df.to_csv(path, index=False)
    print(f"  student_performance.csv: {df.shape} → {path}")
    return df


def generate_customer_churn(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    tenure          = rng.integers(1, 72, n)
    monthly_charges = rng.normal(65, 30, n).clip(18, 120)
    total_charges   = tenure * monthly_charges * rng.uniform(0.9, 1.1, n)
    contract        = rng.choice(['month-to-month', 'one-year', 'two-year'], n,
                                  p=[0.55, 0.25, 0.20])
    payment_method  = rng.choice(['electronic_check', 'mailed_check',
                                   'bank_transfer', 'credit_card'], n)
    paperless       = rng.choice([0, 1], n, p=[0.4, 0.6])
    support_calls   = rng.integers(0, 10, n)
    internet_svc    = rng.choice(['DSL', 'fiber_optic', 'no_internet'], n,
                                  p=[0.35, 0.45, 0.20])

    contract_map = {'month-to-month': 0, 'one-year': 1, 'two-year': 2}
    contract_num = pd.Series(contract).map(contract_map)
    logit = (
        -0.04 * tenure +
        0.015 * monthly_charges +
        -0.8 * contract_num +
        0.15 * support_calls +
        0.3 * paperless +
        rng.normal(0, 0.5, n)
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, n) < prob_churn).astype(int)

    # Missing values
    total_charges_m = total_charges.copy().astype(float)
    total_charges_m[rng.choice(n, 50, replace=False)] = np.nan

    df = pd.DataFrame({
        'tenure':           tenure,
        'monthly_charges':  monthly_charges.round(2),
        'total_charges':    total_charges_m.round(2),
        'contract':         contract,
        'payment_method':   payment_method,
        'paperless_billing': paperless,
        'support_calls':    support_calls,
        'internet_service': internet_svc,
        'churn':            churn,
    })
    path = OUTPUT_DIR / 'customer_churn.csv'
    df.to_csv(path, index=False)
    print(f"  customer_churn.csv:      {df.shape} → {path}")
    return df


def generate_customer_segmentation(n=800, seed=42):
    rng = np.random.default_rng(seed)
    annual_income   = rng.normal(60000, 25000, n).clip(15000, 200000)
    spending_score  = rng.integers(1, 101, n)
    age             = rng.integers(18, 70, n)
    num_purchases   = rng.integers(1, 50, n)
    avg_order_value = rng.normal(150, 80, n).clip(10, 600)
    gender          = rng.choice(['M', 'F', 'Other'], n, p=[0.48, 0.50, 0.02])
    region          = rng.choice(['North', 'South', 'East', 'West'], n)

    df = pd.DataFrame({
        'annual_income':    annual_income.round(0).astype(int),
        'spending_score':   spending_score,
        'age':              age,
        'num_purchases':    num_purchases,
        'avg_order_value':  avg_order_value.round(2),
        'gender':           gender,
        'region':           region,
    })
    path = OUTPUT_DIR / 'customer_segmentation.csv'
    df.to_csv(path, index=False)
    print(f"  customer_segmentation.csv: {df.shape} → {path}")
    return df


def generate_end_to_end_dataset(n=1500, seed=42):
    """Employee salary prediction — comprehensive end-to-end dataset."""
    rng = np.random.default_rng(seed)

    age          = rng.integers(22, 60, n)
    years_exp    = np.clip(age - 22 + rng.normal(0, 3, n), 0, 38).astype(int)
    education    = rng.choice(['bachelor', 'master', 'phd', 'associate'], n,
                               p=[0.45, 0.30, 0.10, 0.15])
    job_role     = rng.choice(['engineer', 'manager', 'analyst', 'sales', 'hr'], n)
    industry     = rng.choice(['tech', 'finance', 'healthcare', 'retail', 'education'], n)
    city_size    = rng.choice(['large', 'medium', 'small'], n, p=[0.50, 0.35, 0.15])
    gender       = rng.choice(['M', 'F', 'Other'], n, p=[0.52, 0.46, 0.02])
    skills_count = rng.integers(1, 15, n)
    overtime_hrs = rng.exponential(5, n).clip(0, 40)
    performance  = rng.choice(['low', 'medium', 'high', 'excellent'], n,
                               p=[0.10, 0.35, 0.40, 0.15])

    edu_map  = {'associate': 0, 'bachelor': 1, 'master': 2, 'phd': 3}
    perf_map = {'low': -10, 'medium': 0, 'high': 10, 'excellent': 25}
    city_map = {'small': 0.9, 'medium': 1.0, 'large': 1.2}
    role_map = {'hr': 55000, 'sales': 60000, 'analyst': 70000, 'engineer': 80000, 'manager': 90000}

    edu_num   = pd.Series(education).map(edu_map)
    perf_num  = pd.Series(performance).map(perf_map)
    city_mult = pd.Series(city_size).map(city_map)
    base_role = pd.Series(job_role).map(role_map)

    noise = rng.normal(0, 5000, n)
    salary = (
        base_role +
        years_exp * 2000 +
        edu_num * 8000 +
        perf_num * 500 +
        skills_count * 800 +
        noise
    ) * city_mult

    salary = salary.clip(25000, 250000)

    # Missing values
    years_exp_m = years_exp.astype(float).copy()
    years_exp_m[rng.choice(n, 90, replace=False)] = np.nan
    overtime_m = overtime_hrs.copy()
    overtime_m[rng.choice(n, 70, replace=False)] = np.nan
    industry_m = np.array(industry, dtype=object)
    industry_m[rng.choice(n, 40, replace=False)] = np.nan

    df = pd.DataFrame({
        'age':              age,
        'years_experience': years_exp_m.round(1),
        'education':        education,
        'job_role':         job_role,
        'industry':         industry_m,
        'city_size':        city_size,
        'gender':           gender,
        'skills_count':     skills_count,
        'overtime_hrs_week': overtime_m.round(1),
        'performance':      performance,
        'salary':           salary.round(0).astype(int),
    })
    path = OUTPUT_DIR / 'employee_salary.csv'
    df.to_csv(path, index=False)
    print(f"  employee_salary.csv:     {df.shape} → {path}")
    return df


def main():
    print("Generating datasets...")
    generate_student_performance()
    generate_customer_churn()
    generate_customer_segmentation()
    generate_end_to_end_dataset()
    print("\nAll datasets generated.")


if __name__ == "__main__":
    main()
