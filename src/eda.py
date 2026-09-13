"""
Exploratory Data Analysis (EDA) figure generation module for Loan Approval ML.
Generates publication-quality charts and saves them to reports/figures/.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import RAW_DATA_FILE

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set consistent, modern styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

PALETTE_STATUS = {"Y": "#2ECC71", "N": "#E74C3C"}
PALETTE_MAIN = "#2980B9"


def generate_all_eda_figures():
    print("[EDA] Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_FILE)

    # -------------------------------------------------------------
    # Figure 1: Target Class Distribution
    # -------------------------------------------------------------
    print("  -> Generating 01_target_distribution.png...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts = df["Loan_Status"].value_counts()
    colors = [PALETTE_STATUS["Y"], PALETTE_STATUS["N"]]
    labels = [f"Approved (Y)\n{counts['Y']} ({counts['Y']/len(df)*100:.1f}%)",
              f"Rejected (N)\n{counts['N']} ({counts['N']/len(df)*100:.1f}%)"]

    # Donut chart
    axes[0].pie(
        counts,
        labels=labels,
        colors=colors,
        startangle=140,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    axes[0].set_title("Target Distribution (Donut)", pad=15)

    # Bar chart
    bars = axes[1].bar(["Approved (Y)", "Rejected (N)"], [counts["Y"], counts["N"]], color=colors, width=0.5, edgecolor="black", linewidth=1)
    for bar in bars:
        h = bar.get_height()
        axes[1].annotate(f"{h}\n({h/len(df)*100:.1f}%)",
                         xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                         ha="center", va="center", color="white", fontweight="bold", fontsize=12)
    axes[1].set_title("Target Frequency Count", pad=15)
    axes[1].set_ylabel("Number of Applicants")
    axes[1].set_ylim(0, 500)

    plt.suptitle("Target Variable Analysis: Loan_Status Class Balance", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_target_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: Credit History vs Loan Approval
    # -------------------------------------------------------------
    print("  -> Generating 02_credit_history_vs_approval.png...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ch_df = df.dropna(subset=["Credit_History", "Loan_Status"]).copy()
    ch_df["Credit_History_Label"] = ch_df["Credit_History"].map({1.0: "Meets Guidelines (1.0)", 0.0: "Defaults / Bad History (0.0)"})

    # Approval rate by credit history
    approval_rates = ch_df.groupby("Credit_History_Label")["Loan_Status"].apply(lambda s: (s == "Y").mean() * 100).reset_index()
    approval_rates.columns = ["Credit History", "Approval Rate (%)"]

    bars = ax.bar(approval_rates["Credit History"], approval_rates["Approval Rate (%)"],
                  color=["#E74C3C", "#27AE60"], width=0.45, edgecolor="black", linewidth=1.2)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h + 2),
                    ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_title("Loan Approval Rate by Applicant Credit History", pad=15)
    ax.set_ylabel("Approval Rate (%)")
    ax.set_ylim(0, 100)
    ax.axhline(68.7, color="gray", linestyle="--", alpha=0.7, label="Overall Dataset Avg (68.7%)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_credit_history_vs_approval.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # Figure 3: Numerical Feature Distributions (Skewness & Outliers)
    # -------------------------------------------------------------
    print("  -> Generating 03_numerical_distributions.png...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ApplicantIncome
    sns.histplot(df["ApplicantIncome"], kde=True, ax=axes[0, 0], color="#2980B9", bins=35)
    axes[0, 0].set_title(f"ApplicantIncome (Skew: {df['ApplicantIncome'].skew():.2f})")
    axes[0, 0].set_xlabel("Applicant Income ($)")

    # CoapplicantIncome
    sns.histplot(df["CoapplicantIncome"], kde=True, ax=axes[0, 1], color="#8E44AD", bins=35)
    axes[0, 1].set_title(f"CoapplicantIncome (Skew: {df['CoapplicantIncome'].skew():.2f})")
    axes[0, 1].set_xlabel("Coapplicant Income ($)")

    # LoanAmount
    sns.histplot(df["LoanAmount"].dropna(), kde=True, ax=axes[1, 0], color="#16A085", bins=35)
    axes[1, 0].set_title(f"LoanAmount in Thousands (Skew: {df['LoanAmount'].skew():.2f})")
    axes[1, 0].set_xlabel("Loan Amount ($'000)")

    # Loan_Amount_Term
    sns.countplot(data=df, x="Loan_Amount_Term", ax=axes[1, 1], color="#D35400")
    axes[1, 1].set_title("Loan Term Distribution (Months)")
    axes[1, 1].set_xlabel("Term (Months)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.suptitle("Distributions of Numerical Features (High Positive Skewness)", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_numerical_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # Figure 4: Total Income & Loan Amount vs Loan Approval
    # -------------------------------------------------------------
    print("  -> Generating 04_income_vs_loan_approval.png...")
    df_copy = df.copy()
    df_copy["Total_Income"] = df_copy["ApplicantIncome"] + df_copy["CoapplicantIncome"]
    df_copy["Log_Total_Income"] = np.log1p(df_copy["Total_Income"])
    df_copy["Log_LoanAmount"] = np.log1p(df_copy["LoanAmount"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=df_copy, x="Loan_Status", y="Log_Total_Income", palette=PALETTE_STATUS, ax=axes[0], width=0.4)
    axes[0].set_title("Log(Total Income) vs Loan Status")
    axes[0].set_xticklabels(["Approved (Y)", "Rejected (N)"])
    axes[0].set_ylabel("Log(Applicant + Coapplicant Income)")

    sns.boxplot(data=df_copy, x="Loan_Status", y="Log_LoanAmount", palette=PALETTE_STATUS, ax=axes[1], width=0.4)
    axes[1].set_title("Log(Loan Amount) vs Loan Status")
    axes[1].set_xticklabels(["Approved (Y)", "Rejected (N)"])
    axes[1].set_ylabel("Log(Loan Amount in $'000)")

    plt.suptitle("Financial Variables vs Loan Decision (Log Scale to Mitigate Outliers)", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_income_vs_loan_approval.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # Figure 5: Categorical Features vs Approval Rate
    # -------------------------------------------------------------
    print("  -> Generating 05_categorical_vs_approval.png...")
    cat_cols = ["Property_Area", "Education", "Married", "Dependents", "Self_Employed", "Gender"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, col in enumerate(cat_cols):
        clean_df = df.dropna(subset=[col, "Loan_Status"]).copy()
        rates = clean_df.groupby(col)["Loan_Status"].apply(lambda s: (s == "Y").mean() * 100).reset_index()
        rates.columns = [col, "Approval_Rate"]

        sns.barplot(data=rates, x=col, y="Approval_Rate", ax=axes[idx], palette="Blues_r", edgecolor="black")
        axes[idx].set_title(f"Approval Rate by {col}")
        axes[idx].set_ylabel("Approval Rate (%)")
        axes[idx].set_ylim(0, 100)
        axes[idx].axhline(68.7, color="gray", linestyle="--", alpha=0.6, label="Dataset Mean")
        for p in axes[idx].patches:
            h = p.get_height()
            axes[idx].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2, h + 2),
                               ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Categorical Factors Impact on Loan Approval Rate (%)", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_categorical_vs_approval.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # Figure 6: Correlation Heatmap
    # -------------------------------------------------------------
    print("  -> Generating 06_correlation_heatmap.png...")
    num_df = df[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]].copy()
    num_df["Total_Income"] = num_df["ApplicantIncome"] + num_df["CoapplicantIncome"]
    num_df["Loan_Status_Binary"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.3,
        vmax=1.0,
        linewidths=1,
        linecolor="white",
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap (Including Loan_Status_Binary)", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[EDA] Successfully generated all 6 figures in: {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_eda_figures()
