"""
CAPSTONE MINI PROJECT
Analysis of Air Quality and Its Association with Respiratory Hospital Admissions

Required libraries:
pandas, numpy, matplotlib, seaborn, scipy, statsmodels

Keep both input CSV files in the same folder as this Python file.\nNo output files are created; results are printed and graphs are displayed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Patch

from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

sns.set_theme(style="whitegrid", context="notebook")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_1 = os.path.join(BASE_DIR, "air_quality_health_dataset.csv")
DATASET_2 = os.path.join(BASE_DIR, "air_quality_health_dataset_2.csv")


def load_datasets():
    """Load both CSV datasets."""
    if not os.path.exists(DATASET_1):
        raise FileNotFoundError("Dataset 1 not found. Keep it in the same folder as this code.")
    if not os.path.exists(DATASET_2):
        raise FileNotFoundError("Dataset 2 not found. Keep it in the same folder as this code.")

    data1 = pd.read_csv(DATASET_1)
    data2 = pd.read_csv(DATASET_2)

    print("\nDATASETS LOADED SUCCESSFULLY")
    print("Dataset 1 shape:", data1.shape)
    print("Dataset 2 shape:", data2.shape)

    return data1, data2


def fill_missing_values(data):
    """Fill missing numeric values with median and categorical values with mode."""
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = data[column].fillna(data[column].median())
        else:
            mode_values = data[column].mode()
            if not mode_values.empty:
                data[column] = data[column].fillna(mode_values.iloc[0])
            else:
                data[column] = data[column].fillna("Unknown")
    return data


def clean_dataset_1(data):
    """Clean the original city-wise air quality and hospital admissions dataset."""
    data = data.copy()
    data.columns = data.columns.str.strip().str.lower()

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    invalid_dates = data["date"].isna().sum()
    data = data.dropna(subset=["date"])

    data["city"] = data["city"].astype("string").str.strip().str.title()
    data["population_density"] = (
        data["population_density"].astype("string").str.strip().str.title()
    )

    numeric_columns = [
        "aqi", "pm2_5", "pm10", "no2", "o3", "temperature",
        "humidity", "hospital_admissions", "hospital_capacity"
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = fill_missing_values(data)
    data = data.drop_duplicates().reset_index(drop=True)

    # Keep the original dates, but flag dates outside the realistic study window.
    data["date_within_2020_2021"] = data["date"].between(
        pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")
    )

    print("\nDATASET 1 CLEANING")
    print("Invalid dates removed:", invalid_dates)
    print("Rows after cleaning:", len(data))
    print("Missing values after cleaning:", int(data.isna().sum().sum()))
    print("Exact duplicate rows remaining:", int(data.duplicated().sum()))
    print("Dates outside 2020-2021 retained but not used for time-series analysis:",
          int((~data["date_within_2020_2021"]).sum()))

    return data


def clean_dataset_2(data):
    """Clean the regional air quality and respiratory admissions dataset."""
    data = data.copy()

    rename_columns = {
        "AQI": "aqi",
        "PM2.5": "pm2_5",
        "PM10": "pm10",
        "NO2": "no2",
        "SO2": "so2",
        "CO": "co",
        "O3": "o3"
    }
    data = data.rename(columns=rename_columns)
    data.columns = data.columns.str.strip().str.lower()

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    invalid_dates = data["date"].isna().sum()
    data = data.dropna(subset=["date"])

    data["region"] = data["region"].astype("string").str.strip().str.title()

    for column in data.columns:
        if column not in ["date", "region"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data = fill_missing_values(data)
    data = data.drop_duplicates().reset_index(drop=True)

    # Validate binary indicator columns.
    for column in ["school_closures", "lockdown_status"]:
        if column in data.columns:
            data.loc[~data[column].isin([0, 1]), column] = np.nan
            data[column] = data[column].fillna(data[column].mode().iloc[0])

    print("\nDATASET 2 CLEANING")
    print("Invalid dates removed:", invalid_dates)
    print("Rows after cleaning:", len(data))
    print("Missing values after cleaning:", int(data.isna().sum().sum()))
    print("Exact duplicate rows remaining:", int(data.duplicated().sum()))

    repeated_keys = int(data.duplicated(subset=["date", "region"]).sum())
    print("Repeated date-region combinations:", repeated_keys)

    return data


def identify_outliers(data, dataset_name):
    """Identify potential outliers using the IQR method without deleting them."""
    numeric_data = data.select_dtypes(include=np.number)
    outlier_summary = []

    for column in numeric_data.columns:
        q1 = numeric_data[column].quantile(0.25)
        q3 = numeric_data[column].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        count = int(
            ((numeric_data[column] < lower_limit) |
             (numeric_data[column] > upper_limit)).sum()
        )
        outlier_summary.append({
            "dataset": dataset_name,
            "column": column,
            "outlier_count": count,
            "lower_limit": lower_limit,
            "upper_limit": upper_limit
        })

    return pd.DataFrame(outlier_summary)


def create_dataset_2_time_series(data):
    """
    Average repeated observations within each date-region.
    Then calculate the daily mean across regions to avoid double-counting.
    """
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

    region_daily = data.groupby(["date", "region"], as_index=False)[
        numeric_columns
    ].mean()

    daily = region_daily.groupby("date")[numeric_columns].mean().sort_index()
    daily = daily.asfreq("D")

    # Fill any missing dates created by the daily frequency conversion.
    daily = daily.interpolate(method="time").ffill().bfill()


    print("\nTIME-SERIES PREPARATION")
    print("Unique region-date records:", len(region_daily))
    print("Daily observations:", len(daily))
    print("Daily time-series range:", daily.index.min().date(), "to",
          daily.index.max().date())

    return region_daily, daily


def perform_statistical_analysis(data1, data2, daily_data):
    """Perform correlation, hypothesis testing and time-series analysis."""
    results = []

    # Dataset 1: Pearson and Spearman correlation.
    pearson_1 = stats.pearsonr(data1["pm2_5"], data1["hospital_admissions"])
    spearman_1 = stats.spearmanr(data1["pm2_5"], data1["hospital_admissions"])

    results.extend([
        {"analysis": "Dataset 1: PM2.5 vs Hospital Admissions - Pearson correlation",
         "statistic": pearson_1.statistic, "p_value": pearson_1.pvalue},
        {"analysis": "Dataset 1: PM2.5 vs Hospital Admissions - Spearman correlation",
         "statistic": spearman_1.statistic, "p_value": spearman_1.pvalue}
    ])

    # Dataset 2: Correlation with respiratory admissions.
    pearson_2 = stats.pearsonr(data2["pm2_5"], data2["respiratory_admissions"])
    spearman_2 = stats.spearmanr(data2["pm2_5"], data2["respiratory_admissions"])

    results.extend([
        {"analysis": "Dataset 2: PM2.5 vs Respiratory Admissions - Pearson correlation",
         "statistic": pearson_2.statistic, "p_value": pearson_2.pvalue},
        {"analysis": "Dataset 2: PM2.5 vs Respiratory Admissions - Spearman correlation",
         "statistic": spearman_2.statistic, "p_value": spearman_2.pvalue}
    ])

    # Independent-sample t-test: lockdown vs no lockdown.
    lockdown_group = data2.loc[
        data2["lockdown_status"] == 1, "respiratory_admissions"
    ]
    no_lockdown_group = data2.loc[
        data2["lockdown_status"] == 0, "respiratory_admissions"
    ]

    if len(lockdown_group) > 1 and len(no_lockdown_group) > 1:
        t_test = stats.ttest_ind(
            lockdown_group, no_lockdown_group, equal_var=False
        )
        results.append({
            "analysis": "Dataset 2: Respiratory Admissions - Lockdown vs No Lockdown (Welch t-test)",
            "statistic": t_test.statistic,
            "p_value": t_test.pvalue
        })

    # One-way ANOVA: compare respiratory admissions across regions.
    region_groups = [
        group["respiratory_admissions"].values
        for _, group in data2.groupby("region")
        if len(group) > 1
    ]
    if len(region_groups) >= 2:
        anova_result = stats.f_oneway(*region_groups)
        results.append({
            "analysis": "Dataset 2: Respiratory Admissions across Regions (One-way ANOVA)",
            "statistic": anova_result.statistic,
            "p_value": anova_result.pvalue
        })

    # Augmented Dickey-Fuller stationarity test.
    series = daily_data["respiratory_admissions"].dropna()
    adf_result = adfuller(series)

    results.append({
        "analysis": "Dataset 2: Respiratory Admissions - Augmented Dickey-Fuller test",
        "statistic": adf_result[0],
        "p_value": adf_result[1]
    })

    results_df = pd.DataFrame(results)
    results_df["interpretation_at_5_percent"] = np.where(
        results_df["p_value"] < 0.05,
        "Statistically significant (p < 0.05)",
        "Not statistically significant (p >= 0.05)"
    )


    print("\nSTATISTICAL ANALYSIS RESULTS")
    print(results_df.to_string(index=False))

    return results_df


def create_visualizations(data1, data2, daily_data):
    """Create clear, labelled visualizations using Matplotlib and Seaborn."""
    # Graph 1: Dataset 1 correlation heatmap.
    columns_1 = [
        "aqi", "pm2_5", "pm10", "no2", "o3",
        "temperature", "humidity", "hospital_admissions"
    ]
    plt.figure(figsize=(11, 8))
    correlation_1 = data1[columns_1].corr()
    sns.heatmap(
        correlation_1, annot=True, fmt=".5f", cmap="coolwarm",
        norm=SymLogNorm(linthresh=0.005, linscale=1, vmin=-0.4, vmax=0.4, base=10),
        linewidths=0.5,
        cbar_kws={"label": "Correlation (enhanced nonlinear colour scale; Dataset 1 range)"}
    )
    plt.title("Dataset 1: Correlation Between Air Quality and Health Variables",
              fontsize=14, pad=15)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    # Graph 2: Dataset 1 PM2.5 and general hospital admissions.
    sample_1 = data1.sample(
        n=min(5000, len(data1)), random_state=42
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=sample_1, x="pm2_5", y="hospital_admissions",
        alpha=0.45, s=28, label="Observations", ax=ax
    )
    sns.regplot(
        data=sample_1, x="pm2_5", y="hospital_admissions",
        scatter=False, color="red", line_kws={"linewidth": 2},
        label="Linear regression trend", ax=ax
    )

    # Show average admissions in PM2.5 ranges to make the overall pattern easier to see.
    binned_1 = sample_1.copy()
    binned_1["PM2.5 range"] = pd.qcut(
        binned_1["pm2_5"], q=20, duplicates="drop"
    )
    mean_trend_1 = binned_1.groupby(
        "PM2.5 range", observed=True
    ).agg(
        mean_pm25=("pm2_5", "mean"),
        mean_admissions=("hospital_admissions", "mean")
    ).reset_index()
    ax.plot(
        mean_trend_1["mean_pm25"], mean_trend_1["mean_admissions"],
        color="green", marker="o", linewidth=2.5, markersize=5,
        label="Average admissions by PM2.5 range"
    )
    correlation_value_1 = sample_1["pm2_5"].corr(
        sample_1["hospital_admissions"]
    )
    ax.text(
        0.02, 0.98, f"Pearson correlation (r) = {correlation_value_1:.3f}",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8}
    )
    ax.set_title("Dataset 1: PM2.5 vs General Hospital Admissions", pad=12)
    ax.set_xlabel("PM2.5 concentration")
    ax.set_ylabel("Hospital admissions")
    ax.legend(title="Legend", loc="upper right")
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)

    # Graph 3: Dataset 2 correlation heatmap.
    columns_2 = [
        "aqi", "pm2_5", "pm10", "no2", "so2", "co", "o3",
        "respiratory_admissions", "hospital_visits", "emergency_visits",
        "mobility_index", "mask_usage_rate"
    ]
    plt.figure(figsize=(13, 9))
    correlation_2 = data2[columns_2].corr()
    sns.heatmap(
        correlation_2, annot=True, fmt=".5f", cmap="vlag",
        norm=SymLogNorm(linthresh=0.05, linscale=1, vmin=-1, vmax=1, base=10),
        linewidths=0.4, annot_kws={"size": 8},
        cbar_kws={"label": "Correlation (enhanced nonlinear colour scale)"}
    )
    plt.title("Dataset 2: Correlation Between Environmental and Health Variables",
              fontsize=14, pad=15)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    # Graph 4: Dataset 2 PM2.5 and respiratory admissions.
    sample_2 = data2.sample(
        n=min(2000, len(data2)), random_state=42
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=sample_2, x="pm2_5", y="respiratory_admissions",
        hue="region", alpha=0.65, s=35, ax=ax
    )
    sns.regplot(
        data=sample_2, x="pm2_5", y="respiratory_admissions",
        scatter=False, color="black", line_kws={"linewidth": 2},
        label="Overall linear regression trend", ax=ax
    )

    # Binned means summarize the data without changing the underlying correlation.
    binned_2 = sample_2.copy()
    binned_2["PM2.5 range"] = pd.qcut(
        binned_2["pm2_5"], q=20, duplicates="drop"
    )
    mean_trend_2 = binned_2.groupby(
        "PM2.5 range", observed=True
    ).agg(
        mean_pm25=("pm2_5", "mean"),
        mean_admissions=("respiratory_admissions", "mean")
    ).reset_index()
    ax.plot(
        mean_trend_2["mean_pm25"], mean_trend_2["mean_admissions"],
        color="yellow", marker="D", markeredgecolor="black",
        linewidth=2.5, markersize=5,
        label="Average admissions by PM2.5 range"
    )
    correlation_value_2 = sample_2["pm2_5"].corr(
        sample_2["respiratory_admissions"]
    )
    ax.text(
        0.02, 0.98, f"Pearson correlation (r) = {correlation_value_2:.3f}",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8}
    )
    ax.set_title("Dataset 2: PM2.5 vs Respiratory Admissions", pad=12)
    ax.set_xlabel("PM2.5 concentration")
    ax.set_ylabel("Respiratory admissions")
    ax.legend(title="Region / Trend", loc="upper right")
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)

    # Graph 5: Distribution of PM2.5 concentrations in Dataset 1.
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=data1, x="pm2_5", bins=30, kde=True,
        color="steelblue", edgecolor="white", label="PM2.5 observations", ax=ax
    )
    ax.set_title("Dataset 1: Distribution of PM2.5 Concentrations", pad=12)
    ax.legend(loc="upper right", title="Legend")
    ax.set_xlabel("PM2.5 concentration")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)

    # Graph 6: Distribution of general hospital admissions in Dataset 1.
    fig, ax = plt.subplots(figsize=(10, 6))
    admissions_values = data1["hospital_admissions"].dropna()
    admission_bins = np.arange(
        np.floor(admissions_values.min()) - 0.5,
        np.ceil(admissions_values.max()) + 1.5,
        1
    )
    sns.histplot(
        x=admissions_values, bins=admission_bins, kde=False,
        color="darkorange", edgecolor="white", linewidth=0.8,
        label="Hospital admission records", ax=ax
    )
    ax.set_title("Dataset 1: Distribution of General Hospital Admissions", pad=12)
    ax.legend(loc="upper right", title="Legend")
    ax.set_xlabel("Number of hospital admissions")
    ax.set_ylabel("Number of records")
    ax.set_xticks(np.arange(
        int(np.floor(admissions_values.min())),
        int(np.ceil(admissions_values.max())) + 1,
        1
    ))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)

    # Graph 7: Compare PM2.5 levels across population-density categories.
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=data1, x="population_density", y="pm2_5",
        color="lightblue", width=0.55, ax=ax
    )
    ax.legend(
        handles=[Patch(facecolor="lightblue", label="PM2.5 concentration distribution")],
        loc="upper right", title="Legend"
    )
    ax.set_title("Dataset 1: PM2.5 Levels Across Population-Density Categories",
                 pad=12)
    ax.set_xlabel("Population-density category")
    ax.set_ylabel("PM2.5 concentration")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)

    # Graph 8: Compare average respiratory admissions across regions.
    region_admissions = (
        data2.groupby("region", as_index=False)["respiratory_admissions"]
        .mean()
        .sort_values("respiratory_admissions", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=region_admissions, x="region", y="respiratory_admissions",
        hue="region", legend=False, ax=ax
    )
    ax.legend(
        handles=[Patch(facecolor="grey", label="Average respiratory admissions")],
        loc="upper right", title="Legend"
    )
    ax.set_title("Dataset 2: Average Respiratory Admissions by Region", pad=12)
    ax.set_xlabel("Region")
    ax.set_ylabel("Average respiratory admissions")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)

    # Graph 9: Monthly trend in respiratory admissions in Dataset 2.
    monthly_admissions = (
        data2.set_index("date")["respiratory_admissions"]
        .resample("MS").mean()
        .dropna()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(
        data=monthly_admissions, x="date", y="respiratory_admissions",
        marker="o", linewidth=2, label="Monthly average", ax=ax
    )
    ax.set_title("Dataset 2: Monthly Trend in Respiratory Admissions", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Average respiratory admissions")
    ax.legend(title="Trend", loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)

    # Graph 10: Compare respiratory admissions during lockdown and non-lockdown.
    lockdown_labels = data2.copy()
    lockdown_labels["lockdown_period"] = lockdown_labels["lockdown_status"].map(
        {0: "No lockdown", 1: "Lockdown"}
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        data=lockdown_labels, x="lockdown_period",
        y="respiratory_admissions",
        color="lightgreen", width=0.5, ax=ax
    )
    ax.legend(
        handles=[Patch(facecolor="lightgreen", label="Respiratory admissions distribution")],
        loc="upper right", title="Legend"
    )
    ax.set_title("Dataset 2: Respiratory Admissions During Lockdown vs No Lockdown",
                 pad=12)
    ax.set_xlabel("Period")
    ax.set_ylabel("Respiratory admissions")
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)

    # Graph 11: Explore the relationship between green cover and respiratory admissions.
    sample_green = data2.sample(n=min(1500, len(data2)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(
        data=sample_green, x="green_cover_percentage",
        y="respiratory_admissions", scatter_kws={"alpha": 0.45, "s": 28},
        line_kws={"color": "red", "linewidth": 2}, ax=ax
    )
    ax.set_title("Dataset 2: Green Cover vs Respiratory Admissions", pad=12)
    ax.set_xlabel("Green cover percentage")
    ax.set_ylabel("Respiratory admissions")
    ax.legend(["Observations", "Linear regression trend"], loc="upper right")
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)


def perform_forecasting(daily_data, forecast_days=30):
    """Evaluate an ARIMA model and forecast daily respiratory admissions."""
    series = daily_data["respiratory_admissions"].dropna().astype(float)

    if len(series) < 100:
        print("\nNot enough daily observations for reliable ARIMA forecasting.")
        return None

    test_size = min(forecast_days, max(1, len(series) // 5))
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    try:
        evaluation_model = ARIMA(train, order=(1, 1, 1))
        evaluation_fit = evaluation_model.fit()
        test_forecast = evaluation_fit.forecast(steps=test_size)

        errors = np.asarray(test) - np.asarray(test_forecast)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        final_model = ARIMA(series, order=(1, 1, 1))
        final_fit = final_model.fit()
        future_forecast = final_fit.forecast(steps=forecast_days)

        future_dates = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D"
        )

        forecast_df = pd.DataFrame({
            "date": future_dates,
            "forecast_respiratory_admissions": future_forecast.values
        })


        metrics = pd.DataFrame([{
            "model": "ARIMA(1,1,1)",
            "test_observations": test_size,
            "MAE": mae,
            "RMSE": rmse
        }])

        # Graph 5: Observed data and future forecast.
        fig, ax = plt.subplots(figsize=(12, 6))
        recent_actual = series.iloc[-180:]
        ax.plot(
            recent_actual.index, recent_actual.values,
            label="Observed daily mean", linewidth=1.8
        )
        ax.plot(
            future_dates, future_forecast.values,
            label="ARIMA forecast (next 30 days)",
            linestyle="--", marker="o", markersize=3
        )
        ax.axvline(
            series.index[-1], linestyle=":", color="grey",
            label="Forecast begins"
        )
        ax.set_title("Forecast of Daily Mean Respiratory Admissions using ARIMA",
                     fontsize=14, pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Mean respiratory admissions")
        ax.legend(title="Series", loc="upper right")
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show(block=True)
        plt.close(fig)
    
        print("\nARIMA FORECASTING")
        print("Model: ARIMA(1,1,1)")
        print("Test MAE:", round(mae, 3))
        print("Test RMSE:", round(rmse, 3))
        print("\nNext 30 days forecast:")
        print(forecast_df.head(10).to_string(index=False))

        return forecast_df

    except Exception as error:
        print("\nARIMA forecasting could not be completed:", error)
        return None


def main():
    """Run the complete capstone mini project."""
    data1_raw, data2_raw = load_datasets()

    data1 = clean_dataset_1(data1_raw)
    data2 = clean_dataset_2(data2_raw)

    outliers_1 = identify_outliers(data1, "Dataset 1")
    outliers_2 = identify_outliers(data2, "Dataset 2")
    outlier_report = pd.concat(
        [outliers_1, outliers_2], ignore_index=True
    )

    print("\nCLEANED DATASET 1 PREVIEW")
    print(data1.head().to_string(index=False))
    print("\nCLEANED DATASET 2 PREVIEW")
    print(data2.head().to_string(index=False))
    print("\nOUTLIER DETECTION REPORT (IQR METHOD)")
    print(outlier_report.to_string(index=False))


    region_daily, daily_data = create_dataset_2_time_series(data2)

    perform_statistical_analysis(data1, data2, daily_data)
    create_visualizations(data1, data2, daily_data)
    perform_forecasting(daily_data, forecast_days=30)

    print("\nPROJECT EXECUTION COMPLETED.")


if __name__ == "__main__":
    main()
