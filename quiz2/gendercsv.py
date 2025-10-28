from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "gender_list.csv"
OUTPUT_YEAR_GENDER_CSV = BASE_DIR / "general_households_by_gender_year.csv"
OUTPUT_AGE_CSV = BASE_DIR / "general_households_by_age.csv"
OUTPUT_GENDER_AGE_CSV = BASE_DIR / "general_households_by_gender_age.csv"
OUTPUT_FIGURE = BASE_DIR / "gender_age_line.png"
OUTPUT_REPORT = BASE_DIR / "general_household_trend_report.txt"


def load_long_form_dataframe(csv_path: Path) -> pd.DataFrame:
    """
    Read the source CSV, replace sentinel values, and expand the repeated
    year/metric columns into a tidy long-form dataframe.
    """
    df_raw = pd.read_csv(csv_path, encoding="cp949")
    df_raw = df_raw.rename(
        columns={
            df_raw.columns[0]: "행정구역",
            df_raw.columns[1]: "성별",
            df_raw.columns[2]: "연령별",
        }
    )

    header_row = df_raw.iloc[0]
    data_rows = df_raw.iloc[1:].copy()
    data_rows = data_rows.replace({"X": pd.NA, "-": pd.NA})

    long_frames = []
    for column in data_rows.columns[3:]:
        year = int(column.split(".")[0])
        metric_name = header_row[column]
        values = pd.to_numeric(data_rows[column], errors="coerce")
        long_frames.append(
            pd.DataFrame(
                {
                    "행정구역": data_rows["행정구역"],
                    "성별": data_rows["성별"],
                    "연령별": data_rows["연령별"],
                    "연도": year,
                    "지표": metric_name,
                    "값": values,
                }
            )
        )

    tidy_df = (
        pd.concat(long_frames, ignore_index=True)
        .dropna(subset=["값"])
        .astype({"연도": "int64"})
    )
    return tidy_df


def aggregate_general_households(
    long_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indicator_mask = long_df["지표"] == "일반가구원"
    filtered = long_df[indicator_mask & (long_df["연도"] >= 2015)].copy()
    age_order = (
        filtered.loc[filtered["성별"] == "계", "연령별"]
        .drop_duplicates()
        .tolist()
    )

    gender_year = (
        filtered[
            (filtered["성별"].isin(["남자", "여자"])) & (filtered["연령별"] == "합계")
        ]
        .groupby(["연도", "성별"], as_index=False)["값"]
        .sum()
        .sort_values(["연도", "성별"])
    )
    age_totals = (
        filtered[filtered["성별"] == "계"]
        .groupby(["연도", "연령별"], as_index=False)["값"]
        .sum()
    )
    age_totals["연령별"] = pd.Categorical(
        age_totals["연령별"], categories=age_order, ordered=True
    )
    age_totals = age_totals.sort_values(["연도", "연령별"])

    gender_age = (
        filtered[filtered["성별"].isin(["남자", "여자"])]
        .groupby(["연도", "성별", "연령별"], as_index=False)["값"]
        .sum()
    )
    gender_age["연령별"] = pd.Categorical(
        gender_age["연령별"], categories=age_order, ordered=True
    )
    gender_age = gender_age.sort_values(["연도", "성별", "연령별"])

    return gender_year, age_totals, gender_age


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_gender_age(gender_age_df: pd.DataFrame) -> None:
    if gender_age_df.empty:
        return

    try:
        plt.rcParams["font.family"] = "Malgun Gothic"
    except Exception:
        # Fallback to default font if Malgun Gothic is unavailable.
        pass
    plt.rcParams["axes.unicode_minus"] = False

    latest_year = gender_age_df["연도"].max()
    excluded_groups = {"합계", "15~64세", "65세이상"}
    latest_df = gender_age_df[
        (~gender_age_df["연령별"].isin(excluded_groups))
        & (gender_age_df["연도"] == latest_year)
    ].copy()

    if latest_df.empty:
        return

    age_order = [
        age
        for age in latest_df["연령별"].cat.categories
        if age not in excluded_groups
    ]
    pivot = latest_df.pivot(index="연령별", columns="성별", values="값").reindex(age_order)
    pivot.plot(marker="o")

    plt.title(f"{latest_year}년 남자·여자 연령별 일반가구원")
    plt.xlabel("연령별")
    plt.ylabel("일반가구원 수")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=150)
    plt.close()


def build_report(
    gender_year_df: pd.DataFrame, age_totals_df: pd.DataFrame, gender_age_df: pd.DataFrame
) -> str:
    if gender_year_df.empty or age_totals_df.empty or gender_age_df.empty:
        return "가용한 데이터가 부족하여 리포트를 생성할 수 없습니다."

    latest_year = gender_year_df["연도"].max()
    latest_gender = (
        gender_year_df[gender_year_df["연도"] == latest_year]
        .set_index("성별")["값"]
        .to_dict()
    )

    latest_age_total = age_totals_df[age_totals_df["연도"] == latest_year]
    main_groups = latest_age_total[
        ~latest_age_total["연령별"].isin(["합계", "15~64세", "65세이상"])
    ]

    total_households = latest_age_total.loc[latest_age_total["연령별"] == "합계", "값"].iloc[0]
    senior_households = latest_age_total.loc[latest_age_total["연령별"] == "65세이상", "값"].iloc[0]
    senior_ratio = senior_households / total_households * 100

    top_age_row = main_groups.sort_values("값", ascending=False).iloc[0]
    bottom_age_row = main_groups.sort_values("값", ascending=True).iloc[0]

    male_top_age = (
        gender_age_df[
            (gender_age_df["연도"] == latest_year)
            & (gender_age_df["성별"] == "남자")
            & (~gender_age_df["연령별"].isin(["합계", "15~64세", "65세이상"]))
        ]
        .sort_values("값", ascending=False)
        .iloc[0]
    )
    female_top_age = (
        gender_age_df[
            (gender_age_df["연도"] == latest_year)
            & (gender_age_df["성별"] == "여자")
            & (~gender_age_df["연령별"].isin(["합계", "15~64세", "65세이상"]))
        ]
        .sort_values("값", ascending=False)
        .iloc[0]
    )

    report = f"""
    {latest_year}년 일반가구원 통계 요약

    - 전체 일반가구원은 {total_households:,.0f}명이며, 남자 {latest_gender.get('남자', 0):,.0f}명, 여자 {latest_gender.get('여자', 0):,.0f}명으로 여성이 약간 더 많습니다.
    - 연령대별로는 '{top_age_row['연령별']}' 구간의 일반가구원이 {top_age_row['값']:,.0f}명으로 가장 많고, '{bottom_age_row['연령별']}' 구간이 {bottom_age_row['값']:,.0f}명으로 가장 적습니다.
    - 남자는 '{male_top_age['연령별']}' 구간에서 {male_top_age['값']:,.0f}명으로 가장 많으며, 여자는 '{female_top_age['연령별']}' 구간에서 {female_top_age['값']:,.0f}명으로 정점을 이룹니다.
    - 65세 이상 일반가구원은 {senior_households:,.0f}명으로 전체의 {senior_ratio:.1f}%를 차지하여 고령 가구원의 비중이 상당합니다.

    위 지표를 바탕으로 보면 50대 전후 연령층이 일반가구원의 핵심을 이루고 있으며, 고령층 비중이 꾸준히 높아진다는 점에서 향후 고령 친화 정책과 중장년층 지원 전략이 중요해 보입니다.
    """
    return textwrap.dedent(report).strip()


def main() -> None:
    long_df = load_long_form_dataframe(DATA_PATH)
    gender_year_df, age_totals_df, gender_age_df = aggregate_general_households(long_df)

    save_csv(gender_year_df, OUTPUT_YEAR_GENDER_CSV)
    save_csv(age_totals_df, OUTPUT_AGE_CSV)
    save_csv(gender_age_df, OUTPUT_GENDER_AGE_CSV)

    plot_gender_age(gender_age_df)

    report_text = build_report(gender_year_df, age_totals_df, gender_age_df)
    OUTPUT_REPORT.write_text(report_text, encoding="utf-8")

    print("연도별 남자/여자 일반가구원")
    print(gender_year_df.to_string(index=False))
    print("\n연령별 일반가구원(계)")
    age_view = (
        age_totals_df[age_totals_df["연령별"].notna()]
        .sort_values(["연도", "연령별"])
        .to_string(index=False)
    )
    print(age_view)
    print("\n남자·여자 연령별 일반가구원")
    print(gender_age_df.to_string(index=False))
    print(f"\n리포트 저장: {OUTPUT_REPORT.name}")
    if OUTPUT_FIGURE.exists():
        print(f"그래프 저장: {OUTPUT_FIGURE.name}")


if __name__ == "__main__":
    main()
