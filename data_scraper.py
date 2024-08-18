"""
Simple wrapers for the nfl_data_py package since I hate their names.

TODO: Checks on dataframe to ensure proper values and everything is as expected.
"""
import nfl_data_py as nfl_data_source
import pandas as pd


def get_seasons_data(years: list[int], s_type: str = "ALL") -> pd.DataFrame:
    """
    Calculated receiving market share stats include:
    Column	is short for
    tgt_sh	target share
    ay_sh	air yards share
    yac_sh	yards after catch share
    wopr	weighted opportunity rating
    ry_sh	receiving yards share
    rtd_sh	receiving TDs share
    rfd_sh	receiving 1st Downs share
    rtdfd_sh	receiving TDs + 1st Downs share
    dom	dominator rating
    w8dom	dominator rating, but weighted in favor of receiving yards over TDs
    yptmpa	receiving yards per team pass attempt
    ppr_sh	PPR fantasy points share
    """
    return nfl_data_source.import_seasonal_data(years, s_type)


def get_scoring_lines(years: list[int]) -> pd.DataFrame:
    # TODO: Add Over/Under and Spread Analysis
    return nfl_data_source.import_sc_lines(years)


def get_play_by_play_data(years: list[int]) -> pd.DataFrame:
    return nfl_data_source.import_pbp_data(
        years, downcast=True, cache=False, alt_path=None
    )


def get_play_by_play_columns():
    return nfl_data_source.see_pbp_cols()


def get_weekly_data(years: list[int]) -> pd.DataFrame:
    return nfl_data_source.import_weekly_data(years)


def get_weekly_data_columns():
    return nfl_data_source.see_weekly_cols()


def get_yearly_rosters(years: list[int]) -> pd.DataFrame:
    return nfl_data_source.import_seasonal_rosters(years)


def get_win_totals(years: list[int]) -> pd.DataFrame:
    return nfl_data_source.import_win_totals(years)


def get_season_schedule(years: list[int]) -> pd.DataFrame:
    return nfl_data_source.import_schedules(years)


if __name__ == "__main__":
    years = [2020]

    # Get the data
    seasons = get_seasons_data(years)
    scoring_lines = get_scoring_lines(years)
    pbp_data = get_play_by_play_data(years)
    weekly_data = get_weekly_data(years)
    rosters = get_yearly_rosters(years)
    win_totals = get_win_totals(years)
    schedules = get_season_schedule(years)

    # Print the data
    print(seasons.head())
    print(scoring_lines.head())
    print(pbp_data.head())
    print(weekly_data.head())
    print(rosters.head())
    print(win_totals.head())
    print(schedules.head())

    # Save each to file
    seasons.to_csv("seasons.csv")
    scoring_lines.to_csv("scoring_lines.csv")
    pbp_data.to_csv("pbp_data.csv")
    weekly_data.to_csv("weekly_data.csv")
    rosters.to_csv("rosters.csv")
    win_totals.to_csv("win_totals.csv")
    schedules.to_csv("schedules.csv")
