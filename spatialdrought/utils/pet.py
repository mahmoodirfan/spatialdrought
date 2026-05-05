"""
Potential Evapotranspiration (PET) estimation methods.

Hargreaves & Samani (1985) — requires only Tmin, Tmax, latitude.
This is the standard choice when only temperature data is available,
which is the common case for remote sensing workflows using MODIS LST.

Units throughout: mm/day unless stated.
"""

import numpy as np


def extraterrestrial_radiation(doy: np.ndarray, lat_rad: np.ndarray) -> np.ndarray:
    """
    Compute extraterrestrial radiation (Ra) in MJ/m²/day.

    FAO-56 equations 21-25.

    Parameters
    ----------
    doy : np.ndarray
        Day of year (1-365). Shape (time,) or scalar.
    lat_rad : np.ndarray
        Latitude in radians. Shape (rows, cols) or scalar.

    Returns
    -------
    Ra : np.ndarray
        Extraterrestrial radiation in MJ/m²/day.
        Shape (time, rows, cols) if both inputs are arrays.
    """
    # Solar declination (radians)
    decl = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)

    # Inverse relative distance Earth-Sun
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)

    # Expand dims for broadcasting (time, rows, cols)
    if np.ndim(doy) == 1 and np.ndim(lat_rad) == 2:
        decl = decl[:, np.newaxis, np.newaxis]
        dr   = dr[:, np.newaxis, np.newaxis]
        lat  = lat_rad[np.newaxis, :, :]
    else:
        lat = lat_rad

    # Sunset hour angle
    ws = np.arccos(-np.tan(lat) * np.tan(decl))

    # Ra (MJ/m²/day)
    Gsc = 0.0820  # solar constant MJ/m²/min
    Ra = (24.0 * 60.0 / np.pi) * Gsc * dr * (
        ws * np.sin(lat) * np.sin(decl) +
        np.cos(lat) * np.cos(decl) * np.sin(ws)
    )
    return Ra


def hargreaves_pet(
    tmin: np.ndarray,
    tmax: np.ndarray,
    doy: np.ndarray,
    lat_deg: np.ndarray,
) -> np.ndarray:
    """
    Hargreaves & Samani (1985) PET estimate.

    PET = 0.0023 * Ra * (Tmean + 17.8) * (Tmax - Tmin)^0.5

    Parameters
    ----------
    tmin : np.ndarray
        Minimum temperature in Celsius. Shape (time, rows, cols).
    tmax : np.ndarray
        Maximum temperature in Celsius. Shape (time, rows, cols).
    doy : np.ndarray
        Day of year for each time step. Shape (time,).
    lat_deg : np.ndarray
        Latitude in degrees. Shape (rows, cols).

    Returns
    -------
    pet : np.ndarray
        PET in mm/day. Shape (time, rows, cols).
    """
    lat_rad = np.deg2rad(lat_deg)
    Ra = extraterrestrial_radiation(doy, lat_rad)  # (time, rows, cols)

    tmean = (tmin + tmax) / 2.0
    td = np.clip(tmax - tmin, 0.0, None)  # temperature range, non-negative

    pet = 0.0023 * Ra * (tmean + 17.8) * np.sqrt(td)
    return np.clip(pet, 0.0, None)


def monthly_pet_from_daily(pet_daily: np.ndarray, days_per_month: np.ndarray) -> np.ndarray:
    """
    Convert daily PET (mm/day) to monthly totals (mm/month).

    Parameters
    ----------
    pet_daily : np.ndarray
        Daily PET. Shape (time_daily, rows, cols).
    days_per_month : np.ndarray
        Number of days in each month. Shape (n_months,).
        Used to aggregate daily → monthly.

    Returns
    -------
    pet_monthly : np.ndarray
        Monthly PET totals. Shape (n_months, rows, cols).
    """
    n_months = len(days_per_month)
    spatial = pet_daily.shape[1:]
    pet_monthly = np.full((n_months, *spatial), np.nan)

    idx = 0
    for m, ndays in enumerate(days_per_month):
        pet_monthly[m] = np.nansum(pet_daily[idx:idx + ndays], axis=0)
        idx += ndays

    return pet_monthly
