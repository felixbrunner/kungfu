"""
This module provides functions to carry out factor analysis.

Classes:
    FactorModel

"""

import warnings

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt


class FactorModel:
    """Financial linear factor model.

    Attributes:
        factor_data: The factor data index by DatetimeIndex.
        is_fitted: Indicates if model is fitted to asset returns data.

    """

    def __init__(self, factor_data: pd.DataFrame):
        """Store factor data to be used in the model."""
        self.factor_data = factor_data
        self.is_fitted = False

    @property
    def factor_data(self) -> pd.DataFrame:
        """The factor returns data used in the model as a dataframe."""
        return self._factor_data

    @factor_data.setter
    def factor_data(self, factor_data: pd.DataFrame):

        # check if data is indexed by datetime
        if not type(factor_data.index) == pd.DatetimeIndex:
            raise ValueError(
                "factor_data needs to have a DatetimeIndex, index has type '{}'".format(
                    type(factor_data.index)
                )
            )

        # transform to dataframe if series
        if isinstance(factor_data, pd.Series):
            factor_data = factor_data.to_frame()

        # set attribute
        self._factor_data = factor_data

    @property
    def k_factors(self) -> int:
        """The number of factors in the factor model."""
        return self.factor_data.shape[1]

    @staticmethod
    def _preprocess_returns_data(returns_data: pd.DataFrame) -> pd.DataFrame:
        """Set up returns timeseries data as a DataFrame in wide format.

        Args:
            returns_data: The asset returns data in any DataFrame format.

        Returns:
            returns_data: The processed returns data in a T by N DataFrame.

        """

        # unstack multiindex
        if type(returns_data.index) == pd.MultiIndex:
            if len(returns_data.columns) != 1:
                raise ValueError("too many columns, supply only return data")
            returns_data = returns_data.unstack()

        # check if returns data is indexed by datetime
        if not type(returns_data.index) == pd.DatetimeIndex:
            raise ValueError(
                "returns_data needs to have a DatetimeIndex, index has type '{}'".format(
                    type(returns_data.index)
                )
            )

        # transform to dataframe if series
        if isinstance(returns_data, pd.Series):
            returns_data = returns_data.to_frame()

        return returns_data

    def _preprocess_factor_data(
        self, returns_data: pd.DataFrame, add_constant: bool
    ) -> pd.DataFrame:
        """Set up factor data to match asset returns data index.

        Args:
            returns_data: The asset returns data in any DataFrame format.
            add_constant: Indicates if constant should be included.

        Returns:
            factor_data: Readily processed factor data in a T by K DataFrame.

        """
        # set up index and constant
        factor_data = pd.DataFrame(index=returns_data.index)
        if add_constant:
            factor_data["const"] = 1

        # fill in factor data
        factor_data = factor_data.merge(
            self.factor_data,
            how="left",
            left_index=True,
            right_index=True,
        )

        # warn if factor data is missing
        if factor_data.isna().sum().sum() > 0:
            warnings.warn(
                "filling in missing factor observations (out of {}) with zeros: \n{}".format(
                    len(factor_data), factor_data.isna().sum()
                )
            )
            factor_data = factor_data.fillna(0)

        return factor_data

    def _set_up_attributes(self, returns: pd.DataFrame, factors: pd.DataFrame):
        """Set up storage arrays for fitting results.

        Args:
            returns: The preprocessed asset returns data.
            factors: The preprocessed factor data.

        """
        # K(+1) times N attributes
        self._coef_ = pd.DataFrame(index=factors.columns, columns=returns.columns)
        self._se_ = pd.DataFrame(index=factors.columns, columns=returns.columns)
        self._factor_means_ = pd.DataFrame(
            index=factors.columns, columns=returns.columns
        )

        # N times 1 attributes
        self._sigma2_ = pd.Series(index=returns.columns, name="sigma2")
        self._r2_ = pd.Series(index=returns.columns, name="R2")
        self._asset_means_ = pd.Series(index=returns.columns, name="mean_return")

        # T times N attributes
        self._fitted_ = pd.DataFrame(index=returns.index, columns=returns.columns)
        self._resid_ = pd.DataFrame(index=returns.index, columns=returns.columns)

    @staticmethod
    def _regress(returns_data: pd.Series, factor_data: pd.DataFrame) -> dict:
        """Calculate factor model regression for a single asset.

        Method will calculate regression coefficients and other statistics and
        return a dictionary with the results.

        Args:
            returns_data: The preprocessed asset returns data.
            factor_data: The preprocessed factor data.

        Returns:
            regression_stats: The regression results.

        """
        # set up
        observations = returns_data.notna()
        X = factor_data.loc[observations].values
        y = returns_data[observations].values

        # calculate
        if observations.sum() >= X.shape[1]:
            coef = np.linalg.inv(X.T @ X) @ (X.T @ y)
        else:
            coef = np.full(
                shape=[
                    X.shape[1],
                ],
                fill_value=np.nan,
            )
            warnings.warn(
                "not enough observations to estimate factor loadings for {}".format(
                    returns_data.name
                )
            )
        fitted = X @ coef
        resid = y - fitted
        sigma2 = (resid ** 2).sum() / (len(y) - X.shape[1])
        if observations.sum() >= X.shape[1]:
            se = sigma2 * np.diag(np.linalg.inv(X.T @ X))
        else:
            se = np.full(
                shape=[
                    X.shape[1],
                ],
                fill_value=np.nan,
            )
        r2 = 1 - sigma2 / y.var()

        # collect
        regression_stats = {
            "name": returns_data.name,
            "coef": coef,
            "fitted": fitted,
            "resid": resid,
            "se": se,
            "sigma2": sigma2,
            "r2": r2,
            "index": returns_data.index[observations],
            "factor_means": X.mean(axis=0),
            "asset_mean": y.mean(),
        }
        return regression_stats

    def _store_regression_stats(self, stats: dict):
        """Store the results of a factor regression in the storage arrays.

        Args:
            stats: Factor regression results.

        """
        self._coef_.loc[:, stats["name"]] = stats["coef"]

        # K(+1) times N attributes
        self._coef_.loc[:, stats["name"]] = stats["coef"]
        self._se_.loc[:, stats["name"]] = stats["se"]
        # self._factor_means_.loc[:, stats["name"]] = stats["factor_means"]

        # N times 1 attributes
        self._sigma2_.loc[stats["name"]] = stats["sigma2"]
        self._r2_.loc[stats["name"]] = stats["r2"]
        self._asset_means_.loc[stats["name"]] = stats["asset_mean"]

        # T times N attributes
        self._fitted_.loc[stats["index"], stats["name"]] = stats["fitted"]
        self._resid_.loc[stats["index"], stats["name"]] = stats["resid"]

    def fit(self, returns_data: pd.DataFrame, add_constant: bool = True):
        """Fit the factor model to an array of returns data.

        Args:
            returns_data: Asset returns data indexed by a DatetimeIndex.
            add_constant: Indicates if model is to be estimated with alpha.

        """
        # prepare
        returns_data = self._preprocess_returns_data(returns_data=returns_data)
        factor_data = self._preprocess_factor_data(
            returns_data=returns_data, add_constant=add_constant
        )
        self._set_up_attributes(returns=returns_data, factors=factor_data)

        # run regressions
        for asset, asset_returns in returns_data.items():
            regression_stats = self._regress(
                returns_data=asset_returns, factor_data=factor_data
            )
            self._store_regression_stats(stats=regression_stats)

        # update
        self.is_fitted = True
        self._sample_factor_data_ = factor_data.iloc[:, int(add_constant) :]

    @property
    def coef_(self):
        """The estimated model coefficients."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self._coef_

    @property
    def alphas_(self):
        """The estimated model alphas."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        elif not "const" in self._coef_.index:
            raise AttributeError("model fitted without intercept")
        else:
            return self._coef_.loc["const"]

    @property
    def betas_(self):
        """The estimated factor loadings."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        elif "const" in self._coef_.index:
            return self._coef_.iloc[1:, :].T
        else:
            return self._coef_.T

    @property
    def se_(self):
        """The estimated coefficient standard errors."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self._se_

    @property
    def sigma2_(self):
        """The estimated idiosyncratic volatilities."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self._sigma2_

    @property
    def r2_(self):
        """The estimated idiosyncratic volatilities."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self._r2_

    @property
    def fitted_(self):
        """The model fitted values."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self._fitted_

    @property
    def t_obs_(self):
        """The model fitted values."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self.fitted_.shape[0]

    @property
    def n_assets_(self):
        """The model fitted values."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self.fitted_.shape[1]

    @property
    def residuals_(self):
        """The model residuals."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self._resid_.astype(float)

    @property
    def factor_means_(self):
        """The mean factor returns."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self._sample_factor_data_.mean()

    @property
    def factor_vcv_(self):
        """The mean factor returns."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self._sample_factor_data_.cov()

    @property
    def asset_means_(self):
        """The mean asset returns."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self._asset_means_

    @property
    def expected_returns_(self):
        """The expected returns from the factor model estimates."""
        expected_returns = (
            (self.betas_ * self.factor_means_.iloc[-self.k_factors :].T)
            .sum(axis=1)
            .rename("expret")
        )
        return expected_returns

    def perform_grs_test(self):
        """Returns the GRS test statistic and its corresponding p-value.

        The test statistic checks the cross-sectional asset-pricing model as in
        Gibbons/Ross/Shanken (1989).
            Hypothesis: alpha1 = alpha2 = ... = alphaN = 0
        That is, if the alphas from N time series regressions on N test assets
        are jointly zero.
        Based on Cochrane (2001) Chapter 12.1

        Returns:
            f_statistic: The calculated test statistic.
            p_value: The corresponding p-value.

        """
        # dimensions
        T = self.t_obs_
        N = self.n_assets_
        K = self.k_factors

        # factor data
        factor_means = self.factor_means_
        factor_vcv = self.factor_vcv_

        # regression outputs
        alphas = self.alphas_
        residuals = self.residuals_

        # asset VCV
        asset_vcv = (T - 1) / (T - 1 - K) * np.matrix(residuals.cov())

        # GRS F-statistic
        f_statistic = (
            (T - N - K)
            / N
            * (1 + factor_means.T @ np.linalg.pinv(factor_vcv) @ factor_means) ** -1
            * (alphas.T @ np.linalg.pinv(asset_vcv) @ alphas)
        )

        # p-Value for GRS statistic: GRS ~ F(N,T-N-K)
        p_value = 1 - sp.stats.f.cdf(f_statistic, N, T - N - K)
        return (f_statistic, p_value)

    def plot_predictions(self, annual_obs: int = 1, **kwargs):
        """Plots the factor model's predictions against the realisations in the
        sample together with the 45-degree line.

        Args:
            annual_obs: The number of annual observations.

        """
        fig, ax = plt.subplots(1, 1, **kwargs)

        ax.scatter(
            self.expected_returns_ * annual_obs,
            self.asset_means_ * annual_obs,
            label="Test assets",
            marker="x",
        )
        limits = (
            max(ax.get_xlim()[0], ax.get_ylim()[0]),
            min(ax.get_xlim()[1], ax.get_ylim()[1]),
        )
        ax.plot(
            limits,
            limits,
            clip_on=True,
            scalex=False,
            scaley=False,
            label="45° Line",
            c="k",
            linewidth=1,
            linestyle=":",
        )
        ax.set_xlabel("Expected return")
        ax.set_ylabel("Realized return")
        ax.legend(loc="lower right")

        return fig

    def plot_results(self, annual_obs: int = 1, **kwargs):

        """
        Plots the factor model's estimates in 4 subplots:
        - alphas
        - betas
        - mean returns
        - r squares
        """

        fig, axes = plt.subplots(4, 1, **kwargs)

        axes[0].errorbar(
            range(1, len(self.alphas_) + 1),
            self.alphas_ * annual_obs,
            yerr=self.se_.loc["const"] * annual_obs,
            fmt="-o",
        )
        axes[0].axhline(0, color="grey", linestyle="--", linewidth=1)
        axes[0].set_title("Annual alphas & standard errors")
        axes[0].set_xticks(range(1, len(self.alphas_) + 1))
        axes[0].set_xticklabels([])
        # axes[0].xaxis.set_tick_params(labeltop=True, labelbottom=False)
        # axes[0].set_xticklabels(self.alphas_.index, rotation="vertical", y=1.1)

        for (factor_name, beta_data) in self.betas_.iteritems():
            axes[1].errorbar(
                range(1, len(self.betas_) + 1),
                beta_data,
                yerr=self.se_.loc[factor_name, :],
                fmt="-o",
                label=factor_name,
            )
        axes[1].axhline(0, color="grey", linestyle="--", linewidth=1)
        axes[1].axhline(1, color="grey", linestyle=":", linewidth=1)
        axes[1].set_title("Factor loadings (betas) & standard errors")
        axes[1].set_xticks(range(1, len(self.alphas_) + 1))
        axes[1].legend(loc="upper left")
        axes[1].set_xticklabels([])

        axes[2].plot(
            range(1, len(self.alphas_) + 1),
            self.asset_means_ * annual_obs,
            marker="o",
            label="Mean return",
        )
        axes[2].plot(
            range(1, len(self.alphas_) + 1),
            self.expected_returns_ * annual_obs,
            marker="o",
            label="Expected return",
        )
        axes[2].axhline(0, color="grey", linestyle="--", linewidth=1)
        axes[2].set_title("Return")
        axes[2].set_xticks(range(1, len(self.alphas_) + 1))
        axes[2].legend(loc="upper left")
        axes[2].set_xticklabels([])

        axes[3].plot(range(1, len(self.alphas_) + 1), self.r2_, marker="o")
        axes[3].set_title("R²")
        axes[3].set_xticks(range(1, len(self.alphas_) + 1))
        axes[3].set_xticklabels(self.r2_.index, rotation="vertical")

        return fig
