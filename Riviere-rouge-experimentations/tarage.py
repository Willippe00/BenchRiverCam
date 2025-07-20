import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class CourbeTarageSplinePoly040241:
    def __init__(self, niveau_txt, debit_txt, arrondi=2):
        # Extraction des données (niveau, remarque)
        self.niv = self._extract_data(niveau_txt, 2, 3)
        self.niv.rename(columns={"Value": "Niveau"}, inplace=True)
        self.niv["Date"] = pd.to_datetime(self.niv["Date"])
        self.deb = self._extract_data(debit_txt, 2, 3)
        self.deb.rename(columns={"Value": "Debit"}, inplace=True)
        self.deb["Date"] = pd.to_datetime(self.deb["Date"])
        self.df = pd.merge(self.niv, self.deb, on="Date", suffixes=('_N', '_Q'))
        mask = ~self.df["Remarque_N"].isin(["R", "P"]) & ~self.df["Remarque_Q"].isin(["R", "P"])
        self.df_valid = self.df[mask].copy()
        self.df_valid["Niveau_arr"] = self.df_valid["Niveau"].round(arrondi)
        df_unique = self.df_valid.groupby("Niveau_arr").mean(numeric_only=True).reset_index()[["Niveau_arr", "Debit"]]
        df_unique = df_unique.sort_values("Niveau_arr")
        self.niveaux = df_unique["Niveau_arr"].values
        self.debits = df_unique["Debit"].values
        self.spline = CubicSpline(self.niveaux, self.debits, extrapolate=False)
        # Ajustement polynôme (loi puissance)
        def powerlaw(N, a, N0, b):
            return a * np.maximum(N - N0, 0) ** b
        # Pour l'ajustement, il faut des niveaux > N0 plausible
        Nfit = self.niveaux
        Qfit = self.debits
        N0_init = Nfit.min() - 0.1
        a_init = (Qfit.max() / np.power(Nfit.max() - N0_init, 2))
        b_init = 2.0
        popt, _ = curve_fit(powerlaw, Nfit, Qfit, p0=[a_init, N0_init, b_init], maxfev=10000)
        self.poly_a, self.poly_N0, self.poly_b = popt
        self._poly = lambda N: self.poly_a * np.maximum(N - self.poly_N0, 0) ** self.poly_b

    def _extract_data(self, filepath, value_col_idx, remark_col_idx):
        data = []
        with open(filepath, encoding="latin-1") as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) >= value_col_idx + 1:
                    try:
                        int(fields[0])
                        pd.to_datetime(fields[1])
                        val = float(fields[value_col_idx])
                        remark = fields[remark_col_idx] if len(fields) > remark_col_idx else ""
                        data.append((fields[1], val, remark))
                    except Exception:
                        continue
        return pd.DataFrame(data, columns=["Date", "Value", "Remarque"])

    def q(self, niveau):
        """
        Retourne le débit (m³/s) pour un niveau donné (m)
        Utilise la spline sur la plage connue, sinon le polynôme extrapolé.
        """
        niveau = np.asarray(niveau)
        q_spline = self.spline(niveau)
        # Spline retourne nan hors plage, on utilise la polynomiale dans ces cas
        mask_nan = np.isnan(q_spline)
        if mask_nan.any():
            q_spline[mask_nan] = self._poly(niveau[mask_nan])
        return q_spline if q_spline.size > 1 else float(q_spline)

    def plot(self, with_points=True):
        plt.figure(figsize=(9,5))
        if with_points:
            plt.scatter(self.niveaux, self.debits, color="green", s=12, label="Points (N uniques)")
        # Spline sur la plage observée
        N_grid = np.linspace(self.niveaux.min(), self.niveaux.max(), 400)
        plt.plot(N_grid, self.spline(N_grid), "b-", lw=2, label="Spline exacte Q=f(N)")
        # Polynôme en dehors de la plage
        N_all = np.linspace(self.niveaux.min()-0.5, self.niveaux.max()+0.5, 600)
        Q_poly = self._poly(N_all)
        plt.plot(N_all, Q_poly, "r--", lw=1.5, label="Polynôme (extrapolation)")
        plt.xlabel("Niveau (m)")
        plt.ylabel("Débit (m³/s)")
        plt.title("Courbe de tarage (Spline + Extrapolation polynomiale)\nStation 040241")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


tarage = CourbeTarageSplinePoly040241('040241_N.txt', '040241_Q.txt')
print(tarage.q(44.0))    # extrapolé avec polynôme
print(tarage.q(45.0))    # interpolé exact (spline)
print(tarage.q(47.0))    # extrapolé avec polynôme
tarage.plot()            # affiche spline, points, polynôme extrapolé
