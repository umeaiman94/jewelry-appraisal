import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# ── Synthetic dataset mimicking Kaggle jewelry price dataset ──────────────────
def generate_dataset(n=1500, seed=42):
    rng = np.random.default_rng(seed)

    metal_types   = ["Gold 18K", "Gold 14K", "Platinum", "Silver 925", "Rose Gold 18K"]
    gemstones     = ["Diamond", "Ruby", "Emerald", "Sapphire", "Pearl", "None"]
    cut_grades    = ["Excellent", "Very Good", "Good", "Fair", "N/A"]
    conditions    = ["Mint", "Excellent", "Good", "Fair", "Poor"]

    metal_base    = {"Gold 18K": 1800, "Gold 14K": 1200, "Platinum": 2200,
                     "Silver 925": 200, "Rose Gold 18K": 1600}
    gem_base      = {"Diamond": 3000, "Ruby": 2000, "Emerald": 1800,
                     "Sapphire": 1500, "Pearl": 600, "None": 0}
    cut_mult      = {"Excellent": 1.25, "Very Good": 1.10, "Good": 1.00,
                     "Fair": 0.85, "N/A": 1.00}
    cond_mult     = {"Mint": 1.00, "Excellent": 0.92, "Good": 0.80,
                     "Fair": 0.65, "Poor": 0.45}

    rows = []
    for _ in range(n):
        metal   = rng.choice(metal_types)
        gem     = rng.choice(gemstones)
        carat   = round(float(rng.uniform(0.1, 5.0)), 2) if gem != "None" else 0.0
        cut     = rng.choice(cut_grades) if gem not in ["Pearl", "None"] else "N/A"
        cond    = rng.choice(conditions)

        price = (metal_base[metal]
                 + gem_base[gem] * carat
                 * cut_mult[cut]
                 * cond_mult[cond]
                 + rng.normal(0, 150))
        price = max(50, round(price, 2))

        rows.append({
            "metal_type": metal, "gemstone": gem, "carat_weight": carat,
            "cut_grade": cut, "condition": cond, "price_usd": price
        })

    return pd.DataFrame(rows)


# ── Encoders & scaler (fit once, reuse) ──────────────────────────────────────
class JewelryAppraisal:
    MODEL_PATH = "knn_jewelry.pkl"

    def __init__(self):
        self.le_metal = LabelEncoder()
        self.le_gem   = LabelEncoder()
        self.le_cut   = LabelEncoder()
        self.le_cond  = LabelEncoder()
        self.scaler   = StandardScaler()
        self.model    = KNeighborsRegressor(n_neighbors=7, weights="distance",
                                            metric="euclidean")
        self.trained  = False

    # ── feature engineering ──────────────────────────────────────────────────
    def _encode(self, df, fit=False):
        d = df.copy()
        # fill missing gemstone attributes
        d["gemstone"]    = d["gemstone"].fillna("None")
        d["cut_grade"]   = d["cut_grade"].fillna("N/A")
        d["carat_weight"]= d["carat_weight"].fillna(0.0)

        if fit:
            d["metal_enc"] = self.le_metal.fit_transform(d["metal_type"])
            d["gem_enc"]   = self.le_gem.fit_transform(d["gemstone"])
            d["cut_enc"]   = self.le_cut.fit_transform(d["cut_grade"])
            d["cond_enc"]  = self.le_cond.fit_transform(d["condition"])
        else:
            # handle unseen labels gracefully
            def safe_transform(le, val):
                if val in le.classes_:
                    return le.transform([val])[0]
                return 0
            d["metal_enc"] = d["metal_type"].apply(lambda v: safe_transform(self.le_metal, v))
            d["gem_enc"]   = d["gemstone"].apply(lambda v: safe_transform(self.le_gem, v))
            d["cut_enc"]   = d["cut_grade"].apply(lambda v: safe_transform(self.le_cut, v))
            d["cond_enc"]  = d["condition"].apply(lambda v: safe_transform(self.le_cond, v))

        features = d[["metal_enc", "gem_enc", "carat_weight", "cut_enc", "cond_enc"]]
        return features

    # ── train ─────────────────────────────────────────────────────────────────
    def train(self, verbose=True):
        df = generate_dataset()
        X_raw = df.drop(columns=["price_usd"])
        y     = df["price_usd"].values

        X_enc = self._encode(X_raw, fit=True)
        X_scaled = self.scaler.fit_transform(X_enc)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        self.trained = True

        if verbose:
            preds = self.model.predict(X_test)
            mae   = mean_absolute_error(y_test, preds)
            r2    = r2_score(y_test, preds)
            print(f"[Model] MAE=${mae:.2f}  R²={r2:.4f}  (k=7, distance-weighted)")

        joblib.dump(self, self.MODEL_PATH)
        return self

    # ── predict ───────────────────────────────────────────────────────────────
    def predict(self, metal_type, gemstone, carat_weight, cut_grade, condition):
        if not self.trained:
            raise RuntimeError("Model not trained.")

        row = pd.DataFrame([{
            "metal_type": metal_type,
            "gemstone": gemstone if gemstone else "None",
            "carat_weight": float(carat_weight) if carat_weight else 0.0,
            "cut_grade": cut_grade if cut_grade else "N/A",
            "condition": condition,
        }])
        X_enc    = self._encode(row, fit=False)
        X_scaled = self.scaler.transform(X_enc)
        price    = self.model.predict(X_scaled)[0]
        return round(max(50, price), 2)

    # ── load or train ─────────────────────────────────────────────────────────
    @classmethod
    def load_or_train(cls):
        if os.path.exists(cls.MODEL_PATH):
            obj = joblib.load(cls.MODEL_PATH)
            print("[Model] Loaded from cache.")
            return obj
        print("[Model] Training new model …")
        return cls().train()


# ── quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    appraisal = JewelryAppraisal().train()
    price = appraisal.predict("Gold 18K", "Diamond", 1.5, "Excellent", "Mint")
    print(f"Sample prediction: ${price:,.2f}")
