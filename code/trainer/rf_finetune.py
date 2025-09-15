import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch


class RandomForestFineTuner:
    def __init__(self, train_loader, valid_loader, n_estimators=200, max_depth=None, n_jobs=-1, random_state=42):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
            min_samples_split=2,
            min_samples_leaf=2,
            criterion="squared_error",
            max_features="sqrt",
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_maes, self.valid_maes = [], []
        self.train_mses, self.valid_mses = [], []
        self.train_r2s, self.valid_r2s = [], []

    def _loader_to_numpy(self, loader, bert_model=None, device="cuda"):
        X, y, areas = [], [], []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for data in loader:
            if bert_model is not None:
                bert_model.eval()
                with torch.no_grad():
                    x = data["bert_input"].to(device)
                    doy = data["timestamp"].to(device)
                    mask = data["bert_mask"].to(device)
                    area = data["area_ha"].cpu().numpy()

                    # Só extrair embedding, sem passar pelo transformer
                    embedding = bert_model(x, doy, mask, use_transformer=False)  # [batch, seq_len, hidden]
                    x = embedding.mean(dim=1)  # Pooling temporal
                    x = x.cpu().numpy()

            else:
                x = data["bert_input"].reshape(data["bert_input"].size(0), -1).numpy()

            labels = data["class_label"].view(-1).numpy()
            X.append(x)
            y.append(labels)
            areas.append(area)

        return np.vstack(X), np.concatenate(y), areas

    def train(self, bert_model=None, device="cuda"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train, y_train, areas = self._loader_to_numpy(self.train_loader, bert_model, device)
        X_valid, y_valid, areas = self._loader_to_numpy(self.valid_loader, bert_model, device)

        self.model.fit(X_train, y_train)

        # Predições
        y_pred_train = self.model.predict(X_train)
        y_pred_valid = self.model.predict(X_valid)

        # Métricas treino
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)

        # Métricas validação
        valid_mae = mean_absolute_error(y_valid, y_pred_valid)
        valid_mse = mean_squared_error(y_valid, y_pred_valid)
        valid_r2 = r2_score(y_valid, y_pred_valid)

        self.train_maes.append(train_mae)
        self.train_mses.append(train_mse)
        self.train_r2s.append(train_r2)
        self.valid_maes.append(valid_mae)
        self.valid_mses.append(valid_mse)
        self.valid_r2s.append(valid_r2)

        print(f"Train - MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"Valid - MAE: {valid_mae:.4f}, MSE: {valid_mse:.4f}, R²: {valid_r2:.4f}")

        return {
            "train": {"MAE": train_mae, "MSE": train_mse, "R2": train_r2},
            "valid": {"MAE": valid_mae, "MSE": valid_mse, "R2": valid_r2},
        }

    def test(self, test_loader, bert_model=None, device="cuda"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_test, y_test, areas = self._loader_to_numpy(test_loader, bert_model, device)
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Test - MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")
        return {"MAE": mae, "MSE": mse, "R2": r2}, y_test, y_pred, areas

