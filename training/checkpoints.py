import torch
import torch.nn as nn

import pandas as pd


def checkpoint(
        cfg: dict, 
        dt: nn.Module, 
        report: dict,
    ) -> None:
    epochs = cfg["epochs"]
    dt_weights = cfg["dt_weights"]
    torch.save(dt.state_dict(), dt_weights)
    csv_name = f"{dt_weights}-epochs{epochs}-report.csv"
    pd.DataFrame(report).to_csv(csv_name, index=False)