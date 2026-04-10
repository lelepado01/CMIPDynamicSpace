import xarray as xr
import matplotlib.pyplot as plt
import torch
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
from tqdm import tqdm

VAR = "t2m0"
path = "climate_inferences/"
runs = [path + f for f in os.listdir(path) if not f.endswith(".zip")]#[1:2]

loss_fn = torch.nn.MSELoss() 
sc = StandardScaler()
pca = PCA(n_components=2)

samples = []
face_index = []
time_index = []
for run in tqdm(runs[:1]): 
    ds = xr.open_dataset(f"{run}/artifacts_GR0/forecast_dlwp.nc", engine="netcdf4")
    inp = torch.load(f"{run}/artifacts_GR0/input_0_0.pt")[0,:,0, 3]

    var = ds[VAR]
    n_steps = var.sizes["step"]

    for f in range(12): 
        face = var.isel(face=f)

        for step in range(n_steps): 
            time_face = face.isel(step=step)
            loss = loss_fn(torch.tensor(time_face.values).squeeze(), inp[f])

            samples.append(time_face.values.flatten() + [loss.item()])
            face_index.append(f)
            time_index.append(step)

samples = sc.fit_transform(samples)
X = pca.fit_transform(samples)

X = pd.DataFrame(X)
X.columns = ["x", "y"]
X["time_index"] = time_index
X["face_index"] = face_index

plt.figure(figsize=(16,9))
sns.scatterplot(X, x="x", y="y", style="face_index", hue="time_index")
plt.savefig(f"loss_pca_time_face_{VAR}.pdf", dpi=300)
# plt.savefig(f"loss_pca_exp_face_{VAR}.pdf", dpi=300)
plt.show()