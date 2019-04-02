# %%
from lifelines import KaplanMeierFitter
from lifelines.datasets import load_dd
import matplotlib

data = load_dd()
data.head()

# %%
kmf = KaplanMeierFitter()

# %%
data.columns

# %%
data.head()

# %%
T = data["duration"]
E = data["observed"]

kmf.fit(T, event_observed=E)

# %%
T

# %%
E

# %%
kmf.survival_function_.plot()


# %%
kmf.predict([20, 2])

# %%
kmf.survival_function_at_times([20, 2])

# %%
kmf

# %%
