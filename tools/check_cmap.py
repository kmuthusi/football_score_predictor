import importlib
mod = importlib.import_module('app')
print('HAS_MATPLOTLIB=', mod.HAS_MATPLOTLIB)
print("resolve YlOrRd ->", mod._resolve_matplotlib_cmap('YlOrRd'))
print("resolve Blues ->", mod._resolve_matplotlib_cmap('Blues'))
print("resolve bogus ->", mod._resolve_matplotlib_cmap('bogus_cmap'))
