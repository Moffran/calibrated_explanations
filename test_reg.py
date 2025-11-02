import external_plugins.fast_explanations

print("Plugin meta BEFORE instantiation:")
print(external_plugins.fast_explanations.FastIntervalCalibratorPlugin.plugin_meta)

print("\nCreating instance...")
instance = external_plugins.fast_explanations.FastIntervalCalibratorPlugin()

print("\nPlugin meta AFTER instantiation:")
print(instance.plugin_meta)

print("\nStarting registration")
try:
    external_plugins.fast_explanations.register()
    print("Registration complete")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

from calibrated_explanations.plugins.registry import find_interval_descriptor, _INTERVAL_PLUGINS
print(f'\nRegistered interval plugins: {list(_INTERVAL_PLUGINS.keys())}')
desc = find_interval_descriptor('core.interval.fast')
print(f'Descriptor: {desc}')


