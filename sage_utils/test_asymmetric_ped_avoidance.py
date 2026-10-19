import importlib.util
import sys
import types
import unittest
from pathlib import Path


class _Settings:
    def __init__(self):
        self.values = {"dynamic_avoidance": False}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value


def _load_patched_module(settings):
    class _CarbUtil:
        pass

    class _PeopleSettings:
        DYNAMIC_AVOIDANCE_ENABLED = "dynamic_avoidance"

    modules = {
        "omni": types.ModuleType("omni"),
        "omni.metropolis": types.ModuleType("omni.metropolis"),
        "omni.metropolis.utils": types.ModuleType("omni.metropolis.utils"),
        "omni.metropolis.utils.carb_util": types.SimpleNamespace(CarbUtil=_CarbUtil),
        "omni.metropolis.utils.simulation_util": types.SimpleNamespace(
            SimulationUtil=object
        ),
        "omni.anim": types.ModuleType("omni.anim"),
        "omni.anim.people": types.ModuleType("omni.anim.people"),
        "omni.anim.people.settings": types.SimpleNamespace(
            CommandID=types.SimpleNamespace(auto_prefix="auto"),
            MetadataTag=object,
            PeopleSettings=_PeopleSettings,
            TaskStatus=types.SimpleNamespace(default="default"),
        ),
        "carb": types.SimpleNamespace(
            settings=types.SimpleNamespace(get_settings=lambda: settings)
        ),
        "fake": types.ModuleType("fake"),
        "fake.commands": types.ModuleType("fake.commands"),
        "fake.utils": types.SimpleNamespace(Utils=object),
        "fake.navigation_manager": types.SimpleNamespace(NavigationManager=object),
        "fake.commands.command_format_helper": types.SimpleNamespace(
            CommandFormatHelper=object
        ),
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        source = Path(__file__).with_name("patched_base_command.py")
        spec = importlib.util.spec_from_file_location(
            "fake.commands.base_command", source
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class _NavigationManager:
    def __init__(self, settings, raises=False):
        self.settings = settings
        self.raises = raises
        self.observed = []

    def update_path(self):
        self.observed.append(self.settings.get("dynamic_avoidance"))
        if self.raises:
            raise RuntimeError("update failed")


class AsymmetricPedestrianAvoidanceTest(unittest.TestCase):
    def setUp(self):
        self.settings = _Settings()
        self.module = _load_patched_module(self.settings)

    def test_priority_character_does_not_avoid(self):
        manager = _NavigationManager(self.settings)
        self.module._update_path_with_asymmetric_avoidance(manager, "Character")
        self.assertEqual(manager.observed, [False])
        self.assertFalse(self.settings.get("dynamic_avoidance"))

    def test_non_priority_character_avoids(self):
        manager = _NavigationManager(self.settings)
        self.module._update_path_with_asymmetric_avoidance(manager, "Character_01")
        self.assertEqual(manager.observed, [True])
        self.assertFalse(self.settings.get("dynamic_avoidance"))

    def test_setting_is_restored_after_failure(self):
        manager = _NavigationManager(self.settings, raises=True)
        with self.assertRaisesRegex(RuntimeError, "update failed"):
            self.module._update_path_with_asymmetric_avoidance(
                manager, "Character_01"
            )
        self.assertFalse(self.settings.get("dynamic_avoidance"))


if __name__ == "__main__":
    unittest.main()
