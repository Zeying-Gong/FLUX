# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import annotations
from typing import Any
from omni.metropolis.utils.carb_util import CarbUtil
from omni.metropolis.utils.simulation_util import SimulationUtil
from omni.anim.people.settings import CommandID, MetadataTag

from ..utils import Utils
from omni.anim.people.settings import CommandID, TaskStatus
from ..navigation_manager import NavigationManager
from .command_format_helper import CommandFormatHelper


class Command(CommandFormatHelper):
    """
    Base class for command, provides default implementation for setup, update, execute and exit_command.

    Also implements the walk function which moves a character to a location based on the target location set in NavigationManager.
    """

    @classmethod
    def is_valid_command(
        cls, command: list[str] | str, target_character_pos: Any = None, target_character_name: str | None = None
    ):
        """check whether the command is correct"""
        parameter_dict = cls.validate_command_format(command=command)
        if not parameter_dict:
            return False
        return True

    def __init__(
        self,
        character,
        command: list[str] = [],
        navigation_manager: NavigationManager = None,
        character_name: str = "",
        command_id: str | None = None,
        update_metadata_callback_fn=None,
        # character_prim_path = ""
    ):
        """
        Initialize the command instance, the command_id and character_name attribute are used to update action tag system
        """
        self.character = character
        self.command = command
        self.navigation_manager = navigation_manager
        self.time_elapsed = 0
        self.is_setup = False
        self.desired_walk_speed = 0
        self.actual_walk_speed = 0
        self.rotation_time = 0
        self.rotation_time_threshold = 2
        self.set_rotation = None
        self.char_start_rot = None
        self.duration = 5
        self.finished = False
        self.command_name = "Base"
        # target character that conducting this command
        self.character_name = character_name
        # command id to distinguish different command
        if not command_id:
            # if command id is not defined
            command_id = Utils.generate_unique_id(character_name=self.character_name, prefix=CommandID.auto_prefix)
        self.command_id = command_id
        self.update_metadata_callback = update_metadata_callback_fn
        self.command_description = ""
        self.command_status = TaskStatus.default
        # self.character_prim_path = character_prim_path

    def get_command_id(self):
        """get current command id"""
        return self.command_id

    def get_command_name(self):
        """get current command"""
        return self.command_name

    def set_status(self, target_status):
        """set command's status"""
        self.command_status = target_status

    def setup(self):
        self.time_elapsed = 0
        self.desired_walk_speed = 0
        self.actual_walk_speed = 0
        self.is_setup = True

    def exit_command(self):
        self.is_setup = False
        self.character.set_variable("Action", "None")
        self.update_metadata_callback(
            agent_name=self.character_name, data_name=MetadataTag.AgentActionTag, data_value="Idle"
        )
        return True

    def force_quit_command(self):
        ## clean all the affect from this command on the navigation system
        self.desired_walk_speed = 0.0
        self.character.set_variable("Action", "None")
        self.navigation_manager.set_path_points(None)
        self.navigation_manager.set_path_target_rot(None)
        self.navigation_manager.clean_path_targets()
        self.is_setup = False
        self.finished = True
        self.update_metadata_callback(
            agent_name=self.character_name, data_name=MetadataTag.AgentActionTag, data_value="Idle"
        )
        # self.command_status = TaskStatus.interrupted
        return

    def update(self, dt):
        self.time_elapsed += dt
        if self.time_elapsed > self.duration:
            return self.exit_command()

    def execute(self, dt):
        if self.finished:
            return True

        if not self.is_setup:
            self.setup()
        return self.update(dt)

    def rotate(self, dt):
        if self.set_rotation == False:
            rot_diff = self.navigation_manager.calculate_rotation_diff()
            self.rotation_time_threshold = (rot_diff / 90) * Utils.CONFIG["SecondPerNightyDegreeTurn"]
            self.set_rotation = True

        trans, rot = Utils.get_character_transform(self.character)
        target_rot = self.navigation_manager.get_path_target_rot()

        if CarbUtil.dot4(rot, target_rot) < 0.0:
            target_rot = CarbUtil.scale4(target_rot, -1.0)

        if self.rotation_time > self.rotation_time_threshold:
            self.char_start_rot = None
            self.rotation_time = 0
            self.character.set_world_transform(trans, target_rot)
            return True

        if self.char_start_rot is None:
            self.char_start_rot = rot

        # Calculate fraction of rotation at each delta time and set that rotation.
        self.rotation_time += dt
        if self.rotation_time_threshold != 0:
            time_fraction_of_completion = min(self.rotation_time / self.rotation_time_threshold, 1.0)
        else:
            time_fraction_of_completion = 1
        rotation_fraction = CarbUtil.nlerp4(self.char_start_rot, target_rot, time_fraction_of_completion)
        self.character.set_world_transform(trans, rotation_fraction)

    def walk(self, dt):
        if self.navigation_manager.destination_reached():
            self.desired_walk_speed = 0.0
            if self.actual_walk_speed < 0.001:
                self.character.set_variable("Action", "None")
                self.navigation_manager.set_path_points(None)
                self.update_metadata_callback(
                    agent_name=self.character_name, data_name=MetadataTag.AgentActionTag, data_value="Idle"
                )
                if self.navigation_manager.get_path_target_rot() is not None:
                    if self.rotate(dt):
                        self.character.set_variable("Action", "None")
                        self.navigation_manager.set_path_target_rot(None)
                        self.navigation_manager.clean_path_targets()
                        return True
                    return False
                else:
                    self.character.set_variable("Action", "None")
                    self.navigation_manager.clean_path_targets()
                    return True
        else:
            self.set_rotation = False
            try:
                import carb
                ws = carb.settings.get_settings().get("/exts/people_sim/character_walk_speed")
                self.desired_walk_speed = ws if (ws is not None and ws > 0.01) else 1.0
            except Exception:
                self.desired_walk_speed = 1.0

        self.update_metadata_callback(
            agent_name=self.character_name, data_name=MetadataTag.AgentActionTag, data_value="Walking"
        )
        self.character.set_variable("Action", "Walk")
        self.navigation_manager.update_path()
        self.character.set_variable("PathPoints", self.navigation_manager.get_path_points())

        # Blends walking animation when starting or stopping.
        max_change = dt / Utils.CONFIG["WalkBlendTime"]
        delta_walk = CarbUtil.clamp(self.desired_walk_speed - self.actual_walk_speed, -1 * max_change, max_change)
        self.actual_walk_speed = CarbUtil.clamp(self.actual_walk_speed + delta_walk, 0.0, 1.0)
        self.character.set_variable("Walk", self.actual_walk_speed)

    def fetch_command_info(self):
        """
        Fetch command information to trace current command status
        """
        character_name = self.character_name
        command_name = self.command_name
        command_id = self.command_id
        command_description = self.command_description
        command_body = " ".join(self.command)
        entire_command = f"{character_name} {command_body}"
        status = self.command_status
        time_code = SimulationUtil.get_current_timecode()

        command_info = {
            "agent_name": character_name,
            "command_name": command_name,
            "command_id": command_id,
            "command_description": command_description,
            "command": entire_command,
            "time_code": time_code,
            "status": status,
        }
        return command_info
