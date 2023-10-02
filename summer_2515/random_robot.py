from abstract_robot import AbstractRobot
import random


class RandomRobot(AbstractRobot):
    """ Robots that moves randomly """

    def _calc_new_position_hook(self):
        """ Hook method for calculating the robot's new position """
        new_x = self._curr_x + random.randint(-4, 4)
        new_y = self._curr_y + random.randint(-4, 4)

        if new_x < self._min_x:
            new_x = self._min_x
        elif new_x > self._max_x:
            new_x = self._max_x

        if new_y < self._min_y:
            new_y = self._min_y
        elif new_y > self._max_y:
            new_y = self._max_y

        self._curr_x = new_x
        self._curr_y = new_y

    def _calc_remaining_battery_hook(self):
        """ Hook method for calculating the robot's remaining battery percentage """
        self._battery_percent -= random.random()

        if self._battery_percent < 0.0:
            self._battery_percent = 0.0
