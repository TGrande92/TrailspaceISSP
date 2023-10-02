from numpy import min_scalar_type
from abstract_robot import AbstractRobot
import random


class MyRobot(AbstractRobot):
    """ Robots that moves randomly """

    def __init__(self, name,  min_x, min_y, max_x, max_y,
                 x_intial, y_initial, distance, direction, charge):
        super().__init__(name, min_x, min_y, max_x, max_y)

        if not isinstance(x_intial, int):
            raise ValueError("must be an integer")
        self._curr_x = x_intial

        if not isinstance(y_initial, int):
            raise ValueError("must be an integer")
        self._curr_y = y_initial

        if not isinstance(distance, int):
            raise ValueError("must be an integer")
        self._distance = distance

        if not isinstance(direction, str):
            raise ValueError("must be a string")
        self._direction = direction

        if not isinstance(charge, float):
            raise ValueError("must be a float")
        self._charge = charge

    def _calc_new_position_hook(self):
        """ Hook method for calculating the robot's new position """

        if self._direction == 'up':
            new_x = self._curr_x + random.randint(-1, 1)

            if self._min_y < self._curr_y - self._distance:
                new_y = self._curr_y - self._distance

            else:
                new_y = self._min_y

        if self._direction == 'down':
            new_x = self._curr_x + random.randint(-1, 1)

            if self._max_y > self._curr_y + self._distance:
                new_y = self._curr_y + self._distance
            else:
                new_y = self._max_y

        if self._direction == 'right':

            new_y = self._curr_y + random.randint(-1, 1)

            if self._max_x > self._curr_x + self._distance:
                new_x = self._curr_x + self._distance
            else:
                new_x = self._max_x

        if self._direction == 'left':
            new_y = self._curr_y + random.randint(-1, 1)

            if self._min_x < self._curr_x - self._distance:
                new_x = self._curr_x - self._distance
            else:
                new_x = self._max_x

        self._curr_x = new_x
        self._curr_y = new_y

    def _calc_remaining_battery_hook(self):
        """ Hook method for calculating the robot's remaining battery percentage """
        self._battery_percent -= self._charge

        if self._battery_percent < 0.0:
            self._battery_percent = 0.0
