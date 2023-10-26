
import random


class AbstractRobot:
    """ Template for an autonomous robot that moves in a grid """

    BATTERY_DEAD = "DEAD"
    BATTERY_LOW = "LOW"
    BATTERY_NORM = "NORM"
    BATTERY_FULL = "FULL"

    def __init__(self, name, min_x, min_y, max_x, max_y):
        """ Sets the name of the robot and the extends of its movement """
        self._name = name

        # Set the extends of the grid in which the robot can move
        self._min_x = min_x
        self._min_y = min_y
        self._max_x = max_x
        self._max_y = max_y

        # Initial battery level is 100%
        self._battery_percent = 100.0

        # Initial positions are at the origin of the grid - 0/0
        self._prev_x = 0
        self._prev_y = 0
        self._curr_x = 0
        self._curr_y = 0

    def get_name(self):
        """ Returns the robot name """
        return self._name

    def get_curr_position(self):
        """ Returns the current position of the robot as a tuple (x, y) """
        return (self._curr_x, self._curr_y)

    def get_prev_position(self):
        """ Returns the current position of the robot as a tuple (x, y) """
        return (self._prev_x, self._prev_y)

    def get_battery_level(self):
        """ Gets the battery level (full, normal, low, dead) """
        if self._battery_percent > 70.0:
            return AbstractRobot.BATTERY_FULL
        elif self._battery_percent > 30.0:
            return AbstractRobot.BATTERY_NORM
        elif self._battery_percent > 0.0:
            return AbstractRobot.BATTERY_LOW
        else:
            return AbstractRobot.BATTERY_DEAD

    def move(self):
        """ Template method for having the robot select a new position to move to """
        self._prev_x = self._curr_x
        self._prev_y = self._curr_y

        self._calc_new_position_hook()
        self._validate_curr_position()
        self._calc_remaining_battery_hook()
        self._validate_battery_level()

    def _calc_new_position_hook(self):
        """ Hook method for calculating the robot's new position """
        raise NotImplementedError(
            "Hook Method Must Be Implemented by Child Class")
        # new_x = self._curr_x + random.randint(-4, 4)
        # new_y = self._curr_y + random.randint(-4, 4)

        # if new_x < self._min_x:
        #     new_x = self._min_x
        # elif new_x > self._max_x:
        #     new_x = self._max_x

        # if new_y < self._min_y:
        #     new_y = self._min_y
        # elif new_y > self._max_y:
        #     new_y = self._max_y

        # self._curr_x = new_x
        # self._curr_y = new_y

    def _calc_remaining_battery_hook(self):
        """ Hook method for calculating the robot's remaining battery percentage """
        raise NotImplementedError(
            "Hook Method Must Be Implemented by Child Class")
        # raise NotImplementedError(
        # o Decrement the battery percent (_battery_percent) by a random value between
        # 0.0 and 1.0 (hint: use random.random() from the built-in random module) o
        # Make sure it does not decrement to a value lower than 0.0
        # self._battery_percent -= random.random()
        # if self._battery_percent < 0.0:
        #     self._battery_percent = 0.0

    def _validate_curr_position(self):
        """ Validates that the robot's current position is within the grid """
        if self._curr_x < self._min_x or self._curr_x > self._max_x:
            raise ValueError("New Position X Out Of Range")

        if self._curr_y < self._min_y or self._curr_y > self._max_y:
            raise ValueError("New Position Y Out Of Range")

    def _validate_battery_level(self):
        """ Validates that the robot's battery percentage is within range """
        if self._battery_percent < 0.0 or self._battery_percent > 100.0:
            raise ValueError("Battery Level Out of Range")

    def __str__(self):
        """ Returns description """
        description = "Robot: %s. Current robot coordinate is (%s, %s), with %d battery level" % \
            (self._name, self._curr_x, self._curr_y, self._battery_percent)

        return description
