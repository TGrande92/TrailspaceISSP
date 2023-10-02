import matplotlib.pyplot as plt

from abstract_robot import AbstractRobot
from my_robot import MyRobot
from random_robot import RandomRobot


def get_battery_level_color(battery_level):
    """ Returns a color to represent a battery level """
    color = ""
    if battery_level == AbstractRobot.BATTERY_FULL:
        color = "green"
    elif battery_level == AbstractRobot.BATTERY_NORM:
        color = "yellow"
    elif battery_level == AbstractRobot.BATTERY_LOW:
        color = "red"
    else:
        color = "black"

    return color


def main():
    """ Runs the robot until the battery is dead """
    my_robot = MyRobot("Andrew", -50, -50, 50, 50,
                       1, 1, 1, 'up', 5.0)
    curr_battery = my_robot.get_battery_level()

    x_coords = []
    y_coords = []

    while curr_battery != AbstractRobot.BATTERY_DEAD:

        prev_pos = my_robot.get_prev_position()
        curr_pos = my_robot.get_curr_position()
        x_coords.append(curr_pos[0])
        y_coords.append(curr_pos[1])
        plt.scatter(curr_pos[0], curr_pos[1], color=get_battery_level_color(
            curr_battery), antialiased=False)
        my_robot.move()

        # print("Battery Level: %s" % curr_battery)
        # print("Position: (%d, %d)" % (curr_pos[0], curr_pos[1]))
        print(my_robot)
        curr_battery = my_robot.get_battery_level()

    plt.title("Robot %s: Battery Level and Position" % my_robot.get_name())
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.plot(x_coords, y_coords, color="black")

    plt.show()


if __name__ == "__main__":
    main()
