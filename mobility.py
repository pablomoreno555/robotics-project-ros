from RPi import GPIO
from time import sleep

'''
Mobility Class
--------------

Description
-----------
The mobility class controls the movement of a skidsteer robot. It uses the GPIO
module to link up to general purpose pins on a Raspberry Pi to control the motors
of the robot. There are four links:

l_forward -> Connection to the pin that moves the left motor forward

r_forward -> Connection to the pin that moves the right motor forward

l_reverse -> Connection to the pin that moves the left motor in reverse

r_reverse -> Connection to the pin that moves the right motor in reverse

l_pwm     -> Connection to left velocity motor

r_pwm     -> Connection to left velocity motor

The is_set attribute determines if the pins have been set to receive output.
'''

class Mobility():
    def __init__(self, l_forward, r_forward, l_reverse, r_reverse, l_velocity, r_velocity):
        self.l_forward = l_forward
        self.r_forward = r_forward
        self.l_reverse = l_reverse
        self.r_reverse = r_reverse
        self.l_velocity = l_velocity
        self.r_velocity = r_velocity

        self.is_set = True
        GPIO.setmode(GPIO.BCM)

        GPIO.setup(self.l_forward, GPIO.OUT)
        GPIO.setup(self.r_forward, GPIO.OUT)
        GPIO.setup(self.l_reverse, GPIO.OUT)
        GPIO.setup(self.r_reverse, GPIO.OUT)
        GPIO.setup(self.l_velocity, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.r_velocity, GPIO.OUT, initial=GPIO.HIGH)

        self.l_pwm = GPIO.PWM(self.l_velocity, 1500)
        self.r_pwm = GPIO.PWM(self.r_velocity, 1500)

        self.l_pwm.start(0)
        self.r_pwm.start(0)


    def forward(self, dur=2, duty=50):
        GPIO.output(self.l_forward, GPIO.HIGH)
        GPIO.output(self.l_reverse, GPIO.LOW)
        GPIO.output(self.r_forward, GPIO.HIGH)
        GPIO.output(self.r_reverse, GPIO.LOW)
        self.l_pwm.ChangeDutyCycle(duty)
        self.r_pwm.ChangeDutyCycle(duty)

        sleep(dur)

        GPIO.output(self.l_forward, GPIO.LOW)
        GPIO.output(self.r_forward, GPIO.LOW)

    def backward(self, dur=2, duty=50):
        GPIO.output(self.l_forward, GPIO.LOW)
        GPIO.output(self.l_reverse, GPIO.HIGH)
        GPIO.output(self.r_forward, GPIO.LOW)
        GPIO.output(self.r_reverse, GPIO.HIGH)
        self.l_pwm.ChangeDutyCycle(duty)
        self.r_pwm.ChangeDutyCycle(duty)

        sleep(dur)

        GPIO.output(self.l_reverse, GPIO.LOW)
        GPIO.output(self.r_reverse, GPIO.LOW)


    def turn_right(self):
        GPIO.output(self.l_forward, GPIO.HIGH)
        GPIO.output(self.l_reverse, GPIO.LOW)
        GPIO.output(self.r_forward, GPIO.LOW)
        GPIO.output(self.r_reverse, GPIO.HIGH)
        self.l_pwm.ChangeDutyCycle(50)
        self.r_pwm.ChangeDutyCycle(50)

        sleep(.67)

        GPIO.output(self.l_forward, GPIO.LOW)
        GPIO.output(self.r_reverse, GPIO.LOW)


    def turn_left(self):
        GPIO.output(self.l_forward, GPIO.LOW)
        GPIO.output(self.l_reverse, GPIO.HIGH)
        GPIO.output(self.r_forward, GPIO.HIGH)
        GPIO.output(self.r_reverse, GPIO.LOW)
        self.l_pwm.ChangeDutyCycle(50)
        self.r_pwm.ChangeDutyCycle(50)

        sleep(.67)

        GPIO.output(self.l_reverse, GPIO.LOW)
        GPIO.output(self.r_forward, GPIO.LOW)


    def go_right(self):
        self.turn_right()
        self.forward()
        self.turn_left()

    def go_left(self):
        self.turn_left()
        self.forward()
        self.turn_right()

    def square(self):
        for i in range(0, 4):
            self.turn_right()
            self.forward()

    def spin(self):
        for i in range(0, 4):
            self.turn_left()


    def motor_test(self, dur=1):
        GPIO.output(self.l_forward, GPIO.HIGH)
        self.l_pwm.ChangeDutyCycle(20)
        sleep(dur)
        GPIO.output(self.l_forward, GPIO.LOW)
        self.l_pwm.ChangeDutyCycle(0)

        GPIO.output(self.r_forward, GPIO.HIGH)
        self.r_pwm.ChangeDutyCycle(20)
        sleep(dur)
        GPIO.output(self.r_forward, GPIO.LOW)
        self.r_pwm.ChangeDutyCycle(20)

        GPIO.output(self.l_reverse, GPIO.HIGH)
        self.l_pwm.ChangeDutyCycle(20)
        sleep(dur)
        GPIO.output(self.l_reverse, GPIO.LOW)
        self.l_pwm.ChangeDutyCycle(0)

        GPIO.output(self.r_reverse, GPIO.HIGH)
        self.r_pwm.ChangeDutyCycle(20)
        sleep(dur)
        GPIO.output(self.r_reverse, GPIO.LOW)
        self.r_pwm.ChangeDutyCycle(0)

    def clean_up(self):
        self.is_set = False
        self.l_pwm.stop()
        self.r_pwm.stop()
        GPIO.cleanup()

    def reset_pins(self, new_l_for, new_r_for, new_l_rev, new_r_rev, new_l_vel, new_r_vel):
        self.clean_up()
        GPIO.setmode(GPIO.BCM)

        self.l_forward = new_l_for
        self.r_forward = new_r_for
        self.l_reverse = new_l_rev
        self.r_reverse = new_r_rev
        self.l_velocity = new_l_vel
        self.r_velocity = new_r_vel

        self.l_pwm = GPIO.PWM(self.l_velocity, 0)
        self.r_pwm = GPIO.PWM(self.r_velocity, 0)
