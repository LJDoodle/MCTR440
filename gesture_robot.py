from controller import Robot
import socket

# -----------------------------
# INIT WEBOTS ROBOT
# -----------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# -----------------------------
# SOCKET SERVER (RECEIVE FROM PYTHON)
# -----------------------------
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1", 5000))
server.listen(1)

print("Waiting for Python client...")
conn, addr = server.accept()
print("Connected:", addr)

current_command = "stop"

# -----------------------------
# MAIN LOOP
# -----------------------------
while robot.step(timestep) != -1:

    try:
        data = conn.recv(1024).decode().strip()
        if data:
            current_command = data
            print("Received:", current_command)

    except:
        pass

    # -------------------------
    # CONTROL LOGIC
    # -------------------------
    if current_command == "forward":
        left_motor.setVelocity(5)
        right_motor.setVelocity(5)

    elif current_command == "left":
        left_motor.setVelocity(-2)
        right_motor.setVelocity(2)

    elif current_command == "right":
        left_motor.setVelocity(2)
        right_motor.setVelocity(-2)

    else:  # stop
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)