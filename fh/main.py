from RobotEnv import RobotEnv

# set global variable
MAX_EPISODES = 500
MAX_EP_STEPS = 200
LAT_RANGE = 7  # 30-90
LON_RANGE = 36  # 0-350

# set environment
env = RobotEnv()

# set learning policy
pass

max_visibility = 0

lat_list = []
for i in range(30, 80, 10):
    lat_list.append(str(i))
lon_list = []
for i in range(0, 36, 10):
    lon_list.append(str(i))

visibility = {}

# import dataset
for i in lat_list:
    visibility[i] = {}
    for j in lon_list:
        fo = open(j + "_" + i + ".txt", "rw+")
        line = fo.read()
        visibility[i][j] = line
        if visibility[i][j] > max_visibility:
            max_visibility = visibility[i][j]
        fo.close()


# start training
for i in range(MAX_EPISODES):
    s = env.reset()                 # initial
    for j in range(MAX_EP_STEPS):
        env.render()                # render env
        act = rl.choose_action(s)     # RL choose action
        s_, r, done = env.step(act)   # act on env
