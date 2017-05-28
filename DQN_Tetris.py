import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import random
from collections import deque
np.set_printoptions(threshold=10000000)

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch
observation_space_shape = 1228
action_space_shape = 800

# -----------------------------------------------------------------------------------------------------------------------
# Hyper Parameters
# -----------------------------------------------------------------------------------------------------------------------
EPISODE = 4000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode

SAVEPATH = "D:/computing/workspace/py1/save20170528/_"
SAVENAME = "TerisDQN"
SAVELIMIT = 100
saveCount = 0

INITPATH1 = "D:/computing/workspace/py1/save0/sl39"
# -----------------------------------------------------------------------------------------------------------------------


def weight_variable(shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


class DQN:
    # DQN
    def __init__(self):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        # self.state_dim = env.observation_space.shape[0]
        # self.action_dim = env.action_space.n
        self.state_dim = observation_space_shape
        self.action_dim = action_space_shape

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

        # TODO
        path = INITPATH1
        self.W1.assign(np.loadtxt(path + "W1"))
        self.b1.assign(np.loadtxt(path + "b1"))
        self.W2.assign(np.loadtxt(path + "W2"))
        self.b2.assign(np.loadtxt(path + "b2"))

    def create_Q_network(self):
        # TOIMPROVE
        # network weights
        self.W1 = weight_variable([self.state_dim, 200])
        self.b1 = bias_variable([200])
        self.W2 = weight_variable([200, self.action_dim])
        self.b2 = bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input, self.W1) + self.b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer, self.W2) + self.b2
        #TODO
        self.saver = tf.train.Saver()

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])  # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def randValidAction(self, state):
        tmpVPM = state[:800]
        vCount = sum(tmpVPM)
        choice = random.randint(0, vCount-1)
        for result in range(800):
            if tmpVPM[result]:
                choice -= 1
                if choice < 0:
                    return result
        return 0

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EPISODE
        if random.random() <= self.epsilon:
            return self.randValidAction(state)
        else:
            tmpVPM = state[:800]
            tmpVPM = [(-1000000 if x == 0 else 0) for x in tmpVPM]
            return np.argmax(Q_value + tmpVPM)

    def action(self, state):
        tmpVPM = state[:800]
        tmpVPM = [(-1000000 if x == 0 else 0) for x in tmpVPM]
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
            })[0] + tmpVPM)

    def save(self):
        # TODO
        global saveCount
        savePath = SAVEPATH + str(saveCount % SAVELIMIT) + "_"
        np.savetxt(savePath + "W1", self.session.run(self.W1))
        np.savetxt(savePath + "b1", self.session.run(self.b1))
        np.savetxt(savePath + "W2", self.session.run(self.W2))
        np.savetxt(savePath + "b2", self.session.run(self.b2))
        # self.saver.save(self.session, savePath)
        saveCount = saveCount + 1


# Variables for Teris

MAPWIDTH = 10
MAPHEIGHT = 20
elimBonus = (0, 1, 3, 5, 7)
blockShape = [
    [[0, 0, 1, 0, -1, 0, -1, -1], [0, 0, 0, 1, 0, -1, 1, -1], [0, 0, -1, 0, 1, 0, 1, 1], [0, 0, 0, -1, 0, 1, -1, 1]],
    [[0, 0, -1, 0, 1, 0, 1, -1], [0, 0, 0, -1, 0, 1, 1, 1], [0, 0, 1, 0, -1, 0, -1, 1], [0, 0, 0, 1, 0, -1, -1, -1]],
    [[0, 0, 1, 0, 0, -1, -1, -1], [0, 0, 0, 1, 1, 0, 1, -1], [0, 0, -1, 0, 0, 1, 1, 1], [0, 0, 0, -1, -1, 0, -1, 1]],
    [[0, 0, -1, 0, 0, -1, 1, -1], [0, 0, 0, -1, 1, 0, 1, 1], [0, 0, 1, 0, 0, 1, -1, 1], [0, 0, 0, 1, -1, 0, -1, -1]],
    [[0, 0, -1, 0, 0, 1, 1, 0], [0, 0, 0, -1, -1, 0, 0, 1], [0, 0, 1, 0, 0, -1, -1, 0], [0, 0, 0, 1, 1, 0, 0, -1]],
    [[0, 0, 0, -1, 0, 1, 0, 2], [0, 0, 1, 0, -1, 0, -2, 0], [0, 0, 0, 1, 0, -1, 0, -2], [0, 0, -1, 0, 1, 0, 2, 0]],
    [[0, 0, 0, 1, -1, 0, -1, 1], [0, 0, -1, 0, 0, -1, -1, -1], [0, 0, 0, -1, 1, -0, 1, -1], [0, 0, 1, 0, 0, 1, 1, 1]]
]

gridInfo = np.zeros([2, MAPHEIGHT + 2, MAPWIDTH + 2], int)
trans = np.zeros([2, 4, MAPWIDTH + 2], int)
transCount = [0, 0]
maxHeight = [0, 0]
elimTotal = [0, 0]
typeCountForColor = np.zeros([2, 7], int)
nextTypeForColor = [0, 0]
currBotColor = -1
enemyColor = -1


class Teris:
    """docstring for Teris"""

    def __init__(self, t, color):
        self.blockType = t
        self.shape = blockShape[t]
        self.color = color

    def set(self, x=-1, y=-1, o=-1):
        self.blockX = self.blockX if x == -1 else x
        self.blockY = self.blockY if y == -1 else y
        self.orientation = self.orientation if o == -1 else o
        return self

    def isValid(self, x=-1, y=-1, o=-1):
        x = self.blockX if x == -1 else x
        y = self.blockY if y == -1 else y
        o = self.orientation if o == -1 else o
        if o < 0 or o > 3:
            return False
        for i in range(4):
            tmpX = x + self.shape[o][2 * i]
            tmpY = y + self.shape[o][2 * i + 1]
            if tmpX < 1 or tmpX > MAPWIDTH or \
                    tmpY < 1 or tmpY > MAPHEIGHT or \
                    gridInfo[self.color][tmpY][tmpX]:
                return False
        return True

    def onGround(self):
        if self.isValid() and not self.isValid(-1, self.blockY - 1):
            return True
        return False

    def place(self):
        global gridInfo
        if not self.onGround():
            return False
        for i in range(4):
            tmpX = self.blockX + self.shape[self.orientation][2 * i]
            tmpY = self.blockY + self.shape[self.orientation][2 * i + 1]
            gridInfo[self.color][tmpY][tmpX] = 2
        return True


def initGrid():
    global gridInfo
    gridInfo = np.zeros([2, MAPHEIGHT + 2, MAPWIDTH + 2], int)
    for i in range(MAPHEIGHT + 2):
        gridInfo[1][i][0] = gridInfo[1][i][MAPWIDTH + 1] = -2;
        gridInfo[0][i][0] = gridInfo[0][i][MAPWIDTH + 1] = -2;
    for i in range(MAPWIDTH + 2):
        gridInfo[1][0][i] = gridInfo[1][MAPHEIGHT + 1][i] = -2;
        gridInfo[0][0][i] = gridInfo[0][MAPHEIGHT + 1][i] = -2;


def eliminate(color):
    global gridInfo, trans, transCount, maxHeight, elimTotal
    transCount[color] = 0
    maxHeight[color] = MAPHEIGHT
    for i in range(1, MAPHEIGHT + 1):
        emptyFlag = 1
        fullFlag = 1
        for j in range(1, MAPWIDTH + 1):
            if gridInfo[color][i][j] == 0:
                fullFlag = 0;
            else:
                emptyFlag = 0;
        if fullFlag:
            for j in range(1, MAPWIDTH + 1):
                trans[color][transCount[color]][j] = 1 if gridInfo[color][i][j] == 1 else 0;
                gridInfo[color][i][j] = 0;
            transCount[color] += 1;
        else:
            if emptyFlag:
                maxHeight[color] = i - 1;
                break;
            else:
                for j in range(1, MAPWIDTH + 1):
                    gridInfo[color][i - transCount[color]][j] = \
                        1 if gridInfo[color][i][j] > 0 else gridInfo[color][i][j];
                    if transCount[color]:
                        gridInfo[color][i][j] = 0;
    maxHeight[color] -= transCount[color];
    elimTotal[color] += elimBonus[transCount[color]];


def transfer():
    global gridInfo, trans, transCount, maxHeight
    color1 = 0;
    color2 = 1;
    if (transCount[color1] == 0 and transCount[color2] == 0):
        return -1;
    if (transCount[color1] == 0 or transCount[color2] == 0):
        if (transCount[color1] == 0 and transCount[color2] > 0):
            color1, color2 = color2, color1;
        maxHeight[color2] = h2 = maxHeight[color2] + transCount[color1];
        if (h2 > MAPHEIGHT):
            return color2;
        for i in range(h2, transCount[color1], -1):
            for j in range(1, MAPWIDTH + 1):
                gridInfo[color2][i][j] = gridInfo[color2][i - transCount[color1]][j];
        for i in range(transCount[color1], 0, -1):
            for j in range(1, MAPWIDTH + 1):
                gridInfo[color2][i][j] = trans[color1][i - 1][j];
        return -1;
    else:
        maxHeight[color1] = h1 = maxHeight[color1] + transCount[color2];
        maxHeight[color2] = h2 = maxHeight[color2] + transCount[color1];
        if (h1 > MAPHEIGHT):
            return color1;
        if (h2 > MAPHEIGHT):
            return color2;
        for i in range(h2, transCount[color1], -1):
            for j in range(1, MAPWIDTH + 1):
                gridInfo[color2][i][j] = gridInfo[color2][i - transCount[color1]][j];
        for i in range(transCount[color1], 0, -1):
            for j in range(1, MAPWIDTH + 1):
                gridInfo[color2][i][j] = trans[color1][i - 1][j];
        for i in range(h1, transCount[color2], -1):
            for j in range(1, MAPWIDTH + 1):
                gridInfo[color1][i][j] = gridInfo[color1][i - transCount[color2]][j];
        for i in range(transCount[color2], 0, -1):
            for j in range(1, MAPWIDTH + 1):
                gridInfo[color1][i][j] = trans[color2][i - 1][j];
        return -1;


def getValidPos(color, blockType):
    validPosMaskRaw = np.zeros([4, MAPWIDTH, MAPHEIGHT], bool);
    validPosMask = np.zeros([4, MAPWIDTH, MAPHEIGHT], bool);
    validOnGroundMask = np.zeros([4, MAPWIDTH, MAPHEIGHT], bool);
    validCount = 0;
    for y in range(MAPHEIGHT, 0, -1):
        for x in range(1, MAPWIDTH + 1):
            for o in range(4):
                validPosMaskRaw[o][x - 1][y - 1] = True;
                _def = blockShape[blockType][o];
                for i in range(4):
                    _x = _def[i * 2] + x
                    _y = _def[i * 2 + 1] + y
                    if (_y > MAPHEIGHT or _y < 1 or _x < 1 or _x > MAPWIDTH or gridInfo[color][_y][_x]):
                        validPosMaskRaw[o][x - 1][y - 1] = False;
    for y in range(MAPHEIGHT, 0, -1):
        # //Find top positions
        if y > MAPHEIGHT - 4:
            for x in range(1, MAPWIDTH + 1):
                for o in range(4):
                    if not validPosMaskRaw[o][x - 1][y - 1]:
                        continue;
                    validPosMask[o][x - 1][y - 1] = True;
                    topFlag = False;
                    _def = blockShape[blockType][o];
                    for i in range(4):
                        _y = _def[i * 2 + 1] + y;
                        if (_y == MAPHEIGHT):
                            topFlag = True;
                    if not topFlag:
                        validPosMask[o][x - 1][y - 1] = False;
        # //Descending from y+1
        if y < MAPHEIGHT:
            for x in range(1, MAPWIDTH + 1):
                for o in range(4):
                    if validPosMask[o][x - 1][y]:
                        if validPosMaskRaw[o][x - 1][y - 1]:
                            validPosMask[o][x - 1][y - 1] = True;
                            if(y == 1):
                                validOnGroundMask[o][x - 1][y - 1] = True
                        else:
                            validOnGroundMask[o][x - 1][y] = True;
        # //Moving left/right and rotating
        searchObj = np.zeros([40, 2], int);
        # //notice: searchObj[i][1] take value in [0,MAPWIDTH)
        sE = 0;
        sH = 0;
        for x in range(1, MAPWIDTH + 1):
            for o in range(4):
                if (validPosMask[o][x - 1][y - 1]):
                    searchObj[sE][0] = o;
                    searchObj[sE][1] = x - 1;
                    sE += 1;
        # No more!
        if sE == sH:
            break;
        while sE > sH:
            so = searchObj[sH][0];
            sx = searchObj[sH][1];
            if (not validPosMask[(so + 1) % 4][sx][y - 1]):
                if (validPosMaskRaw[(so + 1) % 4][sx][y - 1]):
                    validPosMask[(so + 1) % 4][sx][y - 1] = True;
                    searchObj[sE][0] = (so + 1) % 4;
                    searchObj[sE][1] = sx;
                    sE += 1
            if (sx > 0 and not validPosMask[so][sx - 1][y - 1]):
                if (validPosMaskRaw[so][sx - 1][y - 1]):
                    validPosMask[so][sx - 1][y - 1] = True;
                    searchObj[sE][0] = so;
                    searchObj[sE][1] = sx - 1;
                    sE += 1
            if (sx < MAPWIDTH - 1 and not validPosMask[(so + 1) % 4][sx][y - 1]):
                if (validPosMaskRaw[(so + 1) % 4][sx][y - 1]):
                    validPosMask[(so + 1) % 4][sx][y - 1] = True;
                    searchObj[sE][0] = (so + 1) % 4;
                    searchObj[sE][1] = sx;
                    sE += 1
            sH += 1;
        for x in range(1, MAPWIDTH + 1):
            for o in range(4):
                validCount += validPosMask[o][x - 1][y - 1];
    return validOnGroundMask, validCount;


# bot of colpor deciding next type for enemy
def getValidType(color):
    _enemyColor = 1 - color
    maxCount = max(typeCountForColor[_enemyColor])
    minCount = min(typeCountForColor[_enemyColor])
    if maxCount - minCount != 2:
        validMask = np.ones([7], int)
    else:
        validMask = [1 if i != maxCount else 0 for i in typeCountForColor[_enemyColor]]
    return validMask


def randomType(color):
    vTM = getValidType(color)
    blockForEnemy = 0
    for blockForEnemy in range(7):
        if vTM[blockForEnemy]:
            break;
    return blockForEnemy;


def printField():
    color1 = currBotColor;
    color2 = enemyColor;
    i2s = ["~~", "~~", "  ", "[]", "##"]
    vPM, _ = getValidPos(currBotColor, nextTypeForColor[currBotColor])
    for y in range(MAPHEIGHT + 1, -1, -1):
        for x in range(0, MAPWIDTH + 2):
            flag = False;
            if (x > 0 and x < MAPWIDTH + 1 and y > 0 and y < MAPHEIGHT + 1):
                for o in range(4):
                    if (vPM[o][x-1][y-1]):
                        print(" "+str(o), end="")
                        flag = True;
                        break;
            if not flag:
                print(i2s[gridInfo[color1][y][x] + 2], end="")
        for x in range(0, MAPWIDTH + 2):
            print(i2s[gridInfo[color2][y][x] + 2], end="")
        print()


def getObservation(color, blockType):
    # state = [4][20][10] + [2][20][10] + [2][7] +[2][7] = [o][y][x] + [color][y][x] + blockNum + blockValid
    """

    :rtype: list
    """
    vPM, vCount = getValidPos(color, blockType)
    state = np.zeros([1228], int)
    for i in range(800):
        # i = o*200 + (y-1)*10 + (x-1)
        state[i] = vPM[int(i / 200)][i % 10][int(i / 10) % 20]
    for i in range(0, 200):
        # i = (y-1)*10 + (x-1)
        state[i + 800] = gridInfo[color][int(i / 10)][i % 10]
    for i in range(0, 200):
        # i = (y-1)*10 + (x-1)
        state[i + 1000] = gridInfo[1 - color][int(i / 10)][i % 10]
    state[1200:1207] = typeCountForColor[color]
    state[1207:1214] = getValidType(color)
    state[1214:1221] = typeCountForColor[1 - color]
    state[1221:1228] = getValidType(1 - color)
    return state


def action2xyo(action):
    return action % 10 + 1, int(action / 10) % 20 + 1, int(action / 200)


class DQN_agent:
    # DQN Agent
    # args: tf.constant
    def __init__(self):
        self.state_dim = observation_space_shape
        self.action_dim = action_space_shape

        self.W1 = weight_variable([self.state_dim, 200])
        self.b1 = bias_variable([200])
        self.W2 = weight_variable([200, self.action_dim])
        self.b2 = bias_variable([self.action_dim])

        self.create_Q_network()
        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # input layer
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input, self.W1) + self.b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer, self.W2) + self.b2
        # TODO
        # self.saver = tf.train.Saver()

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def actionValid(self, state):
        tmpVPM = state[:800]
        tmpVPM = [(-1000000 if x == 0 else 0) for x in tmpVPM]
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0] + tmpVPM)

    def changeArgs(self):
        global saveCount
        #TODO
        # if(saveCount > 0):
        #     self.saver.restore(self.session, SAVEPATH
        #             + str(random.randint(0, min([saveCount, SAVELIMIT]) - 1)))
        #     self.saver.restore(self.session, SAVEPATH)
        if(saveCount > 0):
            path = SAVEPATH + str(random.randint(0, min([saveCount, SAVELIMIT]) - 1)) + "_"
            self.W1.assign(np.loadtxt(path + "W1"))
            self.b1.assign(np.loadtxt(path + "b1"))
            self.W2.assign(np.loadtxt(path + "W2"))
            self.b2.assign(np.loadtxt(path + "b2"))

class terisEnv:
    """terisEnv"""
    def __init__(self):
        # TOIMPROVE
        self.enemyAgent = DQN_agent()
        self.reset()

    def reset(self):
        global gridInfo, trans, transCount, maxHeight, elimTotal, typeCountForColor
        initGrid()
        trans = np.zeros([2, 4, MAPWIDTH + 2], int)
        transCount = [0, 0]
        maxHeight = [0, 0]
        elimTotal = [0, 0]
        typeCountForColor = np.zeros([2, 7], int)
        self.nextTypeForColor = [random.randint(0, 6), random.randint(0, 6)]
        for i in range(MAPHEIGHT + 2):
            gridInfo[1][i][0] = gridInfo[1][i][MAPWIDTH + 1] = -2
            gridInfo[0][i][0] = gridInfo[0][i][MAPWIDTH + 1] = -2
        for i in range(MAPWIDTH + 2):
            gridInfo[1][0][i] = gridInfo[1][MAPHEIGHT + 1][i] = -2
            gridInfo[0][0][i] = gridInfo[0][MAPHEIGHT + 1][i] = -2
        self.enemyAgent.changeArgs()
        return getObservation(0, self.nextTypeForColor[0])

    def step(self, action):
        global currBotColor, enemyColor
        LOSEREWARD = -100
        WINREWARD = 100

        preElimTotal = elimTotal
        preMaxHeight = maxHeight
        # The color for the main bot is 0
        currBotColor = 0
        enemyColor = 1
        currTypeForColor = (self.nextTypeForColor[0], self.nextTypeForColor[1]);
        # read action
        # action = [4][20][10] = [o][y][x]
        x, y, o = action2xyo(action)
        blockType = randomType(currBotColor)

        # check if action is valid
        vPM, vCount = getValidPos(currBotColor, currTypeForColor[currBotColor])
        if not vPM[o][x - 1][y - 1]:
            observation = getObservation(currBotColor, self.nextTypeForColor[currBotColor]);
            return observation, LOSEREWARD, True, 0

        # prepare for enemy
        e_observation = getObservation(enemyColor, self.nextTypeForColor[enemyColor]);
        # ensure valid
        e_action = int(self.enemyAgent.actionValid(e_observation))
        # print(e_action)
        e_x, e_y, e_o = action2xyo(e_action)
        e_blockType = randomType(enemyColor)
        # print(e_x,e_y,e_o)

        # simulate
        myBlock = Teris(currTypeForColor[currBotColor], currBotColor);
        myBlock.set(x, y, o).place();

        # // 鎴戠粰瀵规柟浠�涔堝潡鏉ョ潃锛�
        typeCountForColor[enemyColor][blockType] += 1;
        self.nextTypeForColor[enemyColor] = blockType;

        # // 瀵规柟褰撴椂鎶婁笂涓�鍧楄惤鍒颁簡 e_x e_y e_o锛�
        enemyBlock = Teris(currTypeForColor[enemyColor], enemyColor);
        enemyBlock.set(e_x, e_y, e_o).place();

        # // 瀵规柟缁欐垜浠�涔堝潡鏉ョ潃锛�
        typeCountForColor[currBotColor][e_blockType] += 1;
        self.nextTypeForColor[currBotColor] = e_blockType;

        # // 妫�鏌ユ秷鍘�
        eliminate(0);
        eliminate(1);

        # // 杩涜杞Щ
        trans_boom = transfer();

        observation = getObservation(currBotColor, self.nextTypeForColor[currBotColor]);

        _, vCount = getValidPos(currBotColor, self.nextTypeForColor[currBotColor])
        _, e_vCount = getValidPos(enemyColor, self.nextTypeForColor[enemyColor])
        # TOIMPROVE
        if trans_boom == currBotColor or vCount == 0:
            done = True
            reward = LOSEREWARD
        else:
            if trans_boom == enemyColor or e_vCount == 0:
                done = True
                reward = WINREWARD
            else:
                done = False
                reward = elimTotal[currBotColor] - preElimTotal[currBotColor] \
                    - (preMaxHeight[currBotColor] - maxHeight[currBotColor])
        info = 0
        observation = [float(i) for i in observation]
        return observation, reward, done, info


def main():
    # initialize Teris env and dqn agent
    global saveCount
    env = terisEnv()
    agent = DQN()

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        # Train
        for step in range(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            # Define reward for agent
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        # Test every 100 episodes
        if episode % 100 == 0:
            # outF = open(SAVEPATH + str(saveCount) + ".txt", 'w')
            # print("W1 =", agent.session.run(agent.W1), file=outF)
            # print("b1 =", agent.session.run(agent.b1), file=outF)
            # print("W2 =", agent.session.run(agent.W2), file=outF)
            # print("b2 =", agent.session.run(agent.b2), file=outF)
            # outF.close()
            agent.save()
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    # env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        print(j)
                        printField()
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
            # if ave_reward >= 200:
            #   break
    print("saveCount=", saveCount)


if __name__ == '__main__':
    main()

