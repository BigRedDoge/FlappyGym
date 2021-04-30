from itertools import cycle
import random
import sys
import numpy as np
import math
import pygame
from pygame.locals import *
from ast import literal_eval


try:
    xrange
except NameError:
    xrange = range


class FlappyBird(object):

    def __init__(self, width, height):
        self.FPS = 30
        self.SCREENWIDTH  = width # 288
        self.SCREENHEIGHT = height # 512
        self.PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
        self.BASEY        = self.SCREENHEIGHT * 0.79
        # image, sound and hitmask  dicts
        self.IMAGES, self.SOUNDS, self.HITMASKS = {}, {}, {}
        self.render = False
        self.sound = False
        self.image_obs = True
        
        # list of all possible players (tuple of 3 positions of flap)
        PLAYERS_LIST = (
            # red bird
            (
                'FlappyBird/assets/sprites/redbird-upflap.png',
                'FlappyBird/assets/sprites/redbird-midflap.png',
                'FlappyBird/assets/sprites/redbird-downflap.png',
            ),
            # blue bird
            (
                'FlappyBird/assets/sprites/bluebird-upflap.png',
                'FlappyBird/assets/sprites/bluebird-midflap.png',
                'FlappyBird/assets/sprites/bluebird-downflap.png',
            ),
            # yellow bird
            (
                'FlappyBird/assets/sprites/yellowbird-upflap.png',
                'FlappyBird/assets/sprites/yellowbird-midflap.png',
                'FlappyBird/assets/sprites/yellowbird-downflap.png',
            ),
        )

        # list of backgrounds
        BACKGROUNDS_LIST = (
            'FlappyBird/assets/sprites/background-day.png',
            'FlappyBird/assets/sprites/background-night.png',
        )

        # list of pipes
        PIPES_LIST = (
            'FlappyBird/assets/sprites/pipe-green.png',
            'FlappyBird/assets/sprites/pipe-red.png',
        )

        pygame.init()
        self.FPSCLOCK = pygame.time.Clock()
        self.SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT), flags=pygame.HIDDEN)
        pygame.display.set_caption('Flappy Bird')

        # numbers sprites for score display
        self.IMAGES['numbers'] = (
            pygame.image.load('FlappyBird/assets/sprites/0.png').convert_alpha(),
            pygame.image.load('FlappyBird/assets/sprites/1.png').convert_alpha(),
            pygame.image.load('FlappyBird/assets/sprites/2.png').convert_alpha(),
            pygame.image.load('FlappyBird/assets/sprites/3.png').convert_alpha(),
            pygame.image.load('FlappyBird/assets/sprites/4.png').convert_alpha(),
            pygame.image.load('FlappyBird/assets/sprites/5.png').convert_alpha(),
            pygame.image.load('FlappyBird/assets/sprites/6.png').convert_alpha(),
            pygame.image.load('FlappyBird/assets/sprites/7.png').convert_alpha(),
            pygame.image.load('FlappyBird/assets/sprites/8.png').convert_alpha(),
            pygame.image.load('FlappyBird/assets/sprites/9.png').convert_alpha()
        )

        # game over sprite
        self.IMAGES['gameover'] = pygame.image.load('FlappyBird/assets/sprites/gameover.png').convert_alpha()
        # message sprite for welcome screen
        self.IMAGES['message'] = pygame.image.load('FlappyBird/assets/sprites/message.png').convert_alpha()
        # base (ground) sprite
        self.IMAGES['base'] = pygame.image.load('FlappyBird/assets/sprites/base.png').convert_alpha()

        # self.SOUNDS
        if 'win' in sys.platform:
            soundExt = '.wav'
        else:
            soundExt = '.ogg'

        self.SOUNDS['die']    = pygame.mixer.Sound('FlappyBird/assets/audio/die' + soundExt)
        self.SOUNDS['hit']    = pygame.mixer.Sound('FlappyBird/assets/audio/hit' + soundExt)
        self.SOUNDS['point']  = pygame.mixer.Sound('FlappyBird/assets/audio/point' + soundExt)
        self.SOUNDS['swoosh'] = pygame.mixer.Sound('FlappyBird/assets/audio/swoosh' + soundExt)
        self.SOUNDS['wing']   = pygame.mixer.Sound('FlappyBird/assets/audio/wing' + soundExt)

        self.IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[0]).convert()

        self.IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[0][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[0][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[0][2]).convert_alpha()
        )

        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        self.IMAGES['pipe'] = (
            pygame.transform.flip(
                    pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        self.HITMASKS['pipe'] = (
            self.getHitmask(self.IMAGES['pipe'][0]),
            self.getHitmask(self.IMAGES['pipe'][1])
        )

        # hitmask for player
        self.HITMASKS['player'] = (
            self.getHitmask(self.IMAGES['player'][0]),
            self.getHitmask(self.IMAGES['player'][1]),
            self.getHitmask(self.IMAGES['player'][2])
        )

    def init_game(self):
        movementInfo = self.showWelcomeAnimation()
        observation = self.mainGame(movementInfo)

        return observation


    def showWelcomeAnimation(self):
        """Shows welcome screen animation of flappy bird"""
        # index of player to blit on screen
        self.playerIndex = 0
        self.playerIndexGen = cycle([0, 1, 2, 1])
        # iterator used to change playerIndex after every 5th iteration
        self.loopIter = 0

        self.playerx = int(self.SCREENWIDTH * 0.2)
        self.playery = int((self.SCREENHEIGHT - self.IMAGES['player'][0].get_height()) / 2)

        messagex = int((self.SCREENWIDTH - self.IMAGES['message'].get_width()) / 2)
        messagey = int(self.SCREENHEIGHT * 0.12)

        self.basex = 0
        # amount by which base can maximum shift to left
        self.baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # player shm for up-down motion on welcome screen
        playerShmVals = {'val': 0, 'dir': 1}
        
        if self.render:
            if self.sound:
                self.SOUNDS['wing'].play()
            self.SCREEN.blit(self.IMAGES['background'], (0,0))
            self.SCREEN.blit(self.IMAGES['player'][self.playerIndex],
                        (self.playerx, self.playery + playerShmVals['val']))
            self.SCREEN.blit(self.IMAGES['message'], (messagex, messagey))
            self.SCREEN.blit(self.IMAGES['base'], (self.basex, self.BASEY))

            pygame.display.update()

        self.FPSCLOCK.tick(self.FPS)

        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(self.SCREENWIDTH * 0.2)

        self.baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # get 2 new pipes to add to upperPipes lowerPipes list
        newPipe1 = self.getRandomPipe()
        newPipe2 = self.getRandomPipe()

        # list of upper pipes
        self.upperPipes = [
            {'x': self.SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
            {'x': self.SCREENWIDTH + 200 + (self.SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]

        # list of lowerpipe
        self.lowerPipes = [
            {'x': self.SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
            {'x': self.SCREENWIDTH + 200 + (self.SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        self.pipeVelX = -4

        # player velocity, max velocity, downward accleration, accleration on flap
        self.playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerRot     =  45   # player's rotation
        self.playerVelRot  =   3   # angular speed
        self.playerRotThr  =  20   # rotation threshold
        self.playerFlapAcc =  -9   # players speed on flapping
        self.playerFlapped = False # True when player flaps
        
        self.game_state = {
                'basex': self.basex,
                'playerIndexGen': self.playerIndexGen,
                'playerIndex': self.playerIndex,
                'loopIter': self.loopIter,
                'score': self.score,
                'upperPipes': self.upperPipes,
                'lowerPipes': self.lowerPipes,
                'playerx': self.playerx,
                'playery': self.playery,
                'playerVelY': self.playerVelY,
                'playerRot': self.playerRot,
                'playerFlapped': self.playerFlapped,
                'dead': False
            }

        if self.image_obs:
            return self.image_observation()
        else:
            return self.observation


    def game_step(self, action):
        reward = 0
        info = {}
        if action == 1:
            #reward -= 0.25
            if self.game_state['playery'] > -2 * self.IMAGES['player'][0].get_height():
                self.game_state['playerVelY'] = self.playerFlapAcc
                self.game_state['playerFlapped'] = True
                if self.render and self.sound:
                    self.SOUNDS['wing'].play()
        #else:
        #    reward += 0.5
        # check for crash here
        crashTest = self.checkCrash({'x': self.game_state['playerx'], 'y': self.game_state['playery'], 'index': self.game_state['playerIndex']},
        self.game_state['upperPipes'], self.game_state['lowerPipes'])

        if crashTest[0]:
            reward -= 1
            self.showGameOverScreen()
            info['score'] = self.score
            info['reward'] = reward
            info['dead'] = True
            if self.image_obs:
                return self.image_observation(), reward, info
            return self.game_state(), reward, info

        # check for score
        playerMidPos = self.game_state['playerx'] + self.IMAGES['player'][0].get_width() / 2
        for pipe in self.game_state['upperPipes']:
            pipeMidPos = pipe['x'] + self.IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                reward += 2 #+ self.score #25 * self.game_state['score']
                if self.render and self.sound:
                    self.SOUNDS['point'].play()

        # playerIndex basex change
        if (self.game_state['loopIter'] + 1) % 3 == 0:
            self.game_state['playerIndex'] = next(self.game_state['playerIndexGen'])
        self.game_state['loopIter'] = (self.game_state['loopIter'] + 1) % 30
        self.game_state['basex'] = -((-self.game_state['basex'] + 100) % self.baseShift)

        # rotate the player
        if self.game_state['playerRot'] > -90:
            self.game_state['playerRot'] -= self.playerVelRot

        # player's movement
        if self.game_state['playerVelY'] < self.playerMaxVelY and not self.game_state['playerFlapped']:
            self.game_state['playerVelY'] += self.playerAccY
        if self.game_state['playerFlapped']:
            self.game_state['playerFlapped'] = False

            # more rotation to cover the threshold (calculated in visible rotation)
            self.game_state['playerRot'] = 45

        playerHeight = self.IMAGES['player'][self.game_state['playerIndex']].get_height()
        self.game_state['playery'] += min(self.game_state['playerVelY'], self.BASEY - self.game_state['playery'] - playerHeight)


        # move pipes to left
        for uPipe, lPipe in zip(self.game_state['upperPipes'], self.game_state['lowerPipes']):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if len(self.game_state['upperPipes']) > 0 and 0 < self.game_state['upperPipes'][0]['x'] < 5:
            newPipe = self.getRandomPipe()
            self.game_state['upperPipes'].append(newPipe[0])
            self.game_state['lowerPipes'].append(newPipe[1])

        # remove first pipe if its out of the screen
        if len(self.game_state['upperPipes']) > 0 and self.game_state['upperPipes'][0]['x'] < -self.IMAGES['pipe'][0].get_width():
            self.game_state['upperPipes'].pop(0)
            self.game_state['lowerPipes'].pop(0)

        #if self.game_state['upperPipes'][0]['x'] * self.game_state['upperPipes'][0]['y'] > 0:
        #dist_pipes = 50 / ((self.SCREENHEIGHT + self.game_state['upperPipes'][0]['y']) - self.game_state['playery'])
        #print("playery", self.game_state['playery'])
        
        if self.game_state['upperPipes'][0]['x'] > self.game_state['playerx']:
            dist = 0.75 * (100000 / ((50 + math.sqrt(((self.game_state['upperPipes'][0]['x'] + self.game_state['lowerPipes'][0]['x']) - self.game_state['playerx'])**2 + ((self.game_state['upperPipes'][0]['y'] + self.game_state['lowerPipes'][0]['y']) - self.game_state['playery'])**2))**2))
            dist += 0.25 * (100000 / ((50 + math.sqrt(((self.game_state['upperPipes'][1]['x'] + self.game_state['lowerPipes'][1]['x']) - self.game_state['playerx'])**2 + ((self.game_state['upperPipes'][1]['y'] + self.game_state['lowerPipes'][1]['y']) - self.game_state['playery'])**2))**2))
        else: 
            dist = 2 * (100000 / ((50 + math.sqrt(((self.game_state['upperPipes'][1]['x'] + self.game_state['lowerPipes'][1]['x']) - self.game_state['playerx'])**2 + ((self.game_state['upperPipes'][1]['y'] + self.game_state['lowerPipes'][1]['y']) - self.game_state['playery'])**2))**2))
            #print("pipe behind", reward)
        if dist > 1: 
            dist == 1
        #reward += dist
        #print(reward)
        #    dist_pipes = 10
        #reward += -2 * dist_pipes
        
        if self.render:
            # draw sprites
            self.SCREEN.blit(self.IMAGES['background'], (0,0))

            for uPipe, lPipe in zip(self.game_state['upperPipes'], self.game_state['lowerPipes']):
                self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

            self.SCREEN.blit(self.IMAGES['base'], (self.game_state['basex'], self.BASEY))
            # print score so player overlaps the score
            self.showScore(self.score)

        # Player rotation has a threshold
        visibleRot = self.playerRotThr
        if self.game_state['playerRot'] <= self.playerRotThr:
            visibleRot = self.game_state['playerRot']
        
        playerSurface = pygame.transform.rotate(self.IMAGES['player'][self.game_state['playerIndex']], visibleRot)
        if self.render:
            self.SCREEN.blit(playerSurface, (self.game_state['playerx'], self.game_state['playery']))
            pygame.display.update()
            
        self.FPSCLOCK.tick(self.FPS)

        info['score'] = self.score
        info['reward'] = reward
        info['dead'] = False

        pygame.event.pump()

        if self.image_obs:
            return self.image_observation(), reward, info

        return self.game_state, reward, info

        


    def showGameOverScreen(self):
        """crashes the player down ans shows gameover image"""
        score = self.score
        playerx = self.SCREENWIDTH * 0.2
        playery = self.game_state['playery']
        playerHeight = self.IMAGES['player'][0].get_height()
        playerVelY = self.game_state['playerVelY']
        playerAccY = 2
        playerRot = self.game_state['playerRot']
        playerVelRot = 7

        basex = self.game_state['basex']

        upperPipes, lowerPipes = self.game_state['upperPipes'], self.game_state['lowerPipes']

        # play hit and die self.SOUNDS
        if self.render and self.sound:
            self.SOUNDS['hit'].play()
        #if not self.game_state['groundCrash']:
        #    self.SOUNDS['die'].play()
        """
        while True:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                    if playery + playerHeight >= self.BASEY - 1:
                        return
        """
        # player y shift
        if playery + playerHeight < self.BASEY - 1:
            playery += min(playerVelY, self.BASEY - playery - playerHeight)

        # player velocity change
        if playerVelY < 15:
            playerVelY += playerAccY

        # rotate only when it's a pipe crash
        #if not self.game_state['groundCrash']:
        #    if playerRot > -90:
        #        playerRot -= playerVelRot

        # draw sprites
        self.SCREEN.blit(self.IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        self.SCREEN.blit(self.IMAGES['base'], (basex, self.BASEY))
        self.showScore(score)

        playerSurface = pygame.transform.rotate(self.IMAGES['player'][1], playerRot)
        self.SCREEN.blit(playerSurface, (playerx,playery))
        self.SCREEN.blit(self.IMAGES['gameover'], (50, 180))

        self.FPSCLOCK.tick(self.FPS)
        #pygame.display.update()
        
        

    def playerShm(self, playerShm):
        """oscillates the value of playerShm['val'] between 8 and -8"""
        if abs(playerShm['val']) == 8:
            playerShm['dir'] *= -1

        if playerShm['dir'] == 1:
            playerShm['val'] += 1
        else:
            playerShm['val'] -= 1


    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapY = random.randrange(0, int(self.BASEY * 0.6 - self.PIPEGAPSIZE))
        gapY += int(self.BASEY * 0.2)
        pipeHeight = self.IMAGES['pipe'][0].get_height()
        pipeX = self.SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
            {'x': pipeX, 'y': gapY + self.PIPEGAPSIZE}, # lower pipe
        ]


    def showScore(self, score):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0 # total width of all numbers to be printed

        for digit in scoreDigits:
            totalWidth += self.IMAGES['numbers'][digit].get_width()

        Xoffset = (self.SCREENWIDTH - totalWidth) / 2

        for digit in scoreDigits:
            self.SCREEN.blit(self.IMAGES['numbers'][digit], (Xoffset, self.SCREENHEIGHT * 0.1))
            Xoffset += self.IMAGES['numbers'][digit].get_width()


    def checkCrash(self, player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = self.IMAGES['player'][0].get_width()
        player['h'] = self.IMAGES['player'][0].get_height()

        # if player crashes into ground
        if player['y'] + player['h'] >= self.BASEY - 1:
            return [True, True]
        elif player['y'] + player['h'] <= player['h'] + 1:
            return [True, True]
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                        player['w'], player['h'])
            pipeW = self.IMAGES['pipe'][0].get_width()
            pipeH = self.IMAGES['pipe'][0].get_height()

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

                # player and upper/lower pipe self.HITMASKS
                pHitMask = self.HITMASKS['player'][pi]
                uHitmask = self.HITMASKS['pipe'][0]
                lHitmask = self.HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return [True, False]

        return [False, False]

    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in xrange(rect.width):
            for y in xrange(rect.height):
                if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                    return True
        return False

    def getHitmask(self, image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in xrange(image.get_width()):
            mask.append([])
            for y in xrange(image.get_height()):
                mask[x].append(bool(image.get_at((x,y))[3]))
        return mask

    def enable_rendering(self):
        self.render = True
        self.SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))

    def image_observation(self):
        return pygame.surfarray.array3d(self.SCREEN) #/ 255.0

    def __getstate__(self):
        data = self.__dict__
        del data['SCREEN']
        del data['FPSCLOCK']
        del data['IMAGES']
        del data['SOUNDS']
        del data['HITMASKS']
        print([type(attr) for attr in data.values()])
        return data

