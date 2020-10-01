import pygame,sys
from pygame.locals import *
import random,os
import time

pygame.init()

'''
    pygame.draw.polygon(surface, color, pointlist, width)
    pygame.draw.line(surface, color, start_point, end_point, width)
    pygame.draw.lines(surface, color, closed, pointlist, width)
    pygame.draw.circle(surface, color, center_point, radius, width)
    pygame.draw.ellipse(surface, color, bounding_rectangle, width)
    pygame.draw.rect(surface, color, rectangle_tuple, width)

    red = pygame.Color(255,0,0)
    pygame.draw.line(surface,red,(10,60),(250,60),5)
    green = pygame.Color(0,255,0)
    pygame.draw.line(surface,green,(10,200),(250,200),5)
'''
#to set the fps of frames
FPS = pygame.time.Clock()

surface = pygame.display.set_mode((400,600))
surface.fill((255,255,255)) #white
pygame.display.set_caption("GAME_STARTED")
#street background
back_street = pygame.image.load("game2_Street.png")

#colors
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
SPEED = 5
SCORE = 0
RUSH_SCORE = 0
 
#Setting up Fonts
font = pygame.font.SysFont("Verdana", 60)
font_small = pygame.font.SysFont("Verdana", 20)
game_over = font.render("Game Over", True, BLACK)

class Enemy(pygame.sprite.Sprite):
      def __init__(self, step_size):
        super().__init__() 
        self.enemy_pic_loc = "game2_enemy.png"
        self.image = pygame.image.load(self.enemy_pic_loc)
        self.surf = pygame.Surface((50, 80))
        self.rect = self.surf.get_rect(center = (random.randint(40, 360)
                                               ,0))     
        self.inc_step = step_size
      
      def rush_hell(self,rush_step):
          self.inc_step = rush_step
    
      def move(self):
        global SCORE
        # self.rect.move_ip(0,10)
        self.rect.move_ip(0,self.inc_step)
        if (self.rect.bottom > 600):
            SCORE+=1
            self.rect.top = 0
            self.rect.center = (random.randint(30, 370), 0)
 
      def draw(self, surface):
        surface.blit(self.image, self.rect) 

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.player_pic_loc = "game2_player.png"
        self.image = pygame.image.load(self.player_pic_loc)
        self.surf = pygame.Surface((50, 100))
        self.rect = self.surf.get_rect(center = (200,550))
 
    def move(self):
        pressed_keys = pygame.key.get_pressed()
       #if pressed_keys[K_UP]:
            #self.rect.move_ip(0, -5)
       #if pressed_keys[K_DOWN]:
            #self.rect.move_ip(0,5)
         
        if self.rect.left > 0:
              if pressed_keys[K_LEFT]:
                  self.rect.move_ip(-5, 0)
        if self.rect.right < SCREEN_WIDTH:   #width        
              if pressed_keys[K_RIGHT]:
                  self.rect.move_ip(5, 0)
 
    def draw(self, surface):
        surface.blit(self.image, self.rect)     
 
         
P1 = Player()
E1 = Enemy(10)  #taking step_size  as 10 for start


#Creating Sprites Groups
enemies = pygame.sprite.Group()
enemies.add(E1)

all_sprites = pygame.sprite.Group()
all_sprites.add(P1)
all_sprites.add(E1)
 
#Adding a new User event 
'''
1.) pygame.time.set_timer(INC_SPEED, 1000)
    pygame.time.set_timer(ADD_ENEMY, 10000)
    This is a good gameplay setting

'''
INC_SPEED = pygame.USEREVENT + 1
pygame.time.set_timer(INC_SPEED, 1000)  #5000ms = 5sec

ADD_ENEMY = pygame.USEREVENT + 1
pygame.time.set_timer(ADD_ENEMY, 10000)

enemy_step_size_counter = 0

while True:     
    for event in pygame.event.get(): 
        if event.type == INC_SPEED:
            SPEED +=1
        if event.type == ADD_ENEMY:
            if enemy_step_size_counter==0:
                temp_enemy = Enemy(2)
                all_sprites.add(temp_enemy)
                enemies.add(temp_enemy)
                enemy_step_size_counter+=3
            else:
                temp_enemy = Enemy(enemy_step_size_counter)
                all_sprites.add(temp_enemy)
                enemies.add(temp_enemy)
                enemy_step_size_counter+=6

        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    
    surface.blit(back_street,(0,0))
    scores = font_small.render("SCORE : "+str(SCORE), True, BLACK)
    surface.blit(scores, (10,10))

    #draw all objects
    for entity in all_sprites:
        surface.blit(entity.image,entity.rect)
        entity.move()
    
    #enemy_collision  - enemy collides with other other enemy, speed of collided enemy increases to 8
    for enemy in pygame.sprite.groupcollide(enemies,enemies,0,0).keys():
        RUSH_SCORE+=1
        rush_print = font_small.render("RUSH: "+str(RUSH_SCORE), True, BLACK)
        surface.blit(rush_print,(250,10))
        enemy.rush_hell(8)  #step_szie to 8


    
    
    #collision
    if pygame.sprite.spritecollideany(P1,enemies):
        time.sleep(1)
        surface.fill(RED)
        surface.blit(game_over,(50,50))
        FINAL_SCORE = font_small.render("Your SCOre: "+str(SCORE), True, BLACK)
        surface.blit(FINAL_SCORE,(50,200))
        pygame.display.update()
        for entity in all_sprites:
            entity.kill()
        time.sleep(3)
        pygame.quit()
        sys.exit()
        
         
    pygame.display.update()
    FPS.tick(60)
