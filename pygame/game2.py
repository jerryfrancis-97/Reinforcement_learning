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

#colors
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
SPEED = 5


class Enemy(pygame.sprite.Sprite):
      def __init__(self):
        super().__init__() 
        self.enemy_pic_loc = "game2_enemy.png"
        self.image = pygame.image.load(self.enemy_pic_loc)
        self.surf = pygame.Surface((50, 80))
        self.rect = self.surf.get_rect(center = (random.randint(40, 360)
                                               ,0))     
 
      def move(self):
        self.rect.move_ip(0,10)
        if (self.rect.bottom > 600):
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
E1 = Enemy()
E2 = Enemy()
E3 = Enemy()
E4 = Enemy()


#Creating Sprites Groups
enemies = pygame.sprite.Group()
enemies.add(E1)
enemies.add(E2)
enemies.add(E3)
enemies.add(E4)

all_sprites = pygame.sprite.Group()
all_sprites.add(P1)
all_sprites.add(E1)
all_sprites.add(E2)
all_sprites.add(E3)
all_sprites.add(E4)
 
#Adding a new User event 
INC_SPEED = pygame.USEREVENT + 1
pygame.time.set_timer(INC_SPEED, 10)


while True:     
    for event in pygame.event.get(): 
        if event.type == INC_SPEED:
            SPEED +=2

        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    
    surface.fill(WHITE)

    #draw all objects
    for entity in all_sprites:
        surface.blit(entity.image,entity.rect)
        entity.move()
    
    #collision
    if pygame.sprite.spritecollideany(P1,enemies):
        surface.fill(RED)
        pygame.display.update()
        for entity in all_sprites:
            entity.kill()
        time.sleep(2)
        pygame.quit()
        sys.exit()
        
         
    pygame.display.update()
    FPS.tick(60)
