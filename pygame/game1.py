# import the pygame module, so you can use it
import pygame
 
# define a main function
def main():
     
    # initialize the pygame module
    pygame.init()
    # load and set the image
    image = pygame.image.load("smiley.png")
    image.set_colorkey((200,200,100)) #to remove edges, merge with bg
    # create a surface on screen
    screen = pygame.display.set_mode((600,400))
    screen.fill((200,200,100))  #bgcolor
    screen.blit(image,(60,60))
    pygame.display.flip()   #update
    
    #define bground
    bg_image = pygame.image.load("st.-peters-basilica-and-dome-tour.jpg")
    # define the position of the smiley
    xpos = 50
    ypos = 50
    # how many pixels we move our smiley each frame
    step_x = 10
    step_y = 10
    
    
    # define a variable to control the main loop
    running = True
     
    # main loop
    while running:
        # event handling, gets all event from the event queue
        
        if xpos>600 or xpos<0:
            step_x = -step_x
        if ypos>400 or ypos<0:
            step_y = -step_y
        # update the position of the smiley
        xpos += step_x # move it to the right
        ypos += step_y # move it down

        # first erase the screen 
        #(just blit the background over anything on screen)
        screen.blit(bg_image, (0,0))
        # now blit the smiley on screen
        screen.blit(image, (xpos, ypos))
        # and update the screen (don't forget that!)
        pygame.display.flip()
        

        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False
     
     
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()