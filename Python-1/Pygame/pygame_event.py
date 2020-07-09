import pygame
import sys

# 初始化Pygame
pygame.init()

size = width, height = 800, 600
screen = pygame.display.set_mode(size)
speed = [-2, 1]
bg = (0, 0, 0)

position = 0
font = pygame.font.Font(None, 20)
line_height = font.get_linesize()

screen.fill(bg)

while True:
    for event in pygame.event.get():  # 获取event的所有事件
        if event.type == pygame.QUIT:
            sys.exit()  # UserWarning: To exit: use 'exit', 'quit', or Ctrl-D. 
            
        screen.blit(font.render(str(event), True, (0, 255, 0)), (0, position))
        position += line_height  # 换下一行
        
        # 如果，位置超过屏幕高度，清屏
        if position > height:
            position = 0
            screen.fill(bg)
    
    pygame.display.flip()