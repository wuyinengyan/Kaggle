import pygame
import sys
from pygame.locals import *

# 初始化Pygame
pygame.init()

try:
    size = width, height = 800, 600
    bg = (0, 0, 0)

    clock = pygame.time.Clock()  # 设置游戏的帧率
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("图片剪切")  # 设置标题
    
    kamihikouki = pygame.image.load("kamihikouki.png").convert_alpha()  # convert不会加载透明度，所以对png无效
    bg_image = pygame.image.load("BlueSky.jpg").convert()   
    position = kamihikouki.get_rect()
    position.center = width // 2, height // 2
    
    # kamihikouki.set_colorkey((255, 255, 255))  # 虚化透明，效果不好
    # kamihikouki.set_alpha(200)  # 整个图像都透明

    # 效果依然很差
    # for i in range(position.width):
    #     for j in range(position.height):
    #         temp = kamihikouki.get_at((i, j))
    #         if temp[3] != 0:  # 获取RGBA第4个值，配合convert_alpha使用
    #             temp[3] = 100
    #         kamihikouki.set_at((i, j), temp)
    
    # 这部分代码讲解的很差，难懂
    def blit_alpha(target, source, location, opacity):
        x = location[0]
        y = location[1]
        temp = pygame.Surface((source.get_width(), source.get_height())).convert()
        temp.blit(target, (-x, -y))
        temp.blit(source, (0, 0))
        temp.set_alpha(opacity)
        target.blit(temp, location)
    
    while True:
        for event in pygame.event.get():  # 获取event的所有事件
            if event.type == pygame.QUIT:
                sys.exit() 

        
        screen.blit(bg_image, (0, 0))  # 填充背景     
        # screen.blit(kamihikouki, position)  # 更新图像
        blit_alpha(screen, kamihikouki, position, 200)
           
        pygame.display.flip()  # 更新界面
        clock.tick(30)  # 设置游戏的帧率为30
    
except Exception as e:
    with open("log.txt", "a") as f:
        f.write(str(e))