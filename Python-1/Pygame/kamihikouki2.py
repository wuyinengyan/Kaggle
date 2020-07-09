import pygame
import sys
from pygame.locals import *

# 初始化Pygame
pygame.init()

try:
    screen_list = pygame.display.list_modes()  # 获取支持的屏幕分辨率列表
    size = width, height = 800, 600
    speed = [0, 5]

    fullscreen = False
    # 创建指定大小的窗口 Surface
    screen = pygame.display.set_mode(size)
    # 创建窗口标题
    pygame.display.set_caption("纸飞机")

    # 加载图片
    bg_image = pygame.image.load("BlueSky.jpg").convert_alpha()
    kamihikouki = pygame.image.load("kamihikouki.png")
    kamihikouki = pygame.transform.rotate(kamihikouki, 270)  # 初始转向
    
    fly_left = kamihikouki  # pygame.transform.rotate逆时针旋转图像
    fly_bottom = pygame.transform.rotate(kamihikouki, 90)
    fly_right = pygame.transform.rotate(kamihikouki, 180)
    fly_top = pygame.transform.rotate(kamihikouki, 270)

    # 获取图片的位置矩形
    position = kamihikouki.get_rect()  # <rect(0, 0, 120, 120)>

    while True:
        for event in pygame.event.get():  # 获取event的所有事件
            if event.type == pygame.QUIT:
                sys.exit()   

        # 移动图像
        position = position.move(speed)
        
        if position.bottom > height:
            kamihikouki = fly_bottom
            position = kamihikouki.get_rect()
            position.top = height - position.height
            speed = [5, 0]
            
        if position.right > width:
            kamihikouki = fly_right
            position = kamihikouki.get_rect()
            position.left = width - position.width
            position.top = height - position.height
            speed = [0, -5]
            
        if position.top < 0:
            kamihikouki = fly_top
            position = kamihikouki.get_rect()
            position.left = width - position.width
            speed = [-5, 0]
            
        if position.left < 0:
            kamihikouki = fly_left
            position = kamihikouki.get_rect()
            speed = [0, 5]
            
        # 填充背景
        screen.blit(bg_image, (0, 0))
        # 更新图像
        screen.blit(kamihikouki, position)
        # 更新界面
        pygame.display.flip()
        # 延迟10毫秒
        pygame.time.delay(8)

except Exception as e:
    with open("log.txt", "a") as f:
        f.write(str(e))