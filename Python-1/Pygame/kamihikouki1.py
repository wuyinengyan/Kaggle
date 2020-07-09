import pygame
import sys
from pygame.locals import *
import random

# 初始化Pygame
pygame.init()

try:
    screen_list = pygame.display.list_modes()  # 获取支持的屏幕分辨率列表
    size = width, height = 800, 600
    speed = [2, -1]
    bg = (255, 255, 255)

    #设置缩放的比率
    ratio = 1
    fullscreen = False
    # 创建指定大小的窗口 Surface
    screen = pygame.display.set_mode(size, RESIZABLE)  # 窗口可拖拽
    # 创建窗口标题
    pygame.display.set_caption("纸飞机")

    # 加载图片
    bg_image = pygame.image.load("BlueSky.jpg").convert_alpha()
    okamihikouki = pygame.image.load("kamihikouki.png")
    kamihikouki = okamihikouki
    # 获取图片的位置矩形
    # position = kamihikouki.get_rect()  # <rect(0, 0, 120, 120)>
    init_x, init_y = random.randint(0, width-120), random.randint(0, height-120)
    oposition = pygame.Rect(init_x, init_y, 120, 120)  # 后面两个120，分别代表宽和高
    position = oposition
    
    l_head = kamihikouki
    r_head = pygame.transform.flip(kamihikouki, True, False)

    while True:
        for event in pygame.event.get():  # 获取event的所有事件
            if event.type == pygame.QUIT:
                sys.exit() 
                
            if event.type == KEYDOWN:
                if event.key == K_LEFT:  # 275
                    kamihikouki = r_head
                    speed = [-1, 0]
                if event.key == K_RIGHT:  # 276
                    kamihikouki = l_head
                    speed = [1, 0]
                if event.key == K_UP:  # 273
                    speed = [0, -1]
                if event.key == K_DOWN:  # 274
                    speed = [0, 1]

                # 全屏(F11)
                if event.key == K_F11:
                    fullscreen = not fullscreen
                    width = screen_list[0][0] if fullscreen else 800
                    height = screen_list[0][1] if fullscreen else 600
                    # 初始化位置
                    init_x, init_y = random.randint(0, width-120), random.randint(0, height-120)
                    position = pygame.Rect(init_x, init_y, 120, 120)
                    if fullscreen:  # HWSUFACE硬件加速
                        screen = pygame.display.set_mode(screen_list[0], FULLSCREEN | HWSURFACE)
                    else:
                        screen = pygame.display.set_mode(size)
            
                # +放大、-缩小图像、空格恢复初始
                if event.key == K_EQUALS or event.key == K_MINUS or event.key == K_SPACE:
                    if event.key == K_EQUALS and ratio < 2:  # 最多放大一倍
                        ratio += 0.1
                    if event.key == K_MINUS and ratio > 0.5:  # 最多缩小一半
                        ratio -= 0.1
                    if event.key == K_SPACE:
                        ratio = 1
                        speed = [2, -1]
                
                    kamihikouki = pygame.transform.smoothscale(okamihikouki, (int(oposition.width * ratio), int(oposition.height * ratio)))
                    l_head = kamihikouki
                    r_head = pygame.transform.flip(kamihikouki, True, False)

                    
                
            # 拖拽窗口，改变大小
            if event.type == VIDEORESIZE:
                size = event.size
                width, height = size
                screen = pygame.display.set_mode(size, RESIZABLE)       
                # 初始化位置
                init_x, init_y = random.randint(0, width-120), random.randint(0, height-120)
                position = pygame.Rect(init_x, init_y, 120, 120)

        # 移动图像
        position = position.move(speed)
        if position.left < 0 or (position.right > width):
            # 翻转图像
            kamihikouki = pygame.transform.flip(kamihikouki, True, False)
            # 反方向移动
            speed[0] = -speed[0]
        if position.top < 0 or (position.bottom > height):
            speed[1] = -speed[1]

        # 填充背景
        screen.fill(bg)
        # 更新图像
        # screen.blit(bg_image, (0, 0))
        screen.blit(kamihikouki, position)
        # 更新界面
        pygame.display.flip()
        # 延迟10毫秒
        pygame.time.delay(8)

except Exception as e:
    with open("log.txt", "a") as f:
        f.write(str(e))