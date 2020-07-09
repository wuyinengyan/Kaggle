import pygame
import sys
from pygame.locals import *

# 初始化Pygame
pygame.init()

try:
    size = width, height = 800, 600
    bg = (255, 255, 255)

    clock = pygame.time.Clock()  # 设置游戏的帧率
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("图片剪切")  # 设置标题
    target_image = pygame.image.load("BlueSky.jpg")  # 目标图片   
    select = 0  # 0未选择；1选择中；2完成选择
    drag = 0  # 0未拖拽；1拖拽中；2完成拖拽
    select_rect = pygame.Rect(0, 0, 0, 0)
    position = target_image.get_rect()  # <rect(0, 0, 120, 120)> # 获取图片的位置矩形
    position.center = width // 2, height // 2

    while True:
        for event in pygame.event.get():  # 获取event的所有事件
            if event.type == pygame.QUIT:
                sys.exit() 
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    # 第一次点击，选择范围
                    if select == 0 and drag == 0:
                        pos_start = event.pos  # 位置起始点
                        select = 1
                    # 第二次点击，拖拽图像
                    elif select == 2 and drag == 0:
                        capture = screen.subsurface(select_rect).copy()  # ????
                        cap_rect = capture.get_rect()  # 获取裁剪的图像
                        drag = 1
                    # 第三次点击，初始化
                    elif select == 2 and drag == 2:                    
                        select = drag = 0
                        
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    # 第一次释放，结束选择
                    if select == 1 and drag == 0:
                        pos_stop = event.pos  # 位置结束点
                        select = 2
                    # 第二次释放，结束拖拽
                    elif select == 2 and drag == 1:# 获取裁剪的图像
                        drag = 2
            
        screen.fill(bg)
        screen.blit(target_image, position)
                
        if select:
            mouse_pos = pygame.mouse.get_pos()  # 获取鼠标的实时位置
            if select == 1:
                pos_stop = mouse_pos  # 鼠标松开的地方即为停止点
            select_rect.left, select_rect.top = pos_start
            select_rect.width, select_rect.height = pos_stop[0] - pos_start[0], pos_stop[1] - pos_start[1]
            pygame.draw.rect(screen, (0, 0, 0), select_rect, 1)  # 绘制图像边框，(0, 0, 0)是黑色，1是边框宽度
            # rect(Surface, color, Rect, width=0)  # width=0是填充，且向外扩展
            
        if drag:
            if drag == 1:
                cap_rect.center = mouse_pos  # 拖拽图像中，鼠标处于中心
            screen.blit(capture, cap_rect)
            
        pygame.display.flip()  # 更新界面
        clock.tick(30)  # 设置游戏的帧率为30，降低帧率，不会占满CPU
    
except Exception as e:
    with open("log.txt", "a") as f:
        f.write(str(e))