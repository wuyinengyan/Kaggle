import pygame
import sys
from flight_sprites import *
from pygame.locals import *


class GameMain:
    """ 主游戏类 """

    def __init__(self):

        self.screen = pygame.display.set_mode(BG_SIZE)
        # 创建时钟对象
        self.clock = pygame.time.Clock()
        # 调用私有方法，精灵和精灵组的创建
        self.__create_sprites()
        # 创建敌机出现的定时器事件  1000ms
        pygame.time.set_timer(CREATE_ENEMY_EVENT, 1000)
        pygame.time.set_timer(FIRE_EVENT, 333)
        pygame.time.set_timer(SWITCH_ME, 100)

    def __create_sprites(self):
        # 创建背景
        bg1 = BackGround()
        bg2 = BackGround(True)
        self.back_group = pygame.sprite.Group(bg1, bg2)
        # 创建敌机
        self.enemy_group = pygame.sprite.Group()
        # 创建我方飞机
        self.me = Me()
        self.me_group = pygame.sprite.Group(self.me)

    def __check_collide(self):
        # 子弹摧毁敌机：两个精灵组中的所有精灵的碰撞检测 -> Sprite_dict
        pygame.sprite.groupcollide(self.me.bullets, self.enemy_group, True, True)
        # 敌机摧毁英雄：某个精灵和制定精灵组中的碰撞检测 -> Sprite_list
        enemies = pygame.sprite.spritecollide(self.me, self.enemy_group, True)
        if len(enemies) > 0:
            self.me.kill()
            GameMain.__game_over()

    def __event_handler(self):
        for event in pygame.event.get():  # 获取所有event事件
            if event.type == pygame.QUIT:
                GameMain.__game_over()
            elif event.type == CREATE_ENEMY_EVENT:  # 定时器事件
                enemy = Enemy()
                self.enemy_group.add(enemy)
            elif event.type == KEYUP:  # 松开按键，我方飞机停止移动
                self.me.speed = [0, 0]
            elif event.type == FIRE_EVENT:  # 发射事件
                self.me.fire()
            elif event.type == SWITCH_ME:  # 切换我方飞机图像
                self.me.switch_me()

            # 用户按住方向键不放，就实现持续移动，操作灵活性好
            keys_pressed = pygame.key.get_pressed()
            if keys_pressed[pygame.K_UP]:
                self.me.speed = [0, -ME_SPEED]
            elif keys_pressed[pygame.K_RIGHT]:
                self.me.speed = [ME_SPEED, 0]
            elif keys_pressed[pygame.K_DOWN]:
                self.me.speed = [0, ME_SPEED]
            elif keys_pressed[pygame.K_LEFT]:
                self.me.speed = [-ME_SPEED, 0]
            # 用户必须要抬起按键才算是一次事件，操作灵活性不好
            # elif event.type == KEYDOWN:  # 上右下左控制飞机的移动
            #     if event.key == K_UP:
            #         GameMain.gl_speed = [0, -5]

    def __update_sprites(self):
        # me0 = ME1 if me.switch_image else ME2  # 更新我方飞机图片

        self.back_group.update()  # 让组中所有的精灵更新位置
        self.back_group.draw(self.screen)  # 在screen上绘制所有精灵
        self.enemy_group.update()
        self.enemy_group.draw(self.screen)
        self.me_group.update()
        self.me_group.draw(self.screen)
        self.me.bullets.update()
        self.me.bullets.draw(self.screen)

    @staticmethod
    def __game_over():
        pygame.quit()
        sys.exit()  # 当前执行的程序，直接退出系统，更高级的return

    def start_game(self):
        # me = Me()
        while True:
            self.__event_handler()  # 事件监听
            self.__update_sprites()  # 更新绘制精灵组
            self.__check_collide()  # 碰撞检测
            self.clock.tick(FRAME_PER_SEC)
            pygame.display.update()  # 更新显示


if __name__ == '__main__':
    game = GameMain()
    game.start_game()
