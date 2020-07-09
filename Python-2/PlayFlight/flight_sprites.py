import random
import pygame

# 定义常量
BG_SIZE = BG_WIDTH, BG_HEIGHT = 480, 700  # 窗口大小 == 背景图像大小
FRAME_PER_SEC = 60  # 游戏帧率设置
CREATE_ENEMY_EVENT = pygame.USEREVENT  # 创建敌机的定时器
FIRE_EVENT = pygame.USEREVENT + 1  # 创建敌机的定时器
SWITCH_ME = pygame.USEREVENT + 2  # 切换英雄图像
ME_SPEED = 4  # 操作我方飞机的移动速度
# 加载素材
ME1 = pygame.image.load("./Resource/images/me1.png")
ME2 = pygame.image.load("./Resource/images/me2.png")


class ImageSprite(pygame.sprite.Sprite):
    """ 飞机类 """

    def __init__(self, image_name, speed=1):
        # 调用父类的初始化方法
        super().__init__()

        # 定义对象的属性
        self.image = pygame.image.load(image_name)
        self.rect = self.image.get_rect()
        self.speed = speed

    def update(self):
        self.rect.y += self.speed


class Me(ImageSprite):
    """ 我方飞机类 """

    def __init__(self):
        super().__init__("./Resource/images/me1.png")

        self.rect.centerx = BG_WIDTH / 2
        self.rect.bottom = BG_HEIGHT - 100
        self.speed = [0, 0]
        # 创建子弹精灵组
        self.bullets = pygame.sprite.Group()

        # 我方飞机移动的限制范围：上右下左
        self.__lim_area = [0, BG_WIDTH - self.rect.width, BG_HEIGHT - self.rect.height, 0]
        # 切换我方飞机图像
        self.__switch_flag = False
        self.__image1 = pygame.image.load("./Resource/images/me1.png")
        self.__image2 = pygame.image.load("./Resource/images/me2.png")

    def update(self):
        self.rect.x += self.speed[0]
        self.rect.y += self.speed[1]
        # 限制我方飞机水平的移动范围
        self.rect.y = self.__lim_area[0] if self.rect.y < self.__lim_area[0] else self.rect.y  # 上
        self.rect.x = self.__lim_area[1] if self.rect.x > self.__lim_area[1] else self.rect.x  # 右
        self.rect.y = self.__lim_area[2] if self.rect.y > self.__lim_area[2] else self.rect.y  # 下
        self.rect.x = self.__lim_area[3] if self.rect.x < self.__lim_area[3] else self.rect.x  # 左


    def fire(self):
        for i in range(3):  # 一次性发射３枚子弹
            bullet = Bullet()
            bullet.rect.bottom = self.rect.y - i * 20
            bullet.rect.centerx = self.rect.centerx
            self.bullets.add(bullet)

    def switch_me(self):
        self.image = self.__image1 if self.__switch_flag else self.__image2
        self.__switch_flag = not self.__switch_flag


class BackGround(ImageSprite):
    """ 背景图 """

    def __init__(self, is_alt=False):
        # 调用父类的初始化方法，并设置默认图
        super().__init__("./Resource/images/background.png")

        # 判断是否为交替图像
        if is_alt:
            self.rect.y = -self.rect.height

    def update(self):
        super().update()
        if self.rect.y >= BG_HEIGHT:
            self.rect.y = -self.rect.height


class Enemy(ImageSprite):
    """　敌方飞机　"""

    def __init__(self):
        super().__init__("./Resource/images/enemy1.png")
        self.speed = random.randint(2, 4)

        self.rect.x = random.randint(0, BG_WIDTH - self.rect.width)
        self.rect.bottom = 0  # bottom = y + height

    def update(self):
        super().update()
        if self.rect.y >= BG_HEIGHT:
            self.kill()  # 从精灵组中移除，并从内存中销毁


class Bullet(ImageSprite):
    """ 子弹类 """

    def __init__(self):
        super().__init__("./Resource/images/bullet2.png", -6)

    def update(self):
        super().update()

        if self.rect.bottom < 0:  # 子弹向上飞出屏幕
            self.kill()  # 从精灵组中移除，并从内存中销毁
