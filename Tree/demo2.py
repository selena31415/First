import turtle

# 设置窗口大小
turtle.setup(800, 600)
# 设置画笔速度
turtle.speed(5)


def draw_circle(radius, color):
    turtle.fillcolor(color)
    turtle.begin_fill()
    turtle.circle(radius)
    turtle.end_fill()


def draw_oval(radius_x, radius_y, color):
    turtle.fillcolor(color)
    turtle.begin_fill()
    for i in range(2):
        turtle.circle(radius_x, 90)
        turtle.circle(radius_y, 90)
    turtle.end_fill()


def draw_olympic_ring(x, y, color):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    turtle.pensize(10)
    turtle.color(color)
    turtle.circle(50)


def draw_bingdundun():
    # 绘制身体
    turtle.penup()
    turtle.goto(0, -200)
    turtle.pendown()
    draw_circle(200, 'white')

    # 绘制耳朵
    turtle.penup()
    turtle.goto(-100, 50)
    turtle.pendown()
    draw_circle(70, 'black')
    turtle.penup()
    turtle.goto(100, 50)
    turtle.pendown()
    draw_circle(70, 'black')

    # 绘制眼睛
    turtle.penup()
    turtle.goto(-50, 100)
    turtle.pendown()
    draw_circle(30, 'black')
    turtle.penup()
    turtle.goto(50, 100)
    turtle.pendown()
    draw_circle(30, 'black')
    turtle.penup()
    turtle.goto(-40, 110)
    turtle.pendown()
    draw_circle(15, 'white')
    turtle.penup()
    turtle.goto(60, 110)
    turtle.pendown()
    draw_circle(15, 'white')

    # 绘制鼻子
    turtle.penup()
    turtle.goto(0, 70)
    turtle.pendown()
    draw_circle(20, 'black')

    # 绘制嘴巴
    turtle.penup()
    turtle.goto(-60, 30)
    turtle.pendown()
    turtle.right(45)
    turtle.circle(80, 90)

    # 绘制手
    turtle.penup()
    turtle.goto(-180, -50)
    turtle.pendown()
    draw_oval(50, 20, 'black')
    turtle.penup()
    turtle.goto(180, -50)
    turtle.pendown()
    draw_oval(50, 20, 'black')


def main():
    # 绘制奥运五环
    draw_olympic_ring(-120, 100, 'blue')
    draw_olympic_ring(0, 100, 'black')
    draw_olympic_ring(120, 100, 'red')
    draw_olympic_ring(-60, 50, 'yellow')
    draw_olympic_ring(60, 50, 'green')
    # 绘制冰墩墩
    draw_bingdundun()
    # 隐藏画笔
    turtle.hideturtle()
    # 保持窗口显示
    turtle.done()


if __name__ == "__main__":
    main()