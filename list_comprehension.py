#list comprehension:
# 列表推导式，列表解析式
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)


#使用列表推导式完成上述功能如下：
squares = [x ** 2 for x in nums]
print(squares)

#list comprehensions can also contain conditions:
even_squares = [ x ** 2 for x in nums if x % 2 == 0]
print(even_squares)