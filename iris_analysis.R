# ตัวอย่าง Linear Regression
data(mtcars)   # dataset ตัวอย่างที่มีใน R
model <- lm(mpg ~ wt + hp, data = mtcars)
summary(model)
